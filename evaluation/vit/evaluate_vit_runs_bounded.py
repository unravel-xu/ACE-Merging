"""Continue full FP32 evaluations with a disk cache inside a 32-GB container.

The original evaluator and its source hashes are preserved. Continuations record
this file's hash separately, including the tasks evaluated with the disk cache.
"""
import hashlib
import json
import mmap
import tempfile
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import evaluate_vit_runs as original
from reproduce_vit import ROOT, ImageClassifier, sha256, write_json
from sweep_vit import sample_pair

MAPPINGS = {}
CACHE_ROOT = ROOT / 'data/reproduction_assets/fp32_disk_cache'
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))


def release_mapping(identity, mapping, path):
    MAPPINGS.pop(identity, None)
    mapping._mmap.close()
    path.unlink(missing_ok=True)


def cache_dataset(dataset, workers, name):
    started = time.perf_counter()
    first, _ = sample_pair(dataset[0])
    assert first.dtype == torch.float32
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=name + '-', suffix='.fp32', dir=CACHE_ROOT, delete=False)
    path = Path(handle.name)
    handle.close()
    mapping = np.memmap(path, mode='w+', dtype=np.float32, shape=(len(dataset), *first.shape))
    images = torch.from_numpy(mapping)
    labels = torch.empty(len(dataset), dtype=torch.long)
    image_bytes = first.numel() * first.element_size()
    assert image_bytes % mmap.PAGESIZE == 0
    identity = id(images)
    MAPPINGS[identity] = (mapping, path, image_bytes)
    weakref.finalize(images, release_mapping, identity, mapping, path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(dataset), 256):
            stop = min(start + 256, len(dataset))
            for index, sample in zip(range(start, stop), pool.map(dataset.__getitem__, range(start, stop))):
                image, label = sample_pair(sample)
                images[index].copy_(image)
                labels[index] = label
            mapping._mmap.flush(start * image_bytes, (stop - start) * image_bytes)
            mapping._mmap.madvise(mmap.MADV_DONTNEED, start * image_bytes, (stop - start) * image_bytes)
            if stop % 4096 == 0 or stop == len(dataset):
                print(f'DISK CACHE {name}: {stop}/{len(dataset)}', flush=True)
    for index in {0, len(dataset) // 2, len(dataset) - 1}:
        image, label = sample_pair(dataset[index])
        assert torch.equal(image, images[index]) and label == labels[index].item()
    mapping._mmap.madvise(mmap.MADV_DONTNEED)
    print(f'DISK CACHED {name}: {len(dataset)} FP32 images, {time.perf_counter() - started:.1f}s', flush=True)
    return images, labels


def evaluate_cached(encoder, head, cached, args, target, key, head_info):
    images, labels = cached
    mapping, _, image_bytes = MAPPINGS[id(images)]
    predictions = torch.empty(len(labels), dtype=torch.long)
    started = time.perf_counter()
    model = ImageClassifier(encoder, head).eval()
    with torch.inference_mode():
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            logits = model(images[start:stop].to(args.device))
            assert torch.isfinite(logits).all().item(), key
            predictions[start:stop] = logits.argmax(dim=1).cpu()
            mapping._mmap.madvise(mmap.MADV_DONTNEED, start * image_bytes, (stop - start) * image_bytes)
            if stop % 4096 == 0 or stop == len(labels):
                print(f'{target.name} {key}: evaluated {stop}/{len(labels)}', flush=True)
    correct = (predictions == labels).sum().item()
    prediction_path = target / f'{key}_predictions.json'
    write_json(prediction_path, {'columns': ['label', 'prediction'],
                                'rows': torch.stack((labels, predictions), dim=1).tolist()})
    score = {'accuracy': correct / len(labels), 'correct': correct, 'total': len(labels),
             'seconds': time.perf_counter() - started, 'head': head_info,
             'predictions_sha256': sha256(prediction_path),
             'evaluation_source_sha256': sha256(Path(__file__)),
             'cache_storage': 'disk-backed float32; transforms and image order unchanged'}
    print(f'{target.name} {key}: {correct}/{len(labels)} = {100 * score["accuracy"]:.4f}%', flush=True)
    return score


def record_continuation(path, result):
    if path.name == 'results.json':
        digest = sha256(Path(__file__))
        result['evaluation_continuation'] = {
            'source_sha256': {SOURCE: digest},
            'tasks': [t for t, s in result['scores'].items() if s['evaluation_source_sha256'] == digest],
            'precision': 'FP32; no autocast or TF32; disk storage changes memory use only'}
    write_json(path, result)


if __name__ == '__main__':
    original.cache_dataset = cache_dataset
    original.evaluate_cached = evaluate_cached
    original.write_json = record_continuation
    original.main()
