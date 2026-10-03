"""Explicit, resumable ViT reproduction using the repository's ACE implementation."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import open_clip
import torch

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates
from src.merge.strategy import MergeStrategy
from src.models.modeling import ClassificationHead, ImageClassifier, ImageEncoder

TASKS = ['MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN']
ALL_TASKS = TASKS + ['PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102',
                    'FER2013', 'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST',
                    'FashionMNIST', 'KMNIST']
TASK_SETS = {count: ALL_TASKS[:count] for count in (8, 14, 20)}


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_from_state(model_name, state, device, quick_gelu=False):
    # OpenCLIP 3.3 uses GELU for the repository's ViT-B-32 constructor, even
    # with pretrained='openai'. Keep activation explicit because it is not
    # represented in a state_dict. Every parameter is replaced by local state.
    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None, force_quick_gelu=quick_gelu
    )
    delattr(model, 'transformer')
    encoder = ImageEncoder.__new__(ImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = model
    encoder.train_preprocess = train_preprocess
    encoder.val_preprocess = val_preprocess
    encoder.load_state_dict(state, strict=True)
    return encoder.float().to(device).eval()


def merge(args):
    tasks = TASK_SETS[args.task_count]
    checkpoint_root = args.checkpoints / args.model
    base_path = checkpoint_root / 'MNISTVal/nonlinear_zeroshot.pt'
    base_state = torch.load(base_path, map_location=args.device, weights_only=True)
    encoder = encoder_from_state(args.model, base_state, args.device, args.activation == 'quickgelu')
    parameter_names = tuple(dict(encoder.named_parameters()))
    task_vectors = []
    inputs = {'base': {'path': str(base_path), 'sha256': sha256(base_path)}}
    for task in tasks:
        path = checkpoint_root / f'{task}Val/nonlinear_finetuned.pt'
        state = torch.load(path, map_location=args.device, weights_only=True)
        assert state.keys() == base_state.keys(), f'Checkpoint keys differ: {task}'
        with torch.no_grad():
            delta = {key: state[key].float() - base_state[key].float() for key in parameter_names}
        task_vectors.append(SimpleNamespace(task_vector_param_dict=delta))
        inputs[task] = {'path': str(path), 'sha256': sha256(path)}
        del state
        print(f'Loaded expert {task}', flush=True)

    method = MergeStrategy(parameter_names, task_vectors, args.device)
    started = time.perf_counter()
    merged = method.ace_merging(eps=args.eps, k_frac=args.k_frac, tau=args.tau)
    assert all(torch.isfinite(value).all().item() for value in merged.values())
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            parameter.add_(merged[name], alpha=args.scaling_coefficient)
    duration = time.perf_counter() - started
    target = args.output / 'merged_state.pt'
    torch.save({key: value.cpu() for key, value in encoder.state_dict().items()}, target)
    metadata = {
        'model': args.model, 'tasks': tasks,
        'merge': {'method': 'ace', 'scaling_coefficient': args.scaling_coefficient,
                  'eps': args.eps, 'k_frac': args.k_frac, 'tau': args.tau},
        'precision': 'float32, no autocast, TF32 disabled', 'seed': 0,
        'quick_gelu': args.activation == 'quickgelu', 'merge_seconds': duration,
        'inputs': inputs, 'merged_state_sha256': sha256(target),
        'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: sha256(ROOT / name) for name in [
            'src/merge/strategy.py', 'src/models/modeling.py',
            'evaluation/vit/reproduce_vit.py']},
        'versions': {name: version(name) for name in [
            'torch', 'torchvision', 'open_clip_torch', 'numpy', 'scipy', 'transformers']},
        'python': sys.version, 'device': (torch.cuda.get_device_name(args.device)
                                        if torch.device(args.device).type == 'cuda' else args.device),
    }
    write_json(args.output / 'merge_manifest.json', metadata)
    print(f'Merged all {len(tasks)} experts in {duration:.2f}s; saved {target}', flush=True)


def load_head(args, task):
    path = args.checkpoints / args.model / f'head_{task}Val.pt'
    provenance = 'released checkpoint'
    if task == 'SVHN' and not path.exists():
        # Both datasets use labels 0..9 and exactly the same text prompts.
        assert [f(str(i)) for i in range(10) for f in get_templates('MNIST')] == [
            f(str(i)) for i in range(10) for f in get_templates('SVHN')]
        path = args.checkpoints / args.model / 'head_MNISTVal.pt'
        provenance = 'MNIST zero-shot head: identical digit class order and text templates to SVHN'
    head = ClassificationHead.load(str(path), args.device).float().eval()
    return head, {'path': str(path), 'sha256': sha256(path), 'provenance': provenance}


def evaluate(args):
    manifest = json.loads((args.output / 'merge_manifest.json').read_text())
    assert manifest['model'] == args.model
    tasks = manifest['tasks']
    assert tasks == TASK_SETS[args.task_count]
    datasets = args.datasets if args.datasets is not None else tasks
    assert set(datasets).issubset(tasks), 'Evaluation tasks must belong to the merged task set'
    state_path = args.output / 'merged_state.pt'
    assert sha256(state_path) == manifest['merged_state_sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    encoder = encoder_from_state(args.model, state, args.device, manifest['quick_gelu'])
    del state
    result_path = args.output / 'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {
        'model': args.model, 'merge': manifest['merge'], 'tasks_expected': tasks,
        'data_root': str(args.data_root), 'scores': {}, 'complete': False,
        'batch_size': args.batch_size, 'num_workers': args.workers,
        'merged_state_sha256': manifest['merged_state_sha256'],
        'activation': 'quickgelu' if manifest['quick_gelu'] else 'gelu',
    }
    assert results['data_root'] == str(args.data_root), 'Use a new output directory for a different dataset root'
    assert results['merged_state_sha256'] == manifest['merged_state_sha256']
    for task in datasets:
        head, head_info = load_head(args, task)
        model = ImageClassifier(encoder, head).eval()
        for split in args.splits:
            key = task if split == 'test' else f'{task}Val'
            if key in results['scores']:
                print(f'Resume: already evaluated {key}', flush=True)
                continue
            test_manifest_path = args.data_root / 'sun397' / 'test_manifest.json'
            if task == 'SUN397' and test_manifest_path.exists():
                if split != 'test':
                    print('SUN397 mirror contains test images only; skipping validation.', flush=True)
                    continue
                from torchvision.datasets import ImageFolder
                test_dataset = ImageFolder(str(args.data_root / 'sun397/val'), transform=model.val_preprocess)
                dataset = SimpleNamespace(
                    classnames=test_dataset.classes,
                    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                           shuffle=False, num_workers=args.workers),
                )
            elif split == 'test' and task in ALL_TASKS[8:]:
                from extended_vit_data import CLASS_COUNTS, extended_test_dataset
                test_dataset = extended_test_dataset(task, args.data_root, model.val_preprocess)
                dataset = SimpleNamespace(
                    classnames=list(range(CLASS_COUNTS[task])),
                    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                           shuffle=False, num_workers=args.workers),
                )
            else:
                dataset = get_dataset(key, model.val_preprocess, location=str(args.data_root),
                                      batch_size=args.batch_size, num_workers=args.workers)
            loader = dataset.test_loader
            assert len(loader.dataset) > 0
            assert len(dataset.classnames) == head.out_features, (task, len(dataset.classnames), head.out_features)
            started = time.perf_counter()
            correct = total = 0
            predictions = []
            with torch.inference_mode():
                for index, batch in enumerate(loader):
                    batch = maybe_dictionarize(batch)
                    images = batch['images'].to(args.device, dtype=torch.float32)
                    labels = batch['labels'].to(args.device)
                    logits = model(images)
                    assert torch.isfinite(logits).all().item(), f'Non-finite logits: {key}'
                    predicted = logits.argmax(dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.numel()
                    predictions.extend(zip(labels.cpu().tolist(), predicted.cpu().tolist()))
                    if index % 25 == 0:
                        print(f'{key}: {total}/{len(loader.dataset)} top1={100*correct/total:.4f}%', flush=True)
            assert total == len(loader.dataset), (key, total, len(loader.dataset))
            prediction_path = args.output / f'{key}_predictions.json'
            write_json(prediction_path, {'columns': ['label', 'prediction'], 'rows': predictions})
            results['scores'][key] = {
                'accuracy': correct / total, 'correct': correct, 'total': total,
                'seconds': time.perf_counter() - started, 'head': head_info,
                'predictions_sha256': sha256(prediction_path),
                'evaluation_source_sha256': sha256(Path(__file__)),
            }
            results['complete'] = all(name in results['scores'] for name in tasks)
            if results['complete']:
                results['average_test_accuracy'] = sum(results['scores'][name]['accuracy'] for name in tasks) / len(tasks)
            if all(f'{name}Val' in results['scores'] for name in tasks):
                results['average_val_accuracy'] = sum(results['scores'][f'{name}Val']['accuracy'] for name in tasks) / len(tasks)
            write_json(result_path, results)
            print(f'DONE {key}: {correct}/{total} = {100*correct/total:.4f}%', flush=True)
    missing = [name for name in tasks if name not in results['scores']]
    print(json.dumps({'missing_test_tasks': missing,
                      'average_test_accuracy': results.get('average_test_accuracy'),
                      'average_val_accuracy': results.get('average_val_accuracy')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=['merge', 'evaluate', 'both'], default='both')
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--task-count', type=int, choices=[8, 14, 20], default=8)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'data/models/vit/checkpoints')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--k-frac', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.03)
    parser.add_argument('--scaling-coefficient', type=float, default=1.0)
    parser.add_argument('--activation', choices=['gelu', 'quickgelu'], default='quickgelu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', choices=ALL_TASKS)
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.checkpoints = args.checkpoints.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.phase in ('merge', 'both'):
        if (args.output / 'merged_state.pt').exists():
            raise FileExistsError('Use a new output directory to avoid overwriting an existing merge')
        merge(args)
    if args.phase in ('evaluate', 'both'):
        evaluate(args)


if __name__ == '__main__':
    main()
