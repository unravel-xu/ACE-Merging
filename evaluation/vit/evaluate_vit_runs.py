"""Evaluate several merged ViTs on identical complete, cached FP32 test images."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from extended_vit_data import CLASS_COUNTS, TEST_COUNTS, extended_test_dataset
from reproduce_vit import ALL_TASKS, ROOT, TASKS, TASK_SETS, encoder_from_state, load_head, sha256, write_json
from sweep_vit import EXPECTED_TEST_COUNTS, cache_dataset, evaluate_cached, raw_dataset


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--outputs', nargs='+', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=ALL_TASKS, default=ALL_TASKS)
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'data/models/vit/checkpoints')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--preprocess-workers', type=int, default=8)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoints = args.checkpoints.resolve()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifests = {}
    for output in args.outputs:
        output = output.resolve()
        manifest = json.loads((output / 'merge_manifest.json').read_text())
        assert manifest['model'] == args.model and manifest['quick_gelu']
        assert manifest['tasks'] == TASK_SETS[len(manifest['tasks'])]
        assert sha256(output / 'merged_state.pt') == manifest['merged_state_sha256']
        manifests[output] = manifest
    first_output = next(iter(manifests))
    state = torch.load(first_output / 'merged_state.pt', map_location='cpu', weights_only=True)
    encoder = encoder_from_state(args.model, state, args.device, quick_gelu=True)
    del state
    source_hashes = {name: sha256(ROOT / name) for name in [
        'evaluation/vit/evaluate_vit_runs.py', 'evaluation/vit/extended_vit_data.py',
        'evaluation/vit/reproduce_vit.py', 'evaluation/vit/sweep_vit.py',
        'src/models/modeling.py', 'src/datasets/registry.py']}

    for task in args.datasets:
        pending = []
        for output, manifest in manifests.items():
            if task not in manifest['tasks']:
                continue
            result_path = output / 'results.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                'model': args.model, 'merge': manifest['merge'], 'tasks_expected': manifest['tasks'],
                'data_root': str(args.data_root), 'activation': 'quickgelu', 'scores': {}, 'complete': False,
                'batch_size': args.batch_size, 'preprocess_workers': args.preprocess_workers,
                'merged_state_sha256': manifest['merged_state_sha256'], 'evaluation_source_sha256': source_hashes,
            }
            assert result['data_root'] == str(args.data_root)
            assert result['merged_state_sha256'] == manifest['merged_state_sha256']
            assert result['evaluation_source_sha256'] == source_hashes
            if task not in result['scores']:
                pending.append((output, result))
        if not pending:
            continue
        head, head_info = load_head(args, task)
        torch.set_num_threads(1)
        if task in TASKS:
            dataset = raw_dataset(args, encoder.val_preprocess, task, 'test')
            expected_count = EXPECTED_TEST_COUNTS[task]
        else:
            dataset = extended_test_dataset(task, args.data_root, encoder.val_preprocess)
            expected_count = TEST_COUNTS[task]
            assert CLASS_COUNTS[task] == head.out_features
        cached = cache_dataset(dataset, args.preprocess_workers, task)
        assert len(cached[1]) == expected_count
        assert cached[1].min().item() >= 0 and cached[1].max().item() < head.out_features
        del dataset
        torch.set_num_threads(4)
        for output, result in pending:
            state = torch.load(output / 'merged_state.pt', map_location=args.device, weights_only=True)
            encoder.load_state_dict(state, strict=True)
            del state
            score = evaluate_cached(encoder, head, cached, args, output, task, head_info)
            score['test_population'] = expected_count
            result['scores'][task] = score
            tasks = result['tasks_expected']
            result['complete'] = all(name in result['scores'] for name in tasks)
            if result['complete']:
                result['average_test_accuracy'] = sum(result['scores'][name]['accuracy'] for name in tasks) / len(tasks)
            write_json(output / 'results.json', result)
        del cached, head
    for output, manifest in manifests.items():
        result_path = output / 'results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {'scores': {}}
        print(json.dumps({'output': str(output), 'missing_test_tasks': [t for t in manifest['tasks'] if t not in result['scores']],
                          'average_test_accuracy': result.get('average_test_accuracy')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
