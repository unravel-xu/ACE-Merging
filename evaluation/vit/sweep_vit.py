"""Validation-selected, resumable FP32 ACE sweeps for the eight-task ViT setup.

SUN397 has no prepared validation split and never enters model selection.
Images are preprocessed once in CPU memory, without changing their precision.
"""

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from reproduce_vit import (
    ROOT, TASKS, ClassificationHead, ImageClassifier, MergeStrategy,
    encoder_from_state, get_dataset, load_head, sha256, write_json,
)

VAL_TASKS = [task for task in TASKS if task != 'SUN397']
EXPECTED_TEST_COUNTS = dict(zip(TASKS, [10000, 8041, 1880, 2700, 12630, 6300, 19850, 26032]))


def candidate_name(config):
    return 'eps_{eps:g}_scale_{scaling_coefficient:g}_tau_{tau:g}_k_{k_frac:g}'.format(**config)


def sample_pair(sample):
    if isinstance(sample, dict):
        return sample['images'], int(sample['labels'])
    return sample[0], int(sample[1])


def cache_dataset(dataset, workers, name):
    """Preserve dataset order and the original deterministic float32 transform."""
    started = time.perf_counter()
    first_image, _ = sample_pair(dataset[0])
    assert first_image.dtype == torch.float32
    images = torch.empty((len(dataset), *first_image.shape), dtype=torch.float32)
    labels = torch.empty(len(dataset), dtype=torch.long)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(dataset), 256):
            stop = min(start + 256, len(dataset))
            for index, sample in zip(range(start, stop), pool.map(dataset.__getitem__, range(start, stop))):
                image, label = sample_pair(sample)
                images[index].copy_(image)
                labels[index] = label
            if stop % 4096 == 0 or stop == len(dataset):
                print(f'CACHE {name}: {stop}/{len(dataset)}', flush=True)
    # Independently check several threaded results against ordinary indexing.
    for index in {0, len(dataset) // 2, len(dataset) - 1}:
        image, label = sample_pair(dataset[index])
        assert torch.equal(image, images[index]) and label == labels[index].item()
    print(f'CACHED {name}: {len(dataset)} images, {time.perf_counter() - started:.1f}s', flush=True)
    return images, labels


def raw_dataset(args, preprocess, task, split):
    if (task == 'SUN397' and split == 'test'
            and (args.data_root / 'sun397/test_manifest.json').exists()):
        return ImageFolder(str(args.data_root / 'sun397/val'), transform=preprocess)
    key = task if split == 'test' else task + 'Val'
    return get_dataset(key, preprocess, str(args.data_root), batch_size=args.batch_size,
                       num_workers=0).test_loader.dataset


def evaluate_cached(encoder, head, cached, args, target, key, head_info):
    images, labels = cached
    predictions = torch.empty(len(labels), dtype=torch.long)
    started = time.perf_counter()
    model = ImageClassifier(encoder, head).eval()
    with torch.inference_mode():
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            logits = model(images[start:stop].to(args.device))
            assert torch.isfinite(logits).all().item(), key
            predictions[start:stop] = logits.argmax(dim=1).cpu()
    correct = (predictions == labels).sum().item()
    prediction_path = target / f'{key}_predictions.json'
    write_json(prediction_path, {'columns': ['label', 'prediction'],
                                'rows': torch.stack((labels, predictions), dim=1).tolist()})
    score = {'accuracy': correct / len(labels), 'correct': correct, 'total': len(labels),
             'seconds': time.perf_counter() - started, 'head': head_info,
             'predictions_sha256': sha256(prediction_path),
             'evaluation_source_sha256': sha256(Path(__file__))}
    print(f'{target.name} {key}: {correct}/{len(labels)} = {100 * score["accuracy"]:.4f}%', flush=True)
    return score


def update_result(result):
    scores = result['scores']
    result['complete'] = all(task in scores for task in TASKS)
    if result['complete']:
        result['average_test_accuracy'] = sum(scores[task]['accuracy'] for task in TASKS) / len(TASKS)
    if all(task + 'Val' in scores for task in VAL_TASKS):
        result['validation_macro_accuracy'] = sum(scores[task + 'Val']['accuracy'] for task in VAL_TASKS) / len(VAL_TASKS)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'data/models/vit/checkpoints')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--preprocess-workers', type=int, default=8)
    args = parser.parse_args()
    for name in ['data_root', 'output', 'reference', 'checkpoints']:
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    reference_manifest = json.loads((args.reference / 'merge_manifest.json').read_text())
    reference_results = json.loads((args.reference / 'results.json').read_text())
    sources = {name: sha256(ROOT / name) for name in [
        'src/merge/strategy.py', 'src/models/modeling.py', 'src/datasets/registry.py',
        'evaluation/vit/reproduce_vit.py', 'evaluation/vit/sweep_vit.py']}
    plan = {
        'selection_tasks': VAL_TASKS, 'selection_metric': 'unweighted mean over seven complete validation sets',
        'excluded_from_selection': ['SUN397: no prepared validation split'],
        'eps_grid': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'scale_grid': [0.8, 0.9, 1.0, 1.1, 1.2],
        'adaptive_tau_grid': [0.1, 0.03, 0.01], 'adaptive_k_grid': [0.1, 0.3, 0.5],
        'scale_boundary_extension': 'If stage-one winner is at 0.8 or 1.2, add [0.6,0.7] or [1.3,1.4] for its epsilon.',
        'adaptive_scale_refinement': 'If adaptive winner improves validation, also try scale +/-0.1 and +/-0.2.',
        'full_test_rule': 'Top two distinct validation-ranked configurations, plus source defaults eps=0.01/scale=1/tau=0.3/k=0.3.',
        'activation': 'QuickGELU', 'precision': 'float32; autocast and TF32 disabled',
        'reference': str(args.reference), 'data_root': str(args.data_root),
        'batch_size': args.batch_size, 'source_sha256': sources,
        'data_manifest_sha256': sha256(args.data_root / 'preparation_manifest.json'),
    }
    plan_path = args.output / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Sweep plan changed: use a fresh output directory'
    else:
        write_json(plan_path, plan)

    checkpoint_root = args.checkpoints / args.model
    base_path = checkpoint_root / 'MNISTVal/nonlinear_zeroshot.pt'
    assert sha256(base_path) == reference_manifest['inputs']['base']['sha256']
    base_state = torch.load(base_path, map_location=args.device, weights_only=True)
    encoder = encoder_from_state(args.model, base_state, args.device, quick_gelu=True)
    names = tuple(dict(encoder.named_parameters()))
    vectors = []
    for task in TASKS:
        path = checkpoint_root / f'{task}Val/nonlinear_finetuned.pt'
        assert sha256(path) == reference_manifest['inputs'][task]['sha256']
        state = torch.load(path, map_location=args.device, weights_only=True)
        assert state.keys() == base_state.keys()
        vectors.append(SimpleNamespace(task_vector_param_dict={key: state[key].float() - base_state[key].float() for key in names}))
        del state
    method = MergeStrategy(names, vectors, args.device)
    heads = {task: load_head(args, task) for task in TASKS}

    # Check the newly exposed threshold preserves the previous exact weights.
    baseline_config = dict(eps=1e-5, scaling_coefficient=1.0, tau=0.3, k_frac=0.3)
    baseline_delta = method.ace_merging(eps=1e-5, k_frac=0.3, tau=0.3)
    old_state = torch.load(args.reference / 'merged_state.pt', map_location=args.device, weights_only=True)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            parameter.copy_(base_state[name]).add_(baseline_delta[name])
        assert all(torch.equal(value, old_state[name]) for name, value in encoder.state_dict().items())
    del old_state
    equivalence = {'default_weights_match_reference_exactly': True, 'k_frac_checks_at_tau_0.3': {}}
    for k in [0.1, 0.5, 0.9]:
        other = method.ace_merging(eps=1e-5, k_frac=k, tau=0.3)
        equal = all(torch.equal(other[name], baseline_delta[name]) for name in names)
        assert equal
        equivalence['k_frac_checks_at_tau_0.3'][str(k)] = equal
    del other
    gammas = json.loads((args.reference / 'heterogeneity.json').read_text())['layers']
    equivalence['active_matrix_layers_by_tau'] = {
        str(tau): sum(row['gamma'] > tau for row in gammas) for tau in [0.3, 0.1, 0.03, 0.01]}
    write_json(args.output / 'equivalence_checks.json', equivalence)
    print(json.dumps(equivalence), flush=True)

    cached_val = {task: cache_dataset(raw_dataset(args, encoder.val_preprocess, task, 'val'),
                                     args.preprocess_workers, task + 'Val') for task in VAL_TASKS}
    torch.set_num_threads(4)
    records = {}

    def table():
        ranking = sorted(records.values(), key=lambda row: row['validation_macro_accuracy'], reverse=True)
        write_json(args.output / 'validation_ranking.json', ranking)
        columns = ['name', 'stage', 'eps', 'scaling_coefficient', 'tau', 'k_frac', 'validation_percent'] + [t + 'Val' for t in VAL_TASKS]
        with (args.output / 'validation_ranking.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in ranking:
                writer.writerow({**{k: row[k] for k in ['name', 'stage']}, **row['config'],
                                 'validation_percent': row['validation_macro_accuracy'] * 100,
                                 **{task + 'Val': row['scores'][task + 'Val']['accuracy'] * 100 for task in VAL_TASKS}})
        return ranking

    def run_candidate(config, stage, delta=None):
        name = candidate_name(config)
        if name in records:
            return records[name]
        target = args.output / name
        target.mkdir(exist_ok=True)
        result_path = target / 'results.json'
        manifest_path = target / 'merge_manifest.json'
        model_path = target / 'merged_state.pt'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest['merge'] == {'method': 'ace', **config}
            assert manifest['source_sha256'] == sources
            assert sha256(model_path) == manifest['merged_state_sha256']
            encoder.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True), strict=True)
        else:
            if delta is None:
                delta = method.ace_merging(eps=config['eps'], k_frac=config['k_frac'], tau=config['tau'])
            assert all(torch.isfinite(value).all().item() for value in delta.values())
            with torch.no_grad():
                for param_name, parameter in encoder.named_parameters():
                    parameter.copy_(base_state[param_name]).add_(delta[param_name], alpha=config['scaling_coefficient'])
            torch.save({key: value.cpu() for key, value in encoder.state_dict().items()}, model_path)
            manifest = {**reference_manifest, 'merge': {'method': 'ace', **config}, 'quick_gelu': True,
                        'merged_state_sha256': sha256(model_path), 'source_sha256': sources,
                        'sweep_plan': str(plan_path)}
            manifest.pop('merge_seconds', None)
            write_json(manifest_path, manifest)
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            'model': args.model, 'merge': manifest['merge'], 'tasks_expected': TASKS,
            'validation_tasks': VAL_TASKS, 'data_root': str(args.data_root),
            'scores': {}, 'complete': False, 'activation': 'quickgelu',
            'batch_size': args.batch_size, 'merged_state_sha256': manifest['merged_state_sha256']}
        assert result['merged_state_sha256'] == manifest['merged_state_sha256']
        for task in VAL_TASKS:
            key = task + 'Val'
            if key not in result['scores']:
                head, provenance = heads[task]
                result['scores'][key] = evaluate_cached(encoder, head, cached_val[task], args, target, key, provenance)
                update_result(result)
                write_json(result_path, result)
        row = {'name': name, 'stage': stage, 'config': config, 'path': str(target),
               'validation_macro_accuracy': result['validation_macro_accuracy'], 'scores': result['scores']}
        records[name] = row
        ranked = table()
        print(f'CANDIDATE {len(records)} COMPLETE {name}: val7={100*row["validation_macro_accuracy"]:.4f}%; best={ranked[0]["name"]}', flush=True)
        return row

    baseline = run_candidate(baseline_config, 'reference', baseline_delta)
    for task in VAL_TASKS:
        key = task + 'Val'
        assert baseline['scores'][key]['correct'] == reference_results['scores'][key]['correct'], key
        assert baseline['scores'][key]['total'] == reference_results['scores'][key]['total'], key
        actual = json.loads((Path(baseline['path']) / f'{key}_predictions.json').read_text())['rows']
        expected = json.loads((args.reference / f'{key}_predictions.json').read_text())['rows']
        assert actual == expected, f'Baseline prediction mismatch: {key}'
    equivalence['all_seven_validation_predictions_match_reference_exactly'] = True
    write_json(args.output / 'equivalence_checks.json', equivalence)

    for eps in plan['eps_grid']:
        delta = baseline_delta if eps == 1e-5 else method.ace_merging(eps=eps, k_frac=0.3, tau=0.3)
        for scale in plan['scale_grid']:
            run_candidate(dict(eps=eps, scaling_coefficient=scale, tau=0.3, k_frac=0.3), 'eps_scale_grid', delta)
    stage_one_best = table()[0]
    scale = stage_one_best['config']['scaling_coefficient']
    extra_scales = [0.6, 0.7] if scale == 0.8 else [1.3, 1.4] if scale == 1.2 else []
    for scale in extra_scales:
        run_candidate({**stage_one_best['config'], 'scaling_coefficient': scale}, 'scale_boundary_extension')
    stage_one_best = table()[0]
    for tau in plan['adaptive_tau_grid']:
        for k in plan['adaptive_k_grid']:
            run_candidate({**stage_one_best['config'], 'tau': tau, 'k_frac': k}, 'adaptive_grid')
    adaptive_best = table()[0]
    if adaptive_best['validation_macro_accuracy'] > stage_one_best['validation_macro_accuracy']:
        for offset in [-0.2, -0.1, 0.1, 0.2]:
            scale = round(adaptive_best['config']['scaling_coefficient'] + offset, 4)
            if scale > 0:
                run_candidate({**adaptive_best['config'], 'scaling_coefficient': scale}, 'adaptive_scale_refinement')

    ranking = table()
    finalists = [row['name'] for row in ranking[:2]]
    source_default_name = candidate_name(dict(eps=1e-2, scaling_coefficient=1.0, tau=0.3, k_frac=0.3))
    if source_default_name not in finalists:
        finalists.append(source_default_name)
    selection = {'selected_before_test': True, 'winner': ranking[0]['name'], 'finalists': finalists,
                 'selection_metric': plan['selection_metric'], 'candidates_evaluated': len(records),
                 'reference_validation_percent': baseline['validation_macro_accuracy'] * 100,
                 'winner_validation_percent': ranking[0]['validation_macro_accuracy'] * 100}
    write_json(args.output / 'selection.json', selection)
    print('FROZEN SELECTION ' + json.dumps(selection), flush=True)
    del cached_val, vectors, method, baseline_delta, delta, base_state
    torch.cuda.empty_cache()

    for task in TASKS:
        pending = []
        for name in finalists:
            target = args.output / name
            result = json.loads((target / 'results.json').read_text())
            if task not in result['scores']:
                pending.append((target, result))
        if not pending:
            continue
        torch.set_num_threads(1)
        cached = cache_dataset(raw_dataset(args, encoder.val_preprocess, task, 'test'), args.preprocess_workers, task)
        assert len(cached[1]) == EXPECTED_TEST_COUNTS[task]
        torch.set_num_threads(4)
        for target, result in pending:
            encoder.load_state_dict(torch.load(target / 'merged_state.pt', map_location=args.device, weights_only=True), strict=True)
            head, provenance = heads[task]
            result['scores'][task] = evaluate_cached(encoder, head, cached, args, target, task, provenance)
            update_result(result)
            write_json(target / 'results.json', result)
        del cached

    summary = {'selection': selection, 'reference_test_percent': reference_results['average_test_accuracy'] * 100,
               'equivalence_checks': equivalence, 'finalists': [],
               'limitations': ['Selection uses seven validation tasks; SUN397 is not represented.',
                               'EuroSAT and SUN397 split identity with the paper is still unverified.',
                               'SVHN reuses the identical-digit-template MNIST head.',
                               'DTD test is official test1, unlike the existing repository directory val1.']}
    for name in finalists:
        target = args.output / name
        result = json.loads((target / 'results.json').read_text())
        assert result['complete']
        for key, score in result['scores'].items():
            prediction_path = target / f'{key}_predictions.json'
            assert sha256(prediction_path) == score['predictions_sha256']
            pairs = json.loads(prediction_path.read_text())['rows']
            assert len(pairs) == score['total']
            assert sum(label == prediction for label, prediction in pairs) == score['correct']
        summary['finalists'].append({'name': name, 'config': records[name]['config'],
                                    'validation_percent': result['validation_macro_accuracy'] * 100,
                                    'test_percent': result['average_test_accuracy'] * 100,
                                    'total_test_examples': sum(result['scores'][task]['total'] for task in TASKS),
                                    'per_task_percent': {task: result['scores'][task]['accuracy'] * 100 for task in TASKS},
                                    'prediction_audit_passed': True, 'path': str(target)})
    write_json(args.output / 'summary.json', summary)
    print('SWEEP COMPLETE ' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
