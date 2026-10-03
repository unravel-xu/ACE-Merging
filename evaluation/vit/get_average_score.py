"""Compute task means from eval_vit.py JSON without accepting partial test suites."""

import argparse
import json

ALL_DATASETS = [
    'MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN',
    'PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'FER2013',
    'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST', 'FashionMNIST', 'KMNIST',
]
DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def compute_average(dataset_list, filepath):
    with open(filepath) as handle:
        data = json.load(handle)
    missing = [name for name in dataset_list if name not in data]
    if missing:
        raise ValueError(f'Incomplete test suite; missing tasks: {", ".join(missing)}')
    test_mean = sum(data[name] for name in dataset_list) / len(dataset_list)
    print(f'test: {100 * test_mean:.4f}% ({len(dataset_list)} tasks)')
    val_names = [f'{name}Val' for name in dataset_list]
    val_mean = None
    if all(name in data for name in val_names):
        val_mean = sum(data[name] for name in val_names) / len(val_names)
        print(f'val: {100 * val_mean:.4f}% ({len(dataset_list)} tasks)')
    return test_mean, val_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--task-count', type=int, choices=[8, 14, 20], default=8)
    args = parser.parse_args()
    compute_average(ALL_DATASETS[:args.task_count], args.results)


if __name__ == '__main__':
    main()
