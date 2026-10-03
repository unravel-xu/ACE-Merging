"""Prepare official torchvision datasets and record their test populations."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ['CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'CIFAR10',
            'RenderedSST2', 'FashionMNIST', 'KMNIST']
COUNTS = dict(CIFAR100=10000, STL10=8000, OxfordIIITPet=3669, Flowers102=6149,
              CIFAR10=10000, RenderedSST2=1821, FashionMNIST=10000, KMNIST=10000)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/datasets/vit_extended')
    parser.add_argument('--datasets', nargs='+', choices=STANDARD, default=STANDARD)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'data/datasets/vit_reproduction'
    for name in ['MNIST', 'stanford_cars', 'dtd', 'EuroSAT_splits', 'gtsrb', 'resisc45', 'sun397', 'svhn']:
        path = args.root / name
        if not path.exists():
            path.symlink_to(original / name, target_is_directory=True)
        assert path.resolve() == (original / name).resolve()
    # HTTPS endpoints serve the same official archives and retain MD5 checks.
    datasets.STL10.url = datasets.STL10.url.replace('http:', 'https:')
    datasets.FashionMNIST.mirrors = ['https://fashion-mnist.s3.eu-central-1.amazonaws.com/']
    datasets.KMNIST.mirrors = ['https://codh.rois.ac.jp/kmnist/dataset/kmnist/']

    def prepare(name):
        cls = getattr(datasets, name)
        subdir = {'STL10': 'stl10', 'OxfordIIITPet': 'OxfordIIITPet', 'Flowers102': 'flowers102',
                  'RenderedSST2': 'RenderedSST2', 'FashionMNIST': 'FashionMNIST', 'KMNIST': 'KMNIST'}.get(name, '')
        kwargs = {'train': False} if name in ['CIFAR10', 'CIFAR100', 'FashionMNIST', 'KMNIST'] else {'split': 'test'}
        print('START', name, flush=True)
        dataset = cls(root=str(args.root / subdir), download=True, **kwargs)
        assert len(dataset) == COUNTS[name], (name, len(dataset))
        item = {'test_count': len(dataset), 'loader': f'torchvision.datasets.{name}',
                'root': str(args.root / subdir), 'split_arguments': kwargs, 'download_integrity': 'torchvision official MD5 checks'}
        print('READY', name, len(dataset), flush=True)
        return item

    path = args.root / 'standard_preparation_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, name): name for name in args.datasets}
        for future in as_completed(futures):
            name = futures[future]
            try:
                manifest[name] = future.result()
                path.write_text(json.dumps(manifest, indent=2) + '\n')
            except Exception as error:
                failures[name] = repr(error)
                print('FAILED', name, repr(error), flush=True)
    (args.root / 'standard_preparation_failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    if failures:
        raise RuntimeError(failures)


if __name__ == '__main__':
    main()
