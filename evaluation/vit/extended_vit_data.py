"""Test-only adapters preserving the repository's dataset labels and transforms."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

CLASS_COUNTS = dict(PCAM=2, CIFAR100=100, STL10=10, OxfordIIITPet=37, Flowers102=102,
                    FER2013=7, CIFAR10=10, Food101=101, RenderedSST2=2, EMNIST=10,
                    FashionMNIST=10, KMNIST=10)
TEST_COUNTS = dict(PCAM=32768, CIFAR100=10000, STL10=8000, OxfordIIITPet=3669,
                   Flowers102=6149, FER2013=7178, CIFAR10=10000, Food101=25250,
                   RenderedSST2=1821, EMNIST=40000, FashionMNIST=10000, KMNIST=10000)


class ManifestImages(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'test_manifest.json').read_text())
        self.samples = self.manifest['samples']
        self.classes = self.manifest['class_names']
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative, label = self.samples[index]
        with Image.open(self.root / relative) as image:
            image = image.convert(self.manifest['convert_mode'])
        return self.transform(image), label


class PCAMTestArrays(Dataset):
    def __init__(self, root, transform):
        import h5py
        self.transform = transform
        folder = Path(root) / 'pcam'
        with h5py.File(folder / 'camelyonpatch_level_2_split_test_x.h5', 'r') as handle:
            self.images = handle['x'][:]
        with h5py.File(folder / 'camelyonpatch_level_2_split_test_y.h5', 'r') as handle:
            self.labels = handle['y'][:, 0, 0, 0]
        assert self.images.shape == (32768, 96, 96, 3)
        assert self.labels.shape == (32768,)
        assert np.array_equal(np.unique(self.labels), np.array([0, 1]))
        self.class_counts = np.bincount(self.labels, minlength=2).tolist()
        self.classes = ['lymph node', 'lymph node containing metastatic tumor tissue']
        # The in-memory adapter must match torchvision's unmodified test reader.
        original = datasets.PCAM(root=str(root), split='test', download=False)
        for index in (0, 16384, 32767):
            image, label = original[index]
            assert np.array_equal(np.asarray(image), self.images[index])
            assert label == int(self.labels[index])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index]).convert('RGB')), int(self.labels[index])


def extended_test_dataset(task, root, transform):
    root = Path(root)
    if task == 'PCAM':
        result = PCAMTestArrays(root / 'PCAM', transform)
    elif task in ('FER2013', 'Food101'):
        result = ManifestImages(root / task, transform)
    else:
        subdir = {'STL10': 'stl10', 'OxfordIIITPet': 'OxfordIIITPet', 'Flowers102': 'flowers102',
                  'RenderedSST2': 'RenderedSST2', 'EMNIST': 'EMNIST',
                  'FashionMNIST': 'FashionMNIST', 'KMNIST': 'KMNIST'}.get(task, '')
        kwargs = {'train': False} if task in ['CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'KMNIST'] else {'split': 'test'}
        if task == 'EMNIST':
            kwargs['split'] = 'digits'
        result = getattr(datasets, task)(root=str(root / subdir), download=False, transform=transform, **kwargs)
    assert len(result) == TEST_COUNTS[task], (task, len(result))
    return result
