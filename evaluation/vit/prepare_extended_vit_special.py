"""Download pinned FER2013, Food101, EMNIST, and official PCAM test assets."""

import argparse
import gzip
import hashlib
import io
import json
import pickle
import shutil
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np
from PIL import Image

from download_reproduction_assets import fetch, get_json, sha256

ROOT = Path(__file__).resolve().parents[2]


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Unexpected pickle global: {module}.{name}')


def official_pcam(item):
    path = Path(item['destination'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    def valid(p):
        if not p.exists() or p.stat().st_size != item['size']:
            return False
        h = hashlib.md5()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest() == item['md5']
    if not valid(path):
        for attempt in range(8):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset >= item['size']:
                    if valid(temporary):
                        temporary.replace(path)
                        break
                    offset = 0
                request = urllib.request.Request(item['url'] + '&attempt=' + str(time.time_ns()),
                                                  headers={'Range': f'bytes={offset}-'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        offset = 0
                    else:
                        assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    with temporary.open('ab' if offset else 'wb') as output:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                            output.write(chunk)
                assert valid(temporary), 'Incomplete download or MD5 mismatch'
                temporary.replace(path)
                break
            except Exception as error:
                print('RETRY PCAM', path.name, attempt + 1, repr(error), flush=True)
        else:
            raise RuntimeError(f'Cannot verify {path}')
    print('VERIFIED PCAM', path.name, flush=True)
    return {**item, 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/datasets/vit_extended')
    parser.add_argument('--assets', type=Path, default=ROOT / 'data/reproduction_assets/extended')
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    specs = [('FER2013', 'Jeneral/fer-2013', lambda p: p == 'test.pt'),
             ('Food101', 'ethz/food101', lambda p: p.startswith('data/validation-') and p.endswith('.parquet')),
             ('EMNIST', 'tanganke/emnist_digits', lambda p: p.startswith('raw/') and p.endswith('.gz')),
             ('KMNIST', 'tanganke/kmnist', lambda p: p.endswith('.parquet'))]
    items = []
    for name, repo, selected in specs:
        revision = get_json(f'https://huggingface.co/api/datasets/{repo}')['sha']
        files = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true')
        for file in files:
            if file['type'] == 'file' and selected(file['path']):
                items.append({'dataset': name, 'repository': repo, 'revision': revision,
                              'url': f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{file["path"]}',
                              'size': file['size'], 'sha256': file['lfs']['oid'],
                              'destination': str(args.assets / name / Path(file['path']).name)})
    pcam_files = get_json('https://zenodo.org/api/records/2546921')['files']
    for file in pcam_files:
        if file['key'] in ['camelyonpatch_level_2_split_test_x.h5.gz', 'camelyonpatch_level_2_split_test_y.h5.gz']:
            items.append({'dataset': 'PCAM', 'source': 'Zenodo record 2546921; matches torchvision 0.19 test hashes',
                          'url': f'https://zenodo.org/records/2546921/files/{file["key"]}?download=1',
                          'size': file['size'], 'md5': file['checksum'].split(':')[1],
                          'destination': str(args.assets / 'PCAM' / file['key'])})
    (args.assets / 'download_plan.json').write_text(json.dumps(items, indent=2) + '\n')
    print('DOWNLOAD', len(items), 'files', sum(x['size'] for x in items) / 1e9, 'GB', flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        verified = list(pool.map(lambda item: official_pcam(item) if item['dataset'] == 'PCAM' else fetch(item), items))
    (args.assets / 'verified_downloads.json').write_text(json.dumps(verified, indent=2) + '\n')

    for name in ['PCAM', 'EMNIST']:
        destination = args.root / ('PCAM/pcam' if name == 'PCAM' else 'EMNIST/EMNIST/raw')
        destination.mkdir(parents=True, exist_ok=True)
        for item in [item for item in verified if item['dataset'] == name]:
            source = Path(item['destination'])
            target = destination / source.stem
            if not target.exists():
                with gzip.open(source, 'rb') as input_file, target.open('wb') as output:
                    shutil.copyfileobj(input_file, output)
        print('READY', name, flush=True)

    kmnist_raw = args.root / 'KMNIST/KMNIST/raw'
    kmnist_raw.mkdir(parents=True, exist_ok=True)
    for split, prefix, count in [('train', 'train', 60000), ('test', 't10k', 10000)]:
        files = list((args.assets / 'KMNIST').glob(split + '-*.parquet'))
        assert len(files) == 1
        rows = pq.read_table(files[0]).to_pylist()
        assert len(rows) == count
        images = np.stack([np.asarray(Image.open(io.BytesIO(row['image']['bytes']))) for row in rows])
        labels = np.asarray([row['label'] for row in rows], dtype=np.uint8)
        assert images.shape == (count, 28, 28) and images.dtype == np.uint8
        assert np.bincount(labels, minlength=10).tolist() == [count // 10] * 10
        (kmnist_raw / f'{prefix}-images-idx3-ubyte').write_bytes(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes())
        (kmnist_raw / f'{prefix}-labels-idx1-ubyte').write_bytes(struct.pack('>II', 2049, count) + labels.tobytes())
    print('READY KMNIST (pinned public mirror; unchanged decoded pixels and label indices)', flush=True)

    fer_root = args.root / 'FER2013'
    (fer_root / 'test').mkdir(parents=True, exist_ok=True)
    with (args.assets / 'FER2013/test.pt').open('rb') as handle:
        examples = DataOnlyUnpickler(handle).load()
    assert len(examples) == 7178, len(examples)
    fer_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    samples = []
    for index, row in enumerate(examples):
        data, label = row['img_bytes'], row['labels']
        if isinstance(label, str):
            # Exactly the ClassLabel encoding in the original HF dataset script.
            label = fer_classes.index(label)
        assert isinstance(data, bytes) and isinstance(label, int) and 0 <= label < 7
        # Retain encoded bytes; the adapter uses exactly the original convert('L').
        relative = f'test/{index:05d}.png'
        (fer_root / relative).write_bytes(data)
        samples.append([relative, label])
    fer_manifest = {'dataset': 'Jeneral/fer-2013', 'class_names': fer_classes,
                    'convert_mode': 'L', 'samples': samples,
                    'label_policy': 'Preserve original HF sample labels exactly, as src/datasets/fer2013.py does.',
                    'note': 'The source loader classnames differ for labels 4..6; released head and checkpoint are used unchanged.'}
    (fer_root / 'test_manifest.json').write_text(json.dumps(fer_manifest, indent=2) + '\n')
    print('READY FER2013', len(samples), flush=True)

    food_root = args.root / 'Food101'
    (food_root / 'test').mkdir(parents=True, exist_ok=True)
    samples = []
    classes = None
    source_classes = None
    for file in sorted((args.assets / 'Food101').glob('*.parquet')):
        table = pq.read_table(file)
        metadata = json.loads(table.schema.metadata[b'huggingface'])
        names = metadata['info']['features']['label']['names']
        assert len(names) == len(set(names)) == 101
        assert source_classes is None or source_classes == names
        source_classes = names
        # torchvision sorts metadata keys; HF's cheesecake/cheese_plate order differs.
        classes = sorted(names)
        label_map = {index: classes.index(name) for index, name in enumerate(names)}
        for row in table.to_pylist():
            relative = f'test/{len(samples):05d}.jpg'
            (food_root / relative).write_bytes(row['image']['bytes'])
            samples.append([relative, label_map[row['label']]])
    assert len(samples) == 25250
    assert [sum(label == k for _, label in samples) for k in range(101)] == [250] * 101
    food_manifest = {'dataset': 'ethz/food101', 'class_names': classes, 'convert_mode': 'RGB', 'samples': samples,
                     'source_class_names': source_classes, 'source_to_torchvision_label_map': label_map,
                     'source_split': 'validation in HF; official Food101 test population of 250 images per class',
                     'label_policy': 'Map source class names to sorted torchvision Food101 class names, matching src/datasets/food101.py.'}
    (food_root / 'test_manifest.json').write_text(json.dumps(food_manifest, indent=2) + '\n')
    print('READY Food101', len(samples), flush=True)
    (args.root / 'special_preparation_manifest.json').write_text(json.dumps({
        'verified_assets': str(args.assets / 'verified_downloads.json'),
        'PCAM': {'test_count': 32768, 'protocol': 'Zenodo test files, matching torchvision 0.19 hashes'},
        'EMNIST': {'test_count': 40000, 'protocol': 'digits, original raw orientation unchanged'},
        'KMNIST': {'test_count': 10000, 'protocol': 'Pinned tanganke/kmnist mirror reconstructed into IDX without changing decoded pixels or labels; original server returned HTTP 403'},
        'FER2013': {'test_count': 7178, 'manifest_sha256': sha256(fer_root / 'test_manifest.json')},
        'Food101': {'test_count': 25250, 'manifest_sha256': sha256(food_root / 'test_manifest.json')},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
