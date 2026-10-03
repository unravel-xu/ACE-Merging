"""Arrange downloaded benchmark assets without changing existing dataset folders."""

import argparse
import hashlib
import io
import json
import zipfile
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve() == source.resolve()
    elif not destination.exists():
        destination.symlink_to(source.resolve())
    else:
        assert destination.resolve() == source.resolve(), destination


def prepare_sun(source, target, assets):
    index = {path.name: path for path in (source / 'sun397').rglob('*.jpg')}
    manifests = {}
    with zipfile.ZipFile(assets / 'Partitions.zip') as archive:
        for name, split in [('Training_01.txt', 'train'), ('Testing_01.txt', 'val')]:
            original = archive.read(name)
            rows = original.decode().splitlines()
            assert len(rows) == 19850
            missing = [row for row in rows if Path(row).name not in index]
            assert not missing, f'SUN397 missing {len(missing)} images: {missing[:3]}'
            for row in rows:
                path = Path(row.lstrip('/'))
                classname = '_'.join(path.parts[:-1])
                link_file(index[path.name], target / 'sun397' / split / classname / path.name)
            manifests[split] = {'count': len(rows), 'partition': name,
                                'partition_sha256': digest(original), 'files': rows}
    assert set(manifests['train']['files']).isdisjoint(manifests['val']['files'])
    return {'source': 'SUN397 official partition 01, recovered by basename from existing images',
            'directory_val_means': 'official test', 'splits': manifests}


def prepare_sun_mirror(target, assets):
    files = sorted((assets / 'sun397_test').glob('test-*.parquet'))
    assert len(files) == 18, f'Need all 18 verified test shards, found {len(files)}'
    with zipfile.ZipFile(assets / 'Partitions.zip') as archive:
        classes = sorted(row.strip('/').replace('/', '_')
                         for row in archive.read('ClassName.txt').decode().splitlines())
    expected_names = [name[2:].replace('_', ' ') for name in classes]
    rows_manifest = []
    counts = [0] * 397
    for file in files:
        table = pq.read_table(file)
        metadata = json.loads(table.schema.metadata[b'huggingface'])
        assert metadata['info']['features']['label']['names'] == expected_names
        for row in table.to_pylist():
            data = row['image']['bytes']
            label = row['label']
            name = f'{classes[label]}/{digest(data)}.jpg'
            destination = target / 'sun397' / 'val' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            rows_manifest.append(name)
            counts[label] += 1
        print(f'Extracted {file.name}: {len(table)} images', flush=True)
    assert len(rows_manifest) == len(set(rows_manifest)) == 19850
    assert counts == [50] * 397
    result = {
        'source': 'https://huggingface.co/datasets/tanganke/sun397',
        'class_order': 'Verified against sorted flattened official SUN397 class directory names',
        'limitation': 'Public test split; identity with the ACE paper dataset has not been established.',
        'test_only': True,
        'splits': {'val': {'count': len(rows_manifest), 'files': sorted(rows_manifest),
                           'files_sha256': digest('\n'.join(sorted(rows_manifest)).encode())}},
    }
    (target / 'sun397' / 'test_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def prepare_dtd(source, target, assets):
    index = {str(path.relative_to(source / 'dtd')).split('/', 1)[1]: path
             for path in (source / 'dtd').rglob('*.jpg')}
    with tarfile.open(assets / 'dtd-r1.0.1.tar.gz') as archive:
        labels = {split: archive.extractfile(f'dtd/labels/{split}1.txt').read().decode().splitlines()
                  for split in ('train', 'val', 'test')}
    splits = {'train': labels['train'] + labels['val'], 'val': labels['test']}
    assert len(splits['train']) == 3760 and len(splits['val']) == 1880
    assert set(splits['train']).isdisjoint(splits['val'])
    for split, rows in splits.items():
        assert all(row in index for row in rows), f'Missing DTD images in {split}'
        for row in rows:
            link_file(index[row], target / 'dtd' / split / row)
    return {'source': 'Official DTD split 1: train1+val1 for train, test1 for val directory',
            'splits': {key: {'count': len(rows), 'files': rows,
                             'files_sha256': digest('\n'.join(rows).encode())} for key, rows in splits.items()}}


def prepare_resisc(target, assets):
    folder = target / 'resisc45'
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(assets / 'NWPU-RESISC45.zip') as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            destination = folder / relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    manifests = {}
    for split in ('train', 'val', 'test'):
        path = assets / f'resisc45-{split}.txt'
        link_file(path, folder / path.name)
        rows = path.read_text().splitlines()
        assert all((folder / 'NWPU-RESISC45' / row.rsplit('_', 1)[0] / row).exists() for row in rows)
        manifests[split] = {'count': len(rows), 'files': rows, 'files_sha256': digest(path.read_bytes())}
    assert set(manifests['train']['files']).isdisjoint(manifests['test']['files'])
    return {'source': 'https://huggingface.co/datasets/torchgeo/resisc45', 'splits': manifests}


def pixel_digest(data):
    image = Image.open(io.BytesIO(data)).convert('RGB')
    return digest(image.tobytes())


def prepare_eurosat(target, assets):
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    assigned = {}
    original = {}
    manifests = {}
    for split in ('train', 'test'):
        rows = pq.read_table(assets / f'eurosat-{split}.parquet').to_pylist()
        manifests[split] = []
        for row in rows:
            data = row['image']['bytes']
            key = pixel_digest(data)
            assert key not in assigned, 'Duplicate image across EuroSAT splits'
            assigned[key] = (split, classes[row['label']])
    # The public mirror supplies train/test. Recover the remaining 2,700
    # validation images from the original 27,000-image release by exact pixels.
    with zipfile.ZipFile(assets / 'EuroSAT.zip') as archive:
        for name in archive.namelist():
            if not name.lower().endswith('.jpg'):
                continue
            data = archive.read(name)
            path = Path(name)
            key = pixel_digest(data)
            assert key not in original, 'Duplicate original EuroSAT pixels'
            original[key] = name
            split, classname = assigned.get(key, ('val', path.parent.name))
            assert classname == path.parent.name, (name, classname)
            destination = target / 'EuroSAT_splits' / split / classname / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            manifests.setdefault(split, []).append(f'{classname}/{path.name}')
    assert set(assigned).issubset(original), 'EuroSAT mirror pixels differ from original release'
    assert {key: len(rows) for key, rows in manifests.items()} == {'train': 21600, 'test': 2700, 'val': 2700}
    return {'source': 'https://huggingface.co/datasets/tanganke/eurosat',
            'validation': 'Complement of public train/test in original EuroSAT release, matched by decoded RGB SHA256',
            'limitation': 'Identity with the ACE paper\'s unpublished EuroSAT split has not been established.',
            'splits': {key: {'count': len(rows), 'files': sorted(rows),
                             'files_sha256': digest('\n'.join(sorted(rows)).encode())} for key, rows in manifests.items()}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['DTD', 'SUN397', 'SUN397Mirror', 'RESISC45', 'EuroSAT'], required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/datasets/vit')
    parser.add_argument('--target', type=Path, default=ROOT / 'data/datasets/vit_reproduction')
    parser.add_argument('--assets', type=Path, default=ROOT / 'data/reproduction_assets')
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=True)
    manifest_path = args.target / 'preparation_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for dataset in args.datasets:
        print(f'Preparing {dataset}', flush=True)
        if dataset == 'SUN397':
            result = prepare_sun(args.source, args.target, args.assets)
        elif dataset == 'SUN397Mirror':
            result = prepare_sun_mirror(args.target, args.assets)
        elif dataset == 'DTD':
            result = prepare_dtd(args.source, args.target, args.assets)
        elif dataset == 'RESISC45':
            result = prepare_resisc(args.target, args.assets)
        else:
            result = prepare_eurosat(args.target, args.assets)
        manifest['SUN397' if dataset == 'SUN397Mirror' else dataset] = result
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'READY {dataset}: ' + str({key: split['count'] for key, split in result['splits'].items()}), flush=True)


if __name__ == '__main__':
    main()
