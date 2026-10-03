"""Download pinned Hugging Face assets with size/SHA256 checks and retries."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fetch(item):
    path = Path(item['destination'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    expected = item['sha256']
    if path.exists():
        if path.stat().st_size == item['size'] and sha256(path) == expected:
            print('VERIFIED', path.name, flush=True)
            return item
        path.replace(temporary)
    for attempt in range(8):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset >= item['size']:
                if offset == item['size'] and sha256(temporary) == expected:
                    temporary.replace(path)
                    return item
                offset = 0
            request = urllib.request.Request(
                item['url'] + '?download=true&attempt=' + str(time.time_ns()),
                headers={'Range': f'bytes={offset}-', 'User-Agent': 'ACE-reproduction'},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                else:
                    offset = 0
                with temporary.open('ab' if offset else 'wb') as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
            assert temporary.stat().st_size == item['size'], (temporary.stat().st_size, item['size'])
            assert sha256(temporary) == expected, f'SHA256 mismatch: {path}'
            temporary.replace(path)
            print('VERIFIED', path.name, item['size'], flush=True)
            return item
        except Exception as error:
            print('RETRY', path.name, attempt + 1, repr(error), flush=True)
    raise RuntimeError(f'Could not download and verify {item["url"]}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['SUN397', 'EuroSAT', 'RESISC45'], required=True)
    parser.add_argument('--assets', type=Path, default=ROOT / 'data/reproduction_assets')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    items = []
    for name in args.datasets:
        repo, subdir = {'SUN397': ('tanganke/sun397', 'data'),
                        'EuroSAT': ('torchgeo/eurosat', ''),
                        'RESISC45': ('torchgeo/resisc45', '')}[name]
        revision = get_json(f'https://huggingface.co/api/datasets/{repo}')['sha']
        files = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{subdir}')
        for file in files:
            filename = file['path']
            selected = ((name == 'SUN397' and filename.startswith('data/test-')) or
                        (name == 'EuroSAT' and filename == 'EuroSAT.zip') or
                        (name == 'RESISC45' and filename == 'NWPU-RESISC45.zip'))
            if not selected:
                continue
            destination = args.assets / ('sun397_test' if name == 'SUN397' else '') / Path(filename).name
            items.append({'dataset': name, 'repository': repo, 'revision': revision,
                          'url': f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',
                          'size': file['size'], 'sha256': file['lfs']['oid'],
                          'destination': str(destination)})
    print(f'Downloading {len(items)} files ({sum(x["size"] for x in items)/1e9:.2f} GB)', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(pool.map(fetch, items))
    (args.assets / ('verified_' + '_'.join(args.datasets) + '.json')).write_text(json.dumps(verified, indent=2) + '\n')


if __name__ == '__main__':
    main()
