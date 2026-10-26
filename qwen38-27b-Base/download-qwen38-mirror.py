#!/usr/bin/env python3
"""Resume the pinned official model from the mirror and verify every file."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
MODEL = ROOT / 'models/Qwen3.8-27B'
RESULTS = ROOT / 'results/qwen38-bf16-deploy'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
BASE = f'https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/{REVISION}/'
MANIFEST = PACKAGE / 'metadata/mirror-tree.json'
CHUNK = 32 * 1024**2
LOCK = threading.Lock()
URLS = {}


def direct_url(name):
    with LOCK:
        cached = URLS.get(name)
        if cached and time.time() - cached[0] < 2400:
            return cached[1]
        headers = subprocess.check_output([
            'curl', '-fsSI', '--connect-timeout', '15', '--max-time', '45',
            BASE + name + '?download=true',
        ], text=True)
        urls = re.findall(r'^location:\s*(.+)$', headers, re.I | re.M)
        url = urls[-1].strip() if urls else BASE + name + '?download=true'
        URLS[name] = (time.time(), url)
        return url


def segment(task):
    name, start, end = task
    folder = MODEL / '.segments' / name
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f'{start:012d}-{end:012d}'
    length = end - start + 1
    if dst.exists() and dst.stat().st_size == length:
        return length
    tmp = dst.with_suffix('.part')
    headers = dst.with_suffix('.headers')
    for attempt in range(6):
        url = direct_url(name)
        proc = subprocess.run([
            'curl', '-fsSL', '--connect-timeout', '15', '--max-time', '240',
            '--speed-limit', '16384', '--speed-time', '30',
            '--range', f'{start}-{end}', '-D', str(headers), '-o', str(tmp), url,
        ], capture_output=True)
        text = headers.read_text() if headers.exists() else ''
        ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', text, re.I)
        correct_range = bool(ranges) and tuple(map(int, ranges[-1][:2])) == (start, end)
        if proc.returncode == 0 and correct_range and tmp.stat().st_size == length:
            tmp.replace(dst)
            headers.unlink(missing_ok=True)
            return length
        if '403' in text or '401' in text:
            with LOCK:
                URLS.pop(name, None)
        print(f'Retry {name} offset={start} attempt={attempt+1} rc={proc.returncode}', flush=True)
        time.sleep(2)
    raise RuntimeError(f'Failed segment {name}: {start}-{end}')


def verify(entry, path=None):
    path = path or MODEL / entry['path']
    if path.stat().st_size != entry['size']:
        raise RuntimeError(f'Size mismatch: {path.name}')
    with path.open('rb') as stream:
        if 'lfs' in entry:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected = entry['lfs']['oid']
        else:
            data = stream.read()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            expected = entry['oid']
    if actual != expected:
        raise RuntimeError(f'Hash mismatch: {path.name}')
    return {'size': entry['size'], 'hash': actual, 'verified': True}


def main():
    global MODEL, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--verify-only', action='store_true', help='Verify local files without downloading')
    parser.add_argument('--dry-run', action='store_true', help='Show the pinned download plan without network or writes')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    MODEL, RESULTS = args.model_dir.resolve(), args.results_dir.resolve()
    entries = json.loads(MANIFEST.read_text())
    if args.dry_run:
        print(json.dumps({'revision': REVISION, 'model_dir': str(MODEL),
                          'results_dir': str(RESULTS), 'files': len(entries),
                          'total_bytes': sum(e['size'] for e in entries)}, indent=2))
        return
    MODEL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plans = []
    tasks = []
    for entry in ([] if args.verify_only else entries):
        name, size = entry['path'], entry['size']
        dst = MODEL / name
        if dst.exists() and dst.stat().st_size == size:
            continue
        part = MODEL / (name + '.part')
        part.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise RuntimeError(f'Oversized partial file: {name}')
        pieces = [(name, start, min(start + CHUNK, size) - 1) for start in range(offset, size, CHUNK)]
        plans.append((entry, part, pieces))
        tasks.extend(pieces)
    print(f'Pending files={len(plans)}, segments={len(tasks)}, remaining={sum(e-s+1 for _,s,e in tasks)/1024**3:.2f} GiB', flush=True)
    started = time.monotonic()
    completed = 0
    last = started
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(segment, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            now = time.monotonic()
            if now - last >= 20:
                print(f'Segments received={completed/1024**3:.2f} GiB, average={completed/1024**2/(now-started):.1f} MiB/s', flush=True)
                last = now
    for entry, part, pieces in plans:
        with part.open('ab') as out:
            for name, start, end in pieces:
                with (MODEL / '.segments' / name / f'{start:012d}-{end:012d}').open('rb') as source:
                    shutil.copyfileobj(source, out, 8 * 1024**2)
        # A partial or corrupt file must never replace a model file.
        verify(entry, path=part)
        part.replace(MODEL / entry['path'])
        print('Assembled', entry['path'], flush=True)
    verified = {}
    for entry in entries:
        verified[entry['path']] = verify(entry)
        print('Verified', entry['path'], flush=True)
    (RESULTS / 'verified-files.json').write_text(json.dumps(verified, indent=2) + '\n')
    (RESULTS / 'revision.txt').write_text(REVISION + '\n')
    print(f'VERIFICATION_COMPLETE: all {len(entries)} files verified against repository hashes', flush=True)


if __name__ == '__main__':
    main()
