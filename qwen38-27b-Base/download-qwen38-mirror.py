#!/usr/bin/env python3
"""Download every file of a pinned model snapshot from a Hugging Face mirror."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REPO_ID = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL = ROOT / 'models/Qwen3.8-27B'
RESULTS = ROOT / 'results/qwen38-bf16-deploy'
MANIFEST = PACKAGE / 'metadata/mirror-tree.json'


def load_manifest():
    entries = json.loads(MANIFEST.read_text())
    names = set()
    for entry in entries:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or entry['path'] in names:
            raise ValueError(f'Unsafe or duplicate manifest path: {path}')
        if entry.get('type') != 'file' or entry['size'] < 0:
            raise ValueError(f'Invalid file metadata: {path}')
        names.add(entry['path'])
    return entries


def verify(entry, path=None):
    path = path or MODEL / entry['path']
    if path.stat().st_size != entry['size']:
        raise RuntimeError(f'Size mismatch: {path.name}')
    with path.open('rb') as stream:
        if 'lfs' in entry:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected = entry['lfs']['oid']
        else:
            digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
            actual, expected = digest.hexdigest(), entry['oid']
    if actual != expected:
        raise RuntimeError(f'Hash mismatch: {path.name}')
    return {'size': entry['size'], 'hash': actual, 'verified': True}


def fetch_file(entry, args, downloader):
    path = args.model_dir / entry['path']
    if not args.force_download and path.exists():
        # Do not silently replace local modifications or a running model's files.
        return verify(entry, path)
    downloader(repo_id=REPO_ID, filename=entry['path'], revision=REVISION,
               local_dir=args.model_dir, endpoint=args.endpoint,
               force_download=args.force_download)
    return verify(entry, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true', help='Check all local files without network access')
    parser.add_argument('--force-download', action='store_true', help='Replace local files with the pinned upstream versions')
    parser.add_argument('--dry-run', action='store_true', help='Print the complete manifest without network or writes')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.verify_only and args.force_download:
        parser.error('--verify-only cannot be combined with --force-download')
    args.model_dir = args.model_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    entries = load_manifest()
    if args.dry_run:
        print(json.dumps({'repo_id': REPO_ID, 'revision': REVISION,
                          'endpoint': args.endpoint, 'model_dir': str(args.model_dir),
                          'results_dir': str(args.results_dir), 'files': len(entries),
                          'total_bytes': sum(e['size'] for e in entries),
                          'filenames': [e['path'] for e in entries]}, indent=2))
        return 0
    downloader = None
    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            parser.error('Install this project\'s requirements.txt in its Conda environment first')
        downloader = hf_hub_download
        args.model_dir.mkdir(parents=True, exist_ok=True)
    verified, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            (pool.submit(verify, entry, args.model_dir / entry['path']) if args.verify_only
             else pool.submit(fetch_file, entry, args, downloader)): entry['path']
            for entry in entries
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                verified[name] = future.result()
                print(f'Verified {name}', flush=True)
            except Exception as exc:
                # Signed download URLs may contain credentials; keep them out of logs.
                failures[name] = type(exc).__name__
                print(f'FAILED {name}: {type(exc).__name__}', flush=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = {'repo_id': REPO_ID, 'revision': REVISION,
              'verified': verified, 'failed': failures}
    (args.results_dir / 'download-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        print(f'INCOMPLETE: {len(verified)}/{len(entries)} files verified. '
              'Check the endpoint and local files; --force-download replaces local files explicitly.')
        return 1
    (args.results_dir / 'verified-files.json').write_text(json.dumps(verified, indent=2) + '\n')
    (args.results_dir / 'revision.txt').write_text(REVISION + '\n')
    print(f'VERIFICATION_COMPLETE: all {len(entries)} files verified against repository hashes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
