#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
parser.add_argument('--wait', type=float, default=0)
parser.add_argument('--compact', action='store_true')
args = parser.parse_args()
if not 0 <= args.wait <= 50:
    parser.error('wait must be between 0 and 50 seconds')
time.sleep(args.wait)
root = Path(args.run_dir)
manifest = json.loads((root / 'manifest.json').read_text())
result = {'run': root.name, 'pid': manifest['pid'], 'process_exists': Path(f"/proc/{manifest['pid']}").exists()}
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{manifest['port']}/health", timeout=2) as response:
        result['health'] = response.status
except Exception as exc:
    result['health'] = str(exc)
soak_dir = Path(manifest.get('active_soak_dir', str(root)))
if (soak_dir / 'soak-status.json').exists():
    status = json.loads((soak_dir / 'soak-status.json').read_text())
    result['soak_directory'] = str(soak_dir)
    result['soak'] = {k: status.get(k) for k in [
        'elapsed_seconds', 'completed', 'errors', 'inflight', 'completions_per_second',
        'latency_seconds', 'stop_reason', 'prompt_token_min', 'prompt_token_max', 'final',
    ]}
log = (root / 'server.log').read_text(errors='replace')
result['server_log_tail'] = log.splitlines()[-5:]
result['cuda_errors'] = [line for line in log.splitlines() if any(word in line.lower() for word in [
    'illegal memory access', 'misaligned address', 'cuda error:', 'out of memory', 'cudnn_status',
    'segfault', 'segmentation fault', 'engine core exited unexpectedly', 'enginedeaderror',
])][-10:]
result['coredumps'] = [{'name': p.name, 'size': p.stat().st_size} for p in root.glob('*.nvcudmp')]
for path in (root / 'vllm-cache').rglob('autotune_configs.json'):
    try:
        data = json.loads(path.read_text())
        result['autotune_cache'] = str(path)
        result['autotune_entry_count'] = len(data)
        entries = data.get('configs', data.get('entries', data)) if isinstance(data, dict) else data
        if isinstance(entries, dict):
            entries = list(entries.values())
        if isinstance(data, dict):
            runner_names = [match.group(1) for key in data for match in [re.match(r"\('fp8_gemm', '([^']+)'", key)] if match]
            result['fp8_runner_counts'] = dict(Counter(runner_names))
    except Exception as exc:
        result['autotune_read_error'] = str(exc)
result['gpu'] = subprocess.check_output([
    'nvidia-smi', '-i', 'GPU-a697cda4-5468-8365-b094-805565a294a4',
    '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
    '--format=csv,noheader',
], text=True, timeout=5).strip()
if args.compact:
    result = {key: result[key] for key in ['run', 'process_exists', 'health', 'soak', 'cuda_errors', 'gpu'] if key in result}
print(json.dumps(result, ensure_ascii=False))
