#!/usr/bin/env python3
"""Start a bounded client and GPU telemetry for an already healthy local test server."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
PYTHON = '/data/miniconda3/envs/qwen38-27b-vllm/bin/python'
GPU = 'GPU-a697cda4-5468-8365-b094-805565a294a4'

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
parser.add_argument('--concurrency', type=int, default=16)
parser.add_argument('--duration', type=int, default=1800)
args = parser.parse_args()
run = (ROOT / args.run_name).resolve()
if run.parent != ROOT:
    parser.error('run must belong to this experiment')
path = run / 'manifest.json'
manifest = json.loads(path.read_text())
if manifest.get('soak_pid'):
    parser.error('a soak has already been started for this run')
proc = Path(f"/proc/{manifest['pid']}")
if (proc / 'cwd').resolve() != run or b'vllm.entrypoints.openai.api_server' not in (proc / 'cmdline').read_bytes():
    parser.error('server process identity mismatch')
base = f"http://127.0.0.1:{manifest['port']}"
with urllib.request.urlopen(base + '/health', timeout=5) as response:
    if response.status != 200:
        parser.error('server not healthy')
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
with (run / 'gpu-telemetry.csv').open('wb') as output:
    telemetry = subprocess.Popen([
        'nvidia-smi', '-i', GPU,
        '--query-gpu=timestamp,index,pci.bus_id,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
        '--format=csv', '-l', '5',
    ], cwd=run, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
command = [PYTHON, str(ROOT / 'soak_local.py'), '--run-dir', str(run), '--base-url', base,
           '--duration', str(args.duration), '--concurrency', str(args.concurrency)]
with (run / 'soak.log').open('wb') as output:
    soak = subprocess.Popen(command, cwd=run, env=env, stdout=output, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
manifest.update({'telemetry_pid': telemetry.pid, 'soak_pid': soak.pid, 'soak_command': command,
                 'soak_launched_at': datetime.datetime.now().astimezone().isoformat()})
path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'run': run.name, 'soak_pid': soak.pid, 'telemetry_pid': telemetry.pid,
                  'concurrency': args.concurrency, 'duration': args.duration}))
