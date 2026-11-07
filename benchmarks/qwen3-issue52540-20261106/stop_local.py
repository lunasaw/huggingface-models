#!/usr/bin/env python3
"""Stop only processes recorded by this experiment, with identity checks."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
args = parser.parse_args()
root = Path(args.run_dir).resolve()
if root.parent != Path(__file__).resolve().parent:
    parser.error('run_dir must belong to this experiment')
manifest = json.loads((root / 'manifest.json').read_text())


def live(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


for kind in ['soak_pid', 'pid', 'telemetry_pid']:
    pid = manifest.get(kind)
    if not pid or not live(pid):
        continue
    proc = Path(f'/proc/{pid}')
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if kind == 'pid':
        valid = 'vllm.entrypoints.openai.api_server' in command and (proc / 'cwd').resolve() == root
        environment = (proc / 'environ').read_bytes()
        valid = valid and b'CUDA_VISIBLE_DEVICES=GPU-a697cda4-5468-8365-b094-805565a294a4\0' in environment
    elif kind == 'soak_pid':
        valid = 'soak_local.py' in command and str(root) in command
    else:
        valid = 'nvidia-smi' in command and 'GPU-a697cda4-5468-8365-b094-805565a294a4' in command
    if not valid or os.getpgid(pid) != pid:
        raise SystemExit(f'Refusing to signal unverified process {pid}')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 25
    while live(pid) and time.monotonic() < deadline:
        time.sleep(.5)
    if live(pid):
        os.killpg(pid, signal.SIGKILL)
    print(json.dumps({'stopped': kind, 'pid': pid}))
