#!/usr/bin/env python3
"""Launch one isolated arm of the user-authorized GPU 1 experiment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/data/miniconda3/envs/qwen38-27b-vllm/bin/python')
MODEL = '/data/ai/chenzhangyue/qwen38-27B/model/Qwen3.8-27B-NVFP4'
GPU_UUID = 'GPU-a697cda4-5468-8365-b094-805565a294a4'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['B-flashinfer', 'A-cutlass'])
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--port', type=int, default=28841)
    parser.add_argument('--reuse-flashinfer-builds-from')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error('run-name must be a directory name')
    run_dir = ROOT / args.run_name
    run_dir.mkdir(exist_ok=False)
    gpu = subprocess.check_output([
        'nvidia-smi', '-i', GPU_UUID,
        '--query-gpu=index,uuid,name,memory.used,driver_version',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    if int(gpu.split(',')[3].strip()) > 512:
        raise SystemExit('GPU 1 is no longer idle: ' + gpu)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('VLLM_', 'FLASHINFER_', 'CUDA_')) or name in {
            'PYTHONPATH', 'LD_LIBRARY_PATH', 'HTTP_PROXY', 'HTTPS_PROXY',
            'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
        }:
            env.pop(name)
    settings = {
        'CUDA_VISIBLE_DEVICES': GPU_UUID,
        'CUDA_HOME': '/usr/local/cuda-13.2',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'NO_PROXY': '127.0.0.1,localhost',
        'no_proxy': '127.0.0.1,localhost',
        'VLLM_NO_USAGE_STATS': '1',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_CACHE_ROOT': str(run_dir / 'vllm-cache'),
        'TORCHINDUCTOR_CACHE_DIR': str(run_dir / 'torchinductor-cache'),
        'TRITON_CACHE_DIR': str(run_dir / 'triton-cache'),
        'FLASHINFER_WORKSPACE_BASE': str(run_dir / 'flashinfer-workspace'),
        'CUDA_ENABLE_COREDUMP_ON_EXCEPTION': '1',
        'CUDA_COREDUMP_GENERATION_FLAGS': 'skip_global_memory',
        'CUDA_COREDUMP_FILE': str(run_dir / 'cuda-core.%p.nvcudmp'),
        'PYTHONUNBUFFERED': '1',
    }
    if args.reuse_flashinfer_builds_from:
        if Path(args.reuse_flashinfer_builds_from).name != args.reuse_flashinfer_builds_from:
            parser.error('reuse source must be an experiment directory name')
        build_root = ROOT / args.reuse_flashinfer_builds_from / 'flashinfer-workspace'
        if not build_root.is_dir():
            parser.error('reuse source does not contain compiled FlashInfer artifacts')
        settings['FLASHINFER_WORKSPACE_BASE'] = str(build_root)
    if args.arm == 'A-cutlass':
        settings['VLLM_DISABLED_KERNELS'] = 'FlashInferFP8ScaledMMLinearKernel'
    env.update(settings)
    env['PATH'] = '/usr/local/cuda-13.2/bin:' + env.get('PATH', '')
    command = [
        str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', MODEL, '--served-model-name', 'qwen3-issue52540',
        '--host', '127.0.0.1', '--port', str(args.port),
        '--trust-remote-code', '--dtype', 'bfloat16',
        '--quantization', 'modelopt_mixed',
        '--kv-cache-dtype', 'fp8_e4m3', '--mamba-ssm-cache-dtype', 'bfloat16',
        '--attention-backend', 'FLASHINFER',
        '--gpu-memory-utilization', '0.90',
        '--max-model-len', '32768', '--max-num-seqs', '24',
        '--max-num-batched-tokens', '8192',
        '--cudagraph-capture-sizes', '1', '2', '4', '8', '16', '24',
        '--enable-chunked-prefill', '--enable-prefix-caching',
        '--default-chat-template-kwargs', '{"enable_thinking":false}',
        '--generation-config', 'vllm',
    ]
    manifest = {
        'arm': args.arm, 'created_at': datetime.datetime.now().astimezone().isoformat(),
        'command': command, 'environment_overrides': settings,
        'gpu_before': gpu, 'cwd': str(run_dir), 'port': args.port,
    }
    with (run_dir / 'server.log').open('wb') as output:
        process = subprocess.Popen(command, cwd=run_dir, env=env,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    manifest['pid'] = process.pid
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'run_dir': str(run_dir), 'pid': process.pid, 'port': args.port}))


if __name__ == '__main__':
    main()
