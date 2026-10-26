#!/usr/bin/env python3
"""Run the complete vLLM serving benchmark matrix against local Hauhau."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
BASE_URL = "http://127.0.0.1:18089"
ENDPOINT = "/v1/chat/completions"
MODEL = "qwen3.8-27b-hauhau-aggressive"
TOKENIZER = PROJECT_ROOT / "models/HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP"


@dataclass(frozen=True)
class Config:
    output_dir: Path
    base_url: str = BASE_URL
    model: str = MODEL
    tokenizer: Path = TOKENIZER
    vllm: str = "vllm"
    timeout: int = 1800


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--tokenizer', type=Path, default=TOKENIZER)
    parser.add_argument('--vllm-bin', default=os.environ.get('VLLM_BIN', 'vllm'))
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT_ROOT / 'results' / f'hauhau-benchmark-{dt.datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without requests or writes')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    config = Config(args.output_dir.resolve(), args.base_url.rstrip('/'), args.model,
                    args.tokenizer.resolve(), args.vllm_bin, args.timeout)
    return config, args.dry_run


def wait_ready(config: Config, timeout: int = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(config.base_url + '/health', timeout=3) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    return False


def check_model(config: Config) -> None:
    with urllib.request.urlopen(config.base_url + '/v1/models', timeout=10) as response:
        available = {model['id'] for model in json.load(response)['data']}
    if config.model not in available:
        raise RuntimeError(f'Model {config.model!r} is not served; available: {sorted(available)}')


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def spec(name: str, family: str, *, concurrency: int = 16,
         input_tokens: int = 1024, output_tokens: int = 256,
         request_rate: str | float = "inf", burstiness: float = 1.0,
         thinking: bool = False) -> dict:
    return {
        "name": name,
        "family": family,
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requests": 20,
        "request_rate": request_rate,
        "burstiness": burstiness,
        "thinking": thinking,
    }


def matrix() -> list[dict]:
    cases = [spec(f"concurrency_{c}", "concurrency", concurrency=c)
             for c in (1, 4, 8, 16, 32, 64)]
    cases += [spec(f"input_{n}", "input_length", input_tokens=n)
              for n in (256, 4096, 16384, 32768)]
    cases += [spec(f"output_{n}", "output_length", output_tokens=n)
              for n in (64, 1024)]
    cases += [spec(f"rate_{r}", "arrival_rate", concurrency=64,
                   request_rate=r) for r in (0.5, 1, 2, 4)]
    cases += [spec("burst_rate_2", "burstiness", concurrency=64,
                   request_rate=2, burstiness=0.25)]
    cases += [spec("thinking_on_1024", "thinking", output_tokens=1024,
                   thinking=True)]
    cases += [spec(f"baseline_repeat_{i}", "repeat") for i in (2, 3)]
    return cases


def run(case: dict, index: int, config: Config, *, group: str = 'runs') -> int:
    rid = f"{index:02d}_{case['name']}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    folder = config.output_dir / group / rid
    folder.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbelow(2**31)
    command = [
        config.vllm, "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", config.base_url,
        "--endpoint", ENDPOINT,
        "--model", config.model,
        "--tokenizer", str(config.tokenizer),
        "--dataset-name", "random",
        "--random-input-len", str(case["input_tokens"]),
        "--random-output-len", str(case["output_tokens"]),
        "--num-prompts", str(case["requests"]),
        "--request-rate", str(case["request_rate"]),
        "--burstiness", str(case["burstiness"]),
        "--max-concurrency", str(case["concurrency"]),
        "--num-warmups", "2",
        "--seed", str(seed),
        "--ignore-eos",
        "--temperature", "0",
        "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": case["thinking"]}}),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "5,10,25,50,75,90,95,99,99.9,100",
        "--disable-tqdm",
        "--save-result", "--save-detailed",
        "--plot-timeline", "--plot-dataset-stats",
        "--result-dir", str(folder),
        "--result-filename", "result.json",
    ]
    meta = dict(case, run_id=rid, index=index, seed=seed, started=now(),
                command=command, status="running", folder=str(folder))
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "command.txt").write_text(shlex.join(command) + "\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", TZ="Asia/Shanghai")
    print(f"START {rid}: {json.dumps(case, ensure_ascii=False)}", flush=True)
    started = time.monotonic()
    with (folder / "console.log").open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        try:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=config.timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"\nTIMEOUT after {config.timeout} seconds\n")
        except OSError as exc:
            code = 127
            log.write(f"\nCould not execute benchmark: {exc}\n")
    result = {}
    result_path = folder / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            meta["result_error"] = repr(exc)
    meta.update({"finished": now(), "elapsed_seconds": time.monotonic() - started,
                 "exit_code": code,
                 "status": "completed" if code == 0 and result.get("completed") == case["requests"] and not result.get("failed") else "failed",
                 "completed": result.get("completed"), "failed": result.get("failed")})
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"END {rid}: status={meta['status']} completed={meta.get('completed')} failed={meta.get('failed')} elapsed={meta['elapsed_seconds']:.1f}s", flush=True)
    return code or (0 if meta['status'] == 'completed' else 1)


def run_suite(config: Config, cases: list[dict], *, safe: bool = False,
              dry_run: bool = False) -> int:
    plan = {'created': now(), 'base_url': config.base_url, 'endpoint': ENDPOINT,
            'model': config.model, 'tokenizer': str(config.tokenizer), 'cases': cases,
            'output_dir': str(config.output_dir)}
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not wait_ready(config):
        print('SERVICE NOT READY: refusing to start matrix', flush=True)
        return 2
    check_model(config)
    plan['vllm_client'] = subprocess.check_output([config.vllm, '--version'], text=True).strip()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan_name = 'safe_plan.json' if safe else 'plan.json'
    (config.output_dir / plan_name).write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    succeeded = attempted = 0
    for i, case in enumerate(cases, 1):
        if safe and not wait_ready(config):
            print(f'SERVICE NOT READY before case {i}; stopping', flush=True)
            break
        code = run(case, i, config, group='safe_runs' if safe else 'runs')
        attempted += 1
        succeeded += code == 0
        if safe and not wait_ready(config):
            print('SERVICE NOT READY after case; stopping', flush=True)
            break
    summary = {'total': len(cases), 'attempted': attempted, 'succeeded': succeeded,
               'failed': attempted - succeeded, 'skipped': len(cases) - attempted}
    (config.output_dir / ('safe_summary.json' if safe else 'run_summary.json')).write_text(
        json.dumps(summary, indent=2) + '\n')
    print(f'MATRIX COMPLETE: {json.dumps(summary)}', flush=True)
    return 0 if succeeded == len(cases) else 1


def main() -> int:
    config, dry_run = parse_args(__doc__)
    return run_suite(config, matrix(), dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
