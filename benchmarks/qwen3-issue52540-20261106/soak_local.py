#!/usr/bin/env python3
"""Bounded local load test; logs metadata only and stops on errors or stalled progress."""
import argparse
import asyncio
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import aiohttp
from transformers import AutoTokenizer

MODEL_PATH = '/data/ai/chenzhangyue/qwen38-27B/model/Qwen3.8-27B-NVFP4'


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


async def main(args):
    run_dir = Path(args.run_dir).resolve()
    if (run_dir / 'soak-summary.json').exists():
        raise SystemExit('A soak result already exists in this directory')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    passage = (
        'A support team reviews a sequence of routine service events. '
        'Each event includes an identifier, an arrival time, a duration, and a status. '
        'The team compares successful responses with delayed responses, checks whether '
        'the records are consistent, and prepares a short factual summary. '
        'No event contains private information. The task is to classify this synthetic '
        'record as normal and explain the classification in plain language. '
    )
    material = tokenizer.encode(passage * 200, add_special_tokens=False)
    rng = random.Random(52540)
    payloads = []
    for index in range(256):
        length = rng.randint(2500, 3500)
        text = tokenizer.decode(material[:length], skip_special_tokens=True)
        payloads.append(text)
    del tokenizer
    stop = asyncio.Event()
    started = time.monotonic()
    deadline = started + args.duration
    state = {
        'started_at': datetime.datetime.now().astimezone().isoformat(),
        'duration_limit_seconds': args.duration, 'concurrency': args.concurrency,
        'base_url': args.base_url, 'model': 'qwen3-issue52540',
        'started': 0, 'completed': 0, 'errors': 0, 'inflight': 0,
        'prompt_tokens': 0, 'completion_tokens': 0, 'last_success_seconds': None,
        'stop_reason': None, 'latencies': [], 'response_hashes_first50': [],
        'http_status_counts': Counter(), 'prompt_token_min': None, 'prompt_token_max': None,
    }
    loop = asyncio.get_running_loop()
    def cancel_requested():
        state['stop_reason'] = 'operator_stop'
        stop.set()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, cancel_requested)

    def snapshot(final=False):
        result = {k: v for k, v in state.items() if k != 'latencies'}
        elapsed = time.monotonic() - started
        result['elapsed_seconds'] = round(elapsed, 3)
        result['completions_per_second'] = round(state['completed'] / max(elapsed, .001), 4)
        ordered = sorted(state['latencies'])
        if ordered:
            result['latency_seconds'] = {
                'p50': ordered[int((len(ordered)-1)*.50)],
                'p95': ordered[int((len(ordered)-1)*.95)],
                'max': ordered[-1],
            }
        result['final'] = final
        return result

    timeout = aiohttp.ClientTimeout(total=90, connect=5)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    event_file = (run_dir / 'requests.jsonl').open('w', buffering=1)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        async def worker(worker_id):
            while not stop.is_set() and time.monotonic() < deadline:
                index = state['started']
                state['started'] += 1
                state['inflight'] += 1
                request_started = time.monotonic()
                # A unique leading identifier prevents shared document prefix reuse.
                prefix = hashlib.sha256(f'issue52540:{index}'.encode()).hexdigest()
                content = f'Record {prefix}\n' + payloads[index % len(payloads)]
                payload = {
                    'model': 'qwen3-issue52540',
                    'messages': [{'role': 'user', 'content': content}],
                    'max_tokens': 48, 'temperature': 0, 'seed': 52540,
                    'ignore_eos': True, 'stream': False,
                    'chat_template_kwargs': {'enable_thinking': False},
                }
                record = {'index': index, 'worker': worker_id,
                          'start_seconds': round(request_started-started, 3)}
                try:
                    async with session.post(args.base_url + '/v1/chat/completions', json=payload) as response:
                        body = await response.text()
                        record['status'] = response.status
                        state['http_status_counts'][str(response.status)] += 1
                        if response.status != 200:
                            raise RuntimeError(f'HTTP {response.status}: {body[:1500]}')
                        result = json.loads(body)
                        usage = result.get('usage', {})
                        if not result.get('choices') or not usage:
                            raise RuntimeError('Missing choices or usage in successful response')
                        latency = round(time.monotonic()-request_started, 4)
                        state['completed'] += 1
                        state['last_success_seconds'] = time.monotonic()-started
                        state['latencies'].append(latency)
                        state['prompt_tokens'] += usage.get('prompt_tokens', 0)
                        state['completion_tokens'] += usage.get('completion_tokens', 0)
                        prompt_count = usage.get('prompt_tokens', 0)
                        state['prompt_token_min'] = min(state['prompt_token_min'] or prompt_count, prompt_count)
                        state['prompt_token_max'] = max(state['prompt_token_max'] or prompt_count, prompt_count)
                        record.update({'latency_seconds': latency, 'usage': usage})
                        if index < 50:
                            digest = hashlib.sha256(json.dumps(result['choices'][0]['message'], sort_keys=True).encode()).hexdigest()
                            state['response_hashes_first50'].append({'index': index, 'sha256': digest})
                except asyncio.CancelledError:
                    record['cancelled'] = True
                    raise
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                    state['errors'] += 1
                    state['stop_reason'] = state['stop_reason'] or 'request_error'
                    stop.set()
                finally:
                    state['inflight'] -= 1
                    event_file.write(json.dumps(record, ensure_ascii=False) + '\n')

        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        try:
            while any(not task.done() for task in workers):
                current = time.monotonic()-started
                if current > args.duration:
                    state['stop_reason'] = state['stop_reason'] or 'duration_reached'
                last = state['last_success_seconds'] or 0
                if current-last > 60 and state['inflight']:
                    state['stop_reason'] = 'no_success_for_60_seconds'
                    stop.set()
                status = snapshot()
                write_json(run_dir / 'soak-status.json', status)
                print(json.dumps({k: status.get(k) for k in ['elapsed_seconds', 'completed', 'errors', 'inflight', 'completions_per_second', 'stop_reason']}), flush=True)
                if stop.is_set():
                    for task in workers:
                        task.cancel()
                    break
                await asyncio.sleep(10)
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            event_file.close()
            state['stop_reason'] = state['stop_reason'] or 'duration_reached'
            result = snapshot(final=True)
            write_json(run_dir / 'soak-status.json', result)
            write_json(run_dir / 'soak-summary.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:28841')
    parser.add_argument('--duration', type=int, default=1800)
    parser.add_argument('--concurrency', type=int, default=24)
    args = parser.parse_args()
    if not args.base_url.startswith('http://127.0.0.1:'):
        parser.error('This experiment is restricted to a local isolated instance')
    if args.duration <= 0 or not 1 <= args.concurrency <= 24:
        parser.error('duration must be positive and concurrency must be 1..24')
    asyncio.run(main(args))
