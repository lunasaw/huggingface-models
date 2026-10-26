#!/usr/bin/env python3
"""Small real requests exercising the replacement service, without executing tools."""
import argparse
import base64
from functools import partial
import io
import json
from pathlib import Path
import time
import urllib.request

BASE = 'http://127.0.0.1:18089'
MODEL = 'Qwen3.8 27b'
RESULTS = Path(__file__).resolve().parents[1] / 'results/qwen38-bf16-deploy'


def post(payload, *, base_url=BASE, model=MODEL):
    started = time.monotonic()
    request = urllib.request.Request(
        base_url + '/v1/chat/completions',
        data=json.dumps({'model': model, **payload}).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode()
        code = response.status
    return {'http_status': code, 'seconds': round(time.monotonic() - started, 3), 'response': body}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=BASE, help='Server URL without /v1')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--alias', default='qwen3.8-27b')
    parser.add_argument('--max-model-len', type=int, default=262144)
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    args = parser.parse_args()
    from PIL import Image

    send = partial(post, base_url=args.base_url.rstrip('/'), model=args.model)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with urllib.request.urlopen(args.base_url.rstrip('/') + '/v1/models', timeout=10) as response:
        models = json.load(response)
    require({args.model, args.alias} == {m['id'] for m in models['data']}, 'Unexpected model aliases')
    require(all(m['max_model_len'] == args.max_model_len for m in models['data']), 'Unexpected context length')
    results['models'] = models
    for name, alias in [('default_chinese', args.model), ('legacy_alias', args.alias)]:
        result = send({'model': alias, 'messages': [{'role': 'user', 'content': '请用一句中文介绍你能做什么。'}], 'max_tokens': 96})
        body = json.loads(result['response'])
        message = body['choices'][0]['message']
        require(message['content'] and any('\u4e00' <= c <= '\u9fff' for c in message['content']), 'Missing Chinese response')
        require(not message.get('reasoning'), 'Thinking must be disabled by default')
        results[name] = result
        print(name, result['http_status'], result['seconds'], message['content'], flush=True)
    result = send({'messages': [{'role': 'user', 'content': '只回复：流式正常'}], 'stream': True, 'max_tokens': 48})
    require('data: [DONE]' in result['response'], 'Stream did not finish')
    chunks = [json.loads(line[6:]) for line in result['response'].splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
    content = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c.get('choices'))
    require('流式正常' in content, 'Unexpected streamed content')
    results['stream'] = result
    print('stream', result['http_status'], result['seconds'], content, flush=True)
    result = send({
        'messages': [{'role': 'user', 'content': '请调用 get_weather 查询杭州天气，不要猜测天气。'}],
        'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': '查询城市天气', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}],
        'tool_choice': 'auto', 'max_tokens': 192,
    })
    body = json.loads(result['response'])
    calls = body['choices'][0]['message'].get('tool_calls')
    require(calls and calls[0]['function']['name'] == 'get_weather', 'Missing weather tool call')
    require(json.loads(calls[0]['function']['arguments']).get('city'), 'Missing tool city argument')
    results['tool_call'] = result
    print('tool_call', result['http_status'], result['seconds'], calls[0]['function'], flush=True)
    output = io.BytesIO()
    Image.new('RGB', (224, 224), (255, 0, 0)).save(output, format='PNG')
    data = base64.b64encode(output.getvalue()).decode()
    result = send({'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '图片是什么颜色？只用中文回答颜色。'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}}]}], 'max_tokens': 48})
    body = json.loads(result['response'])
    content = body['choices'][0]['message']['content']
    require('红' in content, 'Unexpected image description')
    results['vision'] = result
    print('vision', result['http_status'], result['seconds'], content, flush=True)
    (args.results_dir / 'api-verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('ALL_CHECKS_PASSED', flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


if __name__ == '__main__':
    main()
