#!/usr/bin/env python3
"""Run a complete, service-safe vLLM benchmark matrix.

The endpoint is configured with max-num-seqs=2, so this matrix exercises
larger client-side queues with short generations and uses concurrency 2 for
the longer context/output measurements.  The separate stress runs record
what happens when that serving limit is exceeded.
"""
from __future__ import annotations

import run_matrix as base


def safe_matrix() -> list[dict]:
    cases = [
        base.spec(f"concurrency_{c}_short", "concurrency_short",
                  concurrency=c, input_tokens=256, output_tokens=64)
        for c in (1, 2, 4, 8, 16, 32, 64)
    ]
    cases += [
        base.spec(f"input_{n}_c2", "input_length", concurrency=2,
                  input_tokens=n, output_tokens=256)
        for n in (256, 1024, 4096, 16384, 32768)
    ]
    cases += [
        base.spec(f"output_{n}_c2", "output_length", concurrency=2,
                  input_tokens=1024, output_tokens=n)
        for n in (64, 256, 1024)
    ]
    cases += [
        base.spec(f"rate_{r}_c2", "arrival_rate", concurrency=2,
                  request_rate=r, input_tokens=1024, output_tokens=256)
        for r in (0.5, 1, 2, 4)
    ]
    cases += [base.spec("burst_rate_2_c2", "burstiness", concurrency=2,
                        request_rate=2, burstiness=0.25)]
    cases += [base.spec("thinking_on_1024_c2", "thinking", concurrency=2,
                        output_tokens=1024, thinking=True)]
    cases += [base.spec(f"baseline_repeat_{i}_c2", "repeat", concurrency=2)
              for i in (2, 3)]
    # Keep tail/context tests representative without spending most of the
    # suite on a few very long prompts.
    for case in cases:
        if case["input_tokens"] >= 32768:
            case["requests"] = 3
        elif case["input_tokens"] >= 16384:
            case["requests"] = 5
        elif case["output_tokens"] >= 1024:
            case["requests"] = 8
        elif case["input_tokens"] >= 4096:
            case["requests"] = 10
    return cases


def main() -> int:
    config, dry_run = base.parse_args(__doc__)
    return base.run_suite(config, safe_matrix(), safe=True, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
