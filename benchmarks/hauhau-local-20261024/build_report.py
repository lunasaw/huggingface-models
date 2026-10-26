#!/usr/bin/env python3
"""Aggregate the local Hauhau vLLM runs and build a browsable report."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SAFE = ROOT / "safe_runs"
TITLE = 'vLLM serving benchmark'
SERVICE = 'Historical Hauhau benchmark (see plan.json / safe_plan.json)'


def scalar(d: dict, key: str, default=None):
    value = d.get(key, default)
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return None
    return value


def row_from(result: dict, meta: dict, run_dir: Path, *, family=None) -> dict:
    requested = meta.get("requests", result.get("num_prompts"))
    completed = result.get("completed")
    status = meta.get("status")
    if status is None:
        status = "completed" if completed == requested and not result.get("failed") else "failed"
    return {
        "run_id": meta.get("run_id", run_dir.name),
        "name": meta.get("name", run_dir.name),
        "family": family or meta.get("family", "unknown"),
        "status": status,
        "requested": requested,
        "completed": completed,
        "failed": result.get("failed"),
        "concurrency": meta.get("concurrency", result.get("max_concurrency")),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "request_rate": meta.get("request_rate", result.get("request_rate")),
        "burstiness": meta.get("burstiness", result.get("burstiness")),
        "thinking": meta.get("thinking", False),
        "duration_s": scalar(result, "duration"),
        "request_throughput": scalar(result, "request_throughput"),
        "output_throughput": scalar(result, "output_throughput"),
        "total_token_throughput": scalar(result, "total_token_throughput"),
        "max_concurrent_requests": scalar(result, "max_concurrent_requests"),
        "p50_ttft_ms": scalar(result, "p50_ttft_ms"),
        "p95_ttft_ms": scalar(result, "p95_ttft_ms"),
        "p50_tpot_ms": scalar(result, "p50_tpot_ms"),
        "p95_tpot_ms": scalar(result, "p95_tpot_ms"),
        "p50_itl_ms": scalar(result, "p50_itl_ms"),
        "p95_itl_ms": scalar(result, "p95_itl_ms"),
        "p50_e2el_ms": scalar(result, "p50_e2el_ms"),
        "p95_e2el_ms": scalar(result, "p95_e2el_ms"),
        "spec_acceptance_pct": scalar(result, "spec_decode_acceptance_rate"),
        "result_dir": str(run_dir.relative_to(ROOT)),
        "result_json": str((run_dir / "result.json").relative_to(ROOT)),
        "timeline": str((run_dir / "result.timeline.html").relative_to(ROOT))
        if (run_dir / "result.timeline.html").exists() else "",
        "dataset_plot": str((run_dir / "result.dataset_stats.png").relative_to(ROOT))
        if (run_dir / "result.dataset_stats.png").exists() else "",
    }


def collect() -> list[dict]:
    rows: list[dict] = []
    for parent in (RUNS, SAFE):
        if not parent.exists():
            continue
        for run_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
            result_path = run_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                result = json.loads(result_path.read_text())
                meta_path = run_dir / "meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                rows.append(row_from(result, meta, run_dir))
            except Exception as exc:
                print(f"skip {run_dir}: {exc}")

    for run_dir in sorted((ROOT / "diagnostics").glob("*/")):
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            rows.append(row_from(result, {"name": run_dir.name,
                                          "family": "diagnostic",
                                          "run_id": run_dir.name,
                                          "status": "completed"}, run_dir,
                                 family="diagnostic"))

    smoke = ROOT / "smoke-success.json"
    if smoke.exists():
        row = row_from(json.loads(smoke.read_text()),
                             {"name": "smoke_success", "family": "smoke",
                              "run_id": "smoke-success", "status": "completed"},
                             ROOT, family="smoke")
        row['result_json'] = smoke.name
        row['timeline'] = 'smoke-success.timeline.html' if (ROOT / 'smoke-success.timeline.html').exists() else ''
        row['dataset_plot'] = 'smoke-success.dataset_stats.png' if (ROOT / 'smoke-success.dataset_stats.png').exists() else ''
        rows.append(row)
    return rows


def finite_points(rows: list[dict], x: str, y: str) -> list[tuple[float, float]]:
    pts = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        try:
            xv, yv = float(row[x]), float(row[y])
            if math.isfinite(xv) and math.isfinite(yv):
                pts.append((xv, yv))
        except (TypeError, ValueError):
            pass
    return sorted(pts)


def plot_group(rows: list[dict], family: str, x: str, xlabel: str,
               filename: str, title: str, series: list[tuple[str, str, str]]) -> str:
    group = [r for r in rows if r.get("family") == family]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    drawn = False
    for y, label, marker in series:
        points = finite_points(group, x, y)
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker=marker, linewidth=2, label=label)
            drawn = True
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    if drawn:
        ax.legend()
    fig.tight_layout()
    path = ROOT / filename
    fig.savefig(path)
    plt.close(fig)
    return path.name


def write_csv(rows: list[dict]) -> Path:
    path = ROOT / "summary.csv"
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def write_reports(rows: list[dict], charts: list[str]) -> None:
    successful = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") != "completed"]
    rows_for_table = sorted(rows, key=lambda r: r.get("run_id", ""))
    header = ["run", "family", "status", "done/req", "conc", "in", "out",
              "req/s", "out tok/s", "P50 TTFT ms", "P95 E2EL ms"]
    table = [header]
    for r in rows_for_table:
        table.append([
            r["name"], r["family"], r["status"],
            f'{fmt(r["completed"])}/{fmt(r["requested"])}',
            fmt(r["concurrency"]), fmt(r["input_tokens"]), fmt(r["output_tokens"]),
            fmt(r["request_throughput"]), fmt(r["output_throughput"]),
            fmt(r["p50_ttft_ms"]), fmt(r["p95_e2el_ms"]),
        ])
    md = [
        f"# {TITLE}",
        "",
        SERVICE,
        "",
        f"Collected {len(rows)} runs: {len(successful)} completed and {len(failed)} failed/partial.",
        "Long-context and long-output runs use client concurrency 2 because the running service is configured with `max-num-seqs=2`; short-output concurrency tests cover queueing through 64 clients.",
        "",
        "## Charts",
        "",
    ]
    md += [f"- [{Path(c).stem}]({c})" for c in charts]
    md += ["", "## Runs", "", "| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(map(str, row)) + " |" for row in table[1:]]
    (ROOT / "REPORT.md").write_text("\n".join(md) + "\n")

    html_rows = []
    for r in rows_for_table:
        links = []
        if r.get("result_json"):
            links.append(f'<a href="{html.escape(r["result_json"])}">JSON</a>')
        if r.get("timeline"):
            links.append(f'<a href="{html.escape(r["timeline"])}">timeline</a>')
        if r.get("dataset_plot"):
            links.append(f'<a href="{html.escape(r["dataset_plot"])}">dataset plot</a>')
        html_rows.append("<tr>" + "".join([
            f'<td>{html.escape(str(r["name"]))}</td>',
            f'<td>{html.escape(str(r["family"]))}</td>',
            f'<td class="{html.escape(str(r["status"]))}">{html.escape(str(r["status"]))}</td>',
            f'<td>{fmt(r["completed"])}/{fmt(r["requested"])}</td>',
            f'<td>{fmt(r["concurrency"])}</td>', f'<td>{fmt(r["input_tokens"])}</td>',
            f'<td>{fmt(r["output_tokens"])}</td>', f'<td>{fmt(r["request_throughput"])}</td>',
            f'<td>{fmt(r["output_throughput"])}</td>', f'<td>{fmt(r["p50_ttft_ms"])}</td>',
            f'<td>{fmt(r["p95_e2el_ms"])}</td>', f'<td>{" · ".join(links)}</td>',
        ]) + "</tr>")
    chart_html = "\n".join(
        f'<figure><img src="{html.escape(c)}" alt="{html.escape(Path(c).stem)}"><figcaption>{html.escape(Path(c).stem)}</figcaption></figure>'
        for c in charts)
    page = f'''<!doctype html>
<meta charset="utf-8"><title>{html.escape(TITLE)}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#202124}} img{{max-width:900px;width:100%;border:1px solid #ddd}} figure{{margin:1.2rem 0}} table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ddd;padding:5px 7px;white-space:nowrap}}th{{background:#f3f4f6;position:sticky;top:0}}.completed{{color:#087f23}}.failed{{color:#b42318}}.running{{color:#9a6700}}</style>
<h1>{html.escape(TITLE)}</h1>
<p>{html.escape(SERVICE)} · {len(successful)} completed, {len(failed)} failed/partial</p>
<p>Long context/output cases use client concurrency 2 (server <code>max-num-seqs=2</code>); short-output queue tests reach concurrency 64.</p>
<h2>Charts</h2>{chart_html}<h2>Runs</h2>
<table><thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in header)}<th>artifacts</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table>
'''
    (ROOT / "REPORT.html").write_text(page)


def main() -> int:
    global ROOT, RUNS, SAFE, SERVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT,
                        help='Directory containing runs/ and safe_runs/; reports are written here')
    args = parser.parse_args()
    ROOT = args.input_dir.resolve()
    RUNS, SAFE = ROOT / 'runs', ROOT / 'safe_runs'
    services = []
    for name in ('plan.json', 'safe_plan.json'):
        path = ROOT / name
        if path.exists():
            plan = json.loads(path.read_text())
            services.append(f"Model: {plan.get('model', 'unknown')}; endpoint: {plan.get('base_url', '')}{plan.get('endpoint', '')}")
    if services:
        SERVICE = ' | '.join(dict.fromkeys(services))
    rows = collect()
    if not rows:
        parser.error(f'No benchmark results found in {ROOT}')
    write_csv(rows)
    charts = [
        plot_group(rows, "concurrency_short", "concurrency", "client max concurrency",
                   "concurrency-throughput.png", "Short-output concurrency throughput",
                   [("request_throughput", "requests/s", "o"),
                    ("output_throughput", "output tokens/s", "s")]),
        plot_group(rows, "concurrency_short", "concurrency", "client max concurrency",
                   "concurrency-latency.png", "Short-output concurrency latency",
                   [("p50_ttft_ms", "P50 TTFT (ms)", "o"),
                    ("p95_e2el_ms", "P95 E2EL (ms)", "s")]),
        plot_group(rows, "input_length", "input_tokens", "random input tokens",
                   "input-length.png", "Input length impact (client concurrency 2)",
                   [("p50_ttft_ms", "P50 TTFT (ms)", "o"),
                    ("p95_e2el_ms", "P95 E2EL (ms)", "s")]),
        plot_group(rows, "output_length", "output_tokens", "random output tokens",
                   "output-length.png", "Output length impact (client concurrency 2)",
                   [("output_throughput", "output tokens/s", "o"),
                    ("p95_e2el_ms", "P95 E2EL (ms)", "s")]),
        plot_group(rows, "arrival_rate", "request_rate", "offered request rate (req/s)",
                   "arrival-rate.png", "Arrival-rate sweep (client concurrency 2)",
                   [("request_throughput", "actual requests/s", "o"),
                    ("output_throughput", "output tokens/s", "s")]),
    ]
    write_reports(rows, charts)
    print(f"Wrote {len(rows)} rows and {len(charts)} charts to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
