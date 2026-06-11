#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from db_rag_pipeline_common import parse_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch DB-RAG run progress from files already written under a run root.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--methods", default="EVIQUE,LOVO,VOCAL,MIRIS,OTIF,UMT,VISA,FiGO,ZELDA")
    parser.add_argument("--expected-queries", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.0, help="Refresh interval in seconds. Omit or set 0 for one-shot.")
    parser.add_argument("--latest", type=int, default=10, help="Number of latest modified files to show.")
    parser.add_argument("--log-tail-lines", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    while True:
        _print_snapshot(args.run_root, methods, args.expected_queries, args.latest, args.log_tail_lines)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


def _print_snapshot(run_root: Path, methods: list[str], expected_queries: int, latest_count: int, log_tail_lines: int) -> None:
    run_root = Path(run_root)
    expected = expected_queries * len(methods) if expected_queries else 0
    evidence_count = sum(len(list((run_root / f"evidence-{method}").glob("evidence_*.json"))) for method in methods)
    answer_count = sum(len(list((run_root / f"answers-{method}").glob("answer_*.md"))) for method in methods)
    quant_count = _judgement_count(run_root / "evaluation" / "quantitative_judgements.json")
    winrate_count = _judgement_count(run_root / "evaluation" / "winrate_judgements.json")

    print("=" * 80, flush=True)
    print(f"timestamp={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print(f"methods={','.join(methods)}", flush=True)
    if expected:
        print(f"expected_method_query_tasks={expected}", flush=True)
    print(f"evidence files count={_count_text(evidence_count, expected)}", flush=True)
    print(f"answer files count={_count_text(answer_count, expected)}", flush=True)
    print(f"quantitative judgements count={quant_count}", flush=True)
    print(f"winrate judgements count={winrate_count}", flush=True)
    print("", flush=True)
    print("latest modified files:", flush=True)
    for path in _latest_files(run_root, latest_count):
        stat = path.stat()
        rel = path.relative_to(run_root)
        print(f"  {datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')} {stat.st_size:>10} {rel}", flush=True)
    print("", flush=True)
    log_path = _latest_log(run_root)
    if log_path:
        print(f"latest log tail: {log_path.relative_to(run_root)}", flush=True)
        for line in _tail_lines(log_path, log_tail_lines):
            print(line, flush=True)
    else:
        print("latest log tail: (no .log files found)", flush=True)


def _count_text(count: int, expected: int) -> str:
    if not expected:
        return str(count)
    pct = count / max(expected, 1) * 100
    return f"{count}/{expected} ({pct:.1f}%)"


def _judgement_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(data) if isinstance(data, dict) else 0


def _latest_files(run_root: Path, count: int) -> list[Path]:
    if not run_root.exists():
        return []
    files = [path for path in run_root.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[: max(count, 0)]


def _latest_log(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    logs = [path for path in run_root.rglob("*.log") if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def _tail_lines(path: Path, line_count: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [f"(could not read log: {type(exc).__name__}: {exc})"]
    return lines[-max(line_count, 0) :]


if __name__ == "__main__":
    main()
