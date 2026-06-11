from __future__ import annotations

import argparse
from pathlib import Path

from db_benchmark.metrics import compute_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate DB benchmark metric tables from JSONL results.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--registry")
    args = parser.parse_args(argv)

    compute_metrics(
        Path(args.queries),
        Path(args.results_dir),
        Path(args.metrics_dir),
        registry_path=Path(args.registry) if args.registry else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

