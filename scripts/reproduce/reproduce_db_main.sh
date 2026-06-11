#!/usr/bin/env bash
set -euo pipefail
python experiments/db/run_baseline_comparison.py --queries examples/sample_queries/db_queries.json --output-root outputs/db_main --methods EVIQUE-DB --dry-run --skip-query
