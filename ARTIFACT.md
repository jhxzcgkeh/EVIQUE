# Artifact Mapping

| Paper item | Claim | Command | Input | Output | Runtime | Hardware/API |
|---|---|---|---|---|---|---|
| DB benchmark | EVIQUE-DB retrieval | `python experiments/db/run_baseline_comparison.py --queries examples/sample_queries/db_queries.json --output-root outputs/db --methods EVIQUE-DB --dry-run` | query JSON | DB metrics CSV | minutes for smoke | CPU, no API for dry-run |
| RAG comparison | EVIQUE vs RAG baselines | `python experiments/rag/run_comparison.py --help` | external videos/questions | answer/eval summaries | hours full | GPU/API for full |
| Core ablation | planner/packaging contribution | `python experiments/ablation/run_evique_ablation.py --help` | existing workdir/configs | ablation summaries | dataset dependent | API for answer/eval |
| Paper tables | aggregation | `python scripts/tables/merge_db_results.py --help` | verified runs | paper CSVs | minutes | CPU |
