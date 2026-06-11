# EVIQUE

EVIQUE is the cleaned ICDE/GitHub release candidate for the evidence-indexing, DB benchmark, RAG comparison, and ablation code. It keeps first-party source and small reproducibility assets, and excludes generated runs, caches, raw videos, model weights, and unredistributed third-party repositories.

## Structure
- `evique/`: core evidence indexing and retrieval package.
- `db_benchmark/`: DB schema, adapters, registry, and metrics.
- `rag_baselines/`: small local NaiveRAG/TextVideoRAG helpers.
- `configs/ablation/`, `experiments/`, `scripts/`, `examples/`, `results/paper/`, `third_party/`, `docs/`.

## Install
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Smoke Test
```bash
python -m evique.cli.check_standalone
python scripts/smoke/smoke_check.py
```
This smoke path is CPU-only and does not call paid APIs.

## Reproduction Entrypoints
```bash
python experiments/db/run_baseline_comparison.py --help
python experiments/rag/run_comparison.py --help
python experiments/ablation/run_evique_ablation.py --help
```
Full reproduction requires external datasets, model weights, and API credentials described in `DATASETS.md`, `MODELS.md`, and `.env.example`.

## Known Release Blockers
Project license is pending; final paper result CSVs were not present in the source working tree; third-party license/commit review is still required.
