# EVIQUE

This repository is the initial public implementation of EVIQUE.

EVIQUE builds multi-view video evidence indexes and retrieves compact evidence packages for question answering. This initial public code release keeps first-party method code, compact baseline-output adapters, minimal configs, and scripts for rebuilding views, running retrieval, ablations, tables, and figures.

## Core Contributions
- Scope, Target, Track, and Event views over video segment records.
- Evidence graph construction with nodes, relations, planner, and packer.
- Query-time retrieval with bounded graph/view expansion and evidence packaging.
- Baseline output adapters without redistributing third-party repositories.

## Structure
`evique/` contains first-party code. `scripts/` contains build/run/ablation/table/figure entrypoints. `baselines/` contains adapters, prompts, and compact examples. `data/queries/` records query slots and provenance. `results/paper/` contains compact CSV schemas.

## Install
```bash
python -m pip install -r requirements.txt
```
Optional video/model dependencies are in `requirements-vision.txt`.

## Build Views
```bash
python scripts/build_views.py --segments-json examples/demo_segments.json --output-dir outputs/demo_index
```

## Run EVIQUE
```bash
python scripts/run_evique.py --index-dir outputs/demo_index --query-file examples/minimal_query.json
```

## Run Ablation
```bash
python scripts/run_ablation.py --index-dir outputs/demo_index --query-file examples/minimal_query.json
```

## Reproduce Tables And Figures
```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

Raw videos, checkpoints, large indexes, generated runs, and API keys are not distributed. Third-party baseline implementations are not redistributed. Only result adapters, shared prompts, and compact format examples are included.

## License

The project license is currently pending. Until an explicit open-source license is added, no additional rights to use, modify, or redistribute this source code are granted beyond those provided by applicable law.

## Citation
See `CITATION.cff`; update authors and paper metadata before public release.

## Query Data

The complete query text set contains 96 queries: 30 for Warsaw, 25 for Bellevue, 20 for QVHighlights, and 21 for Beach. Warsaw, Bellevue, and Beach are ready single-video query sets.

The QVHighlights query-to-video mapping was not available in the source artifacts used to prepare this initial release. Therefore, the QVHighlights query texts are included, but the full QVHighlights experiment cannot yet be reproduced from this repository alone. QVHighlights remains marked `missing_video_id` in `results/paper/query_manifest.csv`, with zero mapped query video IDs.

```python
from evique.utils.query_io import load_query_file

dataset = load_query_file("data/queries/warsaw.json")
print(dataset.query_count)
```

## Initial Limitations

Verified full paper result CSVs are not present in the current local source workspace. QVHighlights query text is included, but source-video metadata remains unresolved.
