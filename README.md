<div align="center">

# EVIQUE

### Utility-Aware Evidence Construction for Grounded Answering over Large-Scale Video Databases

[![Repository](https://img.shields.io/badge/GitHub-EVIQUE-181717?logo=github)](https://github.com/jhxzcgkeh/EVIQUE)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-Initial_Public_Code-orange)](RELEASE_NOTES.md)

**Compact, connected, and traceable video evidence for grounded answer generation**

</div>

---

## Overview

**EVIQUE** is a utility-aware evidence construction engine for natural-language question answering over large-scale video databases. Instead of returning isolated clips, frames, objects, or captions ranked only by semantic relevance, EVIQUE constructs a compact evidence package that connects targets, temporal anchors, trajectories, events, and provenance before invoking a downstream language model.

The system organizes processed video observations into a unified evidence graph and materializes four evidence views:

- **Scope View** — scene-level descriptions and temporal context.
- **Target View** — grounded objects and visual attributes.
- **Track View** — trajectories, motion summaries, and temporal anchors.
- **Event View** — localized state changes and event descriptions.

At query time, a lightweight **Query-to-Evidence Planner** selects dependency-aware access paths under candidate and context budgets. A utility-aware packer then preserves core evidence and provenance, adds useful supporting context, and filters redundant or low-confidence evidence.

> **Release status.** This repository is the initial public implementation of EVIQUE. It includes the first-party method code, query workload, result adapters, prompts, examples, tests, and reproduction entry points. Raw videos, model weights, large indexes, and complete third-party baseline implementations are not redistributed.

## Highlights

- **Evidence-oriented video querying:** constructs answer-supporting evidence packages rather than independent top-ranked items.
- **Four materialized evidence views:** Scope, Target, Track, and Event.
- **Unified evidence graph:** preserves dependency, temporal, and provenance links.
- **Requirement-driven planning:** selects only the views and graph expansions needed by a query.
- **Budget-aware packaging:** balances coverage, coherence, compactness, and traceability.
- **Grounded answer generation:** instructs the downstream LLM to answer only from the prepared evidence package.

## Experimental Highlights

On the 96-query evaluation workload reported in the paper, EVIQUE:

| Comparison | Reported result |
|---|---:|
| Overall pairwise win rate vs. VideoRAG | **65.21%** |
| Average LLM input tokens | **5.0K**, down from 13.4K |
| LLM input-token reduction vs. VideoRAG | **62.7%** |
| Average online query time | **32.3 s**, down from 145.5 s |
| Online query-time reduction vs. VideoRAG | **77.8%** |
| Pairwise win rate vs. video-database baselines | **78.65%–87.24%** |

These results evaluate the utility of the evidence supplied to the answer model; they should not be interpreted as absolute correctness scores.

## Repository Structure

```text
EVIQUE/
├── evique/                 # First-party EVIQUE implementation
│   ├── views/              # Scope, Target, Track, and Event views
│   ├── evidence/           # Evidence graph, planner, and packer
│   ├── retrieval/          # Query-time retrieval and bounded expansion
│   ├── generation/         # Evidence serialization and answer generation
│   ├── evaluation/         # Common evaluation utilities
│   ├── schemas/            # Public data schemas
│   └── utils/              # Query I/O and shared helpers
├── scripts/
│   ├── build_views.py
│   ├── run_evique.py
│   ├── run_ablation.py
│   ├── normalize_queries.py
│   ├── reproduce_tables.py
│   └── reproduce_figures.py
├── baselines/              # Result adapters, shared prompts, format examples
├── data/queries/           # Public 96-query workload
├── data/configs/           # Minimal public configurations
├── results/paper/          # Compact result schemas and manifests
├── examples/               # Minimal runnable examples
├── tests/                  # Unit and smoke tests
└── third_party/            # External-project provenance
```

## Installation

### Core environment

```bash
git clone https://github.com/jhxzcgkeh/EVIQUE.git
cd EVIQUE

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional video, vision, and model dependencies are listed in:

```bash
python -m pip install -r requirements-vision.txt
```

A Conda specification is also provided:

```bash
conda env create -f environment.yml
conda activate evique
```

## Quick Start

### 1. Build evidence views

```bash
python scripts/build_views.py \
  --segments-json examples/demo_segments.json \
  --output-dir outputs/demo_index
```

### 2. Run EVIQUE

```bash
python scripts/run_evique.py \
  --index-dir outputs/demo_index \
  --query-file examples/minimal_query.json
```

### 3. Run an ablation

```bash
python scripts/run_ablation.py \
  --index-dir outputs/demo_index \
  --query-file examples/minimal_query.json
```

All main entry points expose command-line help:

```bash
python scripts/build_views.py --help
python scripts/run_evique.py --help
python scripts/run_ablation.py --help
python scripts/normalize_queries.py --help
```

## Query Workload

The repository contains the complete query text workload used in the paper:

| Dataset | Queries | Status |
|---|---:|---|
| Warsaw | 30 | Ready |
| Bellevue | 25 | Ready |
| QVHighlights | 20 | Query text included; source-video mapping unavailable |
| Beach | 21 | Ready |
| **Total** | **96** | — |

Load a normalized query file with:

```python
from evique.utils.query_io import load_query_file

dataset = load_query_file("data/queries/warsaw.json")
print(dataset.query_count)
```

The QVHighlights query texts are included, but the source artifacts available for this initial release did not contain a verifiable query-to-video mapping. Therefore, the full QVHighlights experiment cannot yet be reproduced from this repository alone. The limitation is recorded in `data/queries/QVHIGHLIGHTS_MAPPING_REPORT.md` and `results/paper/query_manifest.csv`.

## Reproducing Tables and Figures

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

The scripts read compact files under `results/paper/`. They do not require redistributing raw videos, model checkpoints, or complete experimental work directories.

## Baselines and Third-Party Components

Third-party baseline implementations are **not** redistributed in this repository. The `baselines/` directory contains only:

- adapters that normalize existing baseline outputs into a common evaluation schema;
- shared answer-generation and evaluation prompts;
- compact format examples.

The adapters do not reimplement the corresponding baseline algorithms. Official repositories, versions, and citations are documented under `third_party/`.

All external packages, models, datasets, and baseline implementations remain governed by their own licenses and terms. The EVIQUE license does not relicense third-party material.

## Testing

```bash
python -m compileall -q .
python -m pytest -q
```

The public release includes tests for imports, normalized query files, legacy query loading, baseline-result adapters, and the minimal EVIQUE workflow.

## Citation

If you use EVIQUE in your research, please cite:

```bibtex
@misc{li2026evique,
  title        = {EVIQUE: Utility-Aware Evidence Construction for Grounded Answering over Large-Scale Video Databases},
  author       = {Ji Li and Lan Jin and Ruikai Zhu and Chuanwen Li},
  year         = {2026},
  note         = {Manuscript}
}
```

Please update the venue, DOI, and publication metadata after the paper is formally published. Machine-readable citation metadata is provided in `CITATION.cff`.

## License

The first-party EVIQUE source code and repository documentation are released under the [MIT License](LICENSE).

Third-party software, model weights, datasets, and baseline implementations are not relicensed by EVIQUE and remain subject to their respective licenses. Users are responsible for checking the terms of every external component they install or use.

## Acknowledgements

EVIQUE builds on advances in video databases, video question answering, retrieval-augmented generation, visual grounding, object tracking, and multimodal representation learning. We thank the authors and maintainers of the datasets, models, libraries, and baseline systems referenced in the paper and documented under `third_party/`.

## Contact

For questions, bug reports, and reproducibility issues, please use the repository's [GitHub Issues](https://github.com/jhxzcgkeh/EVIQUE/issues).

---

<div align="center">
  <sub>Developed by the EVIQUE authors at Northeastern University, China.</sub>
</div>
