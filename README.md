# EVIQUE

**EVIQUE: Utility-Aware Evidence Construction for Grounded Answering over Large-Scale Video Databases**

EVIQUE constructs compact and traceable evidence packages for grounded question answering over large-scale video databases. It organizes video observations into Scope, Target, Track, and Event views, connects them through an evidence graph, and prepares evidence for a downstream language model.

## Installation

```bash
git clone https://github.com/jhxzcgkeh/EVIQUE.git
cd EVIQUE
python -m pip install -r requirements.txt
```

## Usage

### 1. Build evidence views

Prepare the segment-level input and build the Scope, Target, Track, and Event views:

```bash
python scripts/build_views.py   --segments-json examples/demo_segments.json   --output-dir outputs/demo_index
```

The generated index is written to `outputs/demo_index`.

### 2. Run EVIQUE

Run a query over the generated evidence index:

```bash
python scripts/run_evique.py   --index-dir outputs/demo_index   --query-file examples/minimal_query.json
```

The query file should contain the natural-language question and the required query metadata. EVIQUE retrieves relevant evidence, expands connected evidence through the evidence graph, and produces a compact evidence package for answer generation.

### 3. Run ablation experiments

Run the public ablation entry point with the same index and query input:

```bash
python scripts/run_ablation.py   --index-dir outputs/demo_index   --query-file examples/minimal_query.json
```

Available options can be inspected with:

```bash
python scripts/run_ablation.py --help
```

### 4. Reproduce tables and figures

Generate the compact paper tables and figures from the files under `results/paper/`:

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

### 5. View command-line options

```bash
python scripts/build_views.py --help
python scripts/run_evique.py --help
python scripts/run_ablation.py --help
python scripts/reproduce_tables.py --help
python scripts/reproduce_figures.py --help
```

## Citation

```bibtex
@misc{li2026evique,
  title  = {EVIQUE: Utility-Aware Evidence Construction for Grounded Answering over Large-Scale Video Databases},
  author = {Ji Li and Lan Jin and Ruikai Zhu and Chuanwen Li},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

This project is released under the [MIT License](LICENSE).
