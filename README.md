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

Build the evidence views:

```bash
python scripts/build_views.py   --segments-json examples/demo_segments.json   --output-dir outputs/demo_index
```

Run EVIQUE:

```bash
python scripts/run_evique.py   --index-dir outputs/demo_index   --query-file examples/minimal_query.json
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
