# Reproducibility

- OS: Windows and Linux paths are supported through relative paths and CLI arguments.
- Python: 3.10-3.12 recommended.
- GPU/CUDA/PyTorch: only required for full visual/model runs, not smoke tests.
- Remote API: not required for dry-run smoke; required for LLM answer generation and judging.
- Seeds/runtime/model versions must be recorded in run manifests.
- Smoke validates imports, schemas, and dry-run plumbing; it is not a full paper result reproduction.
- Deterministic metrics should match exactly for identical inputs; LLM-judge results require model/date provenance.
