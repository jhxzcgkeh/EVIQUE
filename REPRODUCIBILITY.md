# Reproducibility

This document describes the reference environment and the procedures required to reproduce the EVIQUE experiments.

## 1. Reference software environment

The reference software environment is:

| Component | Version |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.11.15 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| torchaudio | 2.1.2 |
| PyTorch CUDA build | 12.1 |
| System CUDA Toolkit | 11.8 |
| Transformers | 4.37.1 |
| Accelerate | 1.13.0 |
| BitsAndBytes | 0.43.1 |
| Ultralytics | 8.4.52 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |
| SciPy | 1.17.1 |
| OpenAI Python client | 2.32.0 |
| Tiktoken | 0.12.0 |
| FFmpeg | Required |

The system CUDA Toolkit version and the CUDA version used to build the PyTorch wheel are different concepts. In the reference environment:

- the system CUDA Toolkit is 11.8;
- the PyTorch wheel is built for CUDA 12.1.

The selected PyTorch wheel should be installed using the command in `INSTALL.md`.

## 2. Reference experiment hardware

The verified paper experiment configuration is:

- one NVIDIA RTX 4090D GPU with 24 GB GPU memory;
- 16 virtual CPU cores;
- Intel Xeon Platinum 8352S CPU at 2.20 GHz;
- 24 GB system memory;
- Ubuntu 22.04.

The exact runtime may vary with storage performance, API latency, model serving speed, and the selected baseline.

## 3. Dependency groups

The repository separates dependencies into three groups.

### Core dependencies

```text
requirements.txt
```

These packages support:

* EVIQUE core logic;
* configuration;
* LLM invocation;
* evaluation;
* aggregation;
* result analysis;
* table and figure generation.

### Vision dependencies

```text
requirements-vision.txt
```

These packages support:

* object detection;
* video decoding;
* multimodal processing;
* caption generation;
* ASR;
* GPU-based visual indexing.

### Baseline dependencies

```text
requirements-baselines.txt
```

These packages support optional DB and RAG baselines. Some third-party methods may require isolated environments or external source repositories.

## 4. External artifacts

The repository does not include:

* raw video datasets;
* model checkpoints;
* API credentials;
* complete experiment work directories;
* generated indexes;
* large answer collections;
* third-party repositories whose licenses do not permit redistribution.

Dataset and model preparation instructions are provided in `DATASETS.md` and `MODELS.md`.

## 5. Randomness and repeated evaluation

All experiment scripts should expose a random seed when stochastic processing is used.

The reproduction documentation must record:

* random seed;
* number of evaluation runs;
* query count;
* dataset split;
* top-k value;
* window size and stride;
* single-pass or order-balanced pairwise evaluation;
* judge model;
* answer model;
* embedding model.

Do not compare results generated with different evaluation protocols without explicitly identifying the difference.

## 6. Multi-video datasets

QVHighlights is evaluated as a multi-video dataset.

Do not force a single-video filter or a single `EVIQUE_VIDEO_PATH` for a multi-video run. Each query must preserve its source-video metadata.

## 7. Visual-relation setting

The repository contains experiments both with and without explicit visual-relation evidence.

The default main configuration must clearly state whether visual relations are enabled:

```bash
export EVIQUE_VISUAL_MODE=hybrid
export EVIQUE_DISABLE_VISUAL_RELATIONS=1
```

Experiments using visual relations must be labeled separately and must not be mixed with the default no-visual-relations results.

## 8. API-based experiments

The answer-generation and judge stages may use an OpenAI-compatible remote API.

The required configuration is supplied through environment variables:

```bash
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
OPENAI_EMBEDDING_MODEL
```

The repository must not include real credentials.

API latency, rate limits, provider revisions, and nondeterministic model behavior may cause small runtime or evaluation differences.

## 9. Smoke reproduction

The smoke test validates:

* package imports;
* configuration loading;
* query parsing;
* result schema;
* metric computation;
* adapter registration;
* command-line interfaces;
* a minimal retrieval path.

It does not reproduce every full-scale paper result.

Run:

```bash
bash scripts/smoke/run_smoke.sh
```

## 10. Full reproduction workflow

The full workflow consists of:

1. downloading and preparing datasets;
2. downloading required model checkpoints;
3. building or restoring EVIQUE indexes;
4. running DB retrieval experiments;
5. running RAG comparison experiments;
6. running core ablations;
7. running evidence-view ablations;
8. generating quantitative and pairwise evaluations;
9. aggregating results;
10. generating paper tables and figures.

The exact commands and paper-to-artifact mapping are documented in `ARTIFACT.md`.

## 11. Expected differences

Exact API-based judge scores may vary slightly because remote language models can be nondeterministic.

Structural outputs must remain consistent:

* output schema;
* method names;
* query counts;
* dataset identifiers;
* evidence fields;
* metric definitions;
* aggregation logic.

Large deviations should be treated as reproduction failures and investigated.

## 12. Validation

Before releasing the repository, run:

```bash
python -m pip check
python -m compileall .
python -m pytest -q
```

Also verify:

```bash
python run_db_baseline_comparison.py --help
python run_rag_comparison.py --help
python run_evique_ablation_db_rag.py --help
```

Record all results in `VALIDATION_REPORT.md`.