# Installation

EVIQUE is developed and tested on Linux with Python 3.11. A GPU is recommended for visual index construction and video processing. Evaluation and paper-table generation can run on CPU after the required predictions have been produced.

## 1. System requirements

Recommended software:

- Ubuntu 22.04 or a compatible Linux distribution
- Python 3.11
- Conda or Miniconda
- Git
- FFmpeg
- An NVIDIA GPU for full visual experiments
- A CUDA-compatible NVIDIA driver for GPU execution

FFmpeg must be available from the command line:

```bash
ffmpeg -version
ffprobe -version
```

## 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate evique
```

Alternatively:

```bash
conda create -n evique python=3.11 -y
conda activate evique
pip install -r requirements.txt
```

## 3. Install PyTorch

The reference environment used PyTorch 2.1.2 with the CUDA 12.1 wheel.

### NVIDIA GPU installation

```bash
pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

### CPU-only installation

```bash
pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cpu
```

Do not manually install the complete collection of `nvidia-*` packages from the original server environment. The selected PyTorch wheel installs the compatible runtime dependencies.

## 4. Install EVIQUE vision dependencies

```bash
pip install -r requirements-vision.txt
```

This installs the dependencies used for:

* video decoding;
* visual object detection;
* vision-language processing;
* visual index construction;
* speech recognition.

The default OpenCV package is `opencv-python-headless`, because the experiments do not require a desktop GUI.

## 5. Install optional baseline dependencies

Install these packages only when reproducing the complete DB and RAG baseline experiments:

```bash
pip install -r requirements-baselines.txt
```

Some third-party baselines require additional repositories, checkpoints, or their own environments. Consult the baseline documentation before running them.

## 6. Configure external models and APIs

Create a local `.env` file from `.env.example` or export the required variables manually.

Example:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="YOUR_OPENAI_COMPATIBLE_ENDPOINT"
export OPENAI_MODEL="YOUR_LLM"
export OPENAI_EMBEDDING_MODEL="YOUR_EMBEDDING_MODEL"

export EVIQUE_DETECTOR_MODEL="/path/to/yolo11n.pt"
export EVIQUE_DATA_ROOT="/path/to/datasets"
```

Never commit API keys, tokens, model weights, dataset files, or machine-specific absolute paths.

## 7. External models

The following models are not distributed in this repository:

* YOLO11n detector weights;
* vision-language caption model;
* ASR model;
* embedding model;
* remote LLM or judge model.

Download them from their official sources and configure their locations through command-line arguments or environment variables.

## 8. Verify the installation

```bash
python -m pip check
python -m compileall evique db_benchmark
```

Verify package imports:

```bash
python -c "import evique; print(evique.__file__)"
python -c "import db_benchmark; print(db_benchmark.__file__)"
```

Verify the command-line interfaces:

```bash
python run_db_baseline_comparison.py --help
python run_rag_comparison.py --help
python run_evique_ablation_db_rag.py --help
```

Run the smoke test:

```bash
bash scripts/smoke/run_smoke.sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke/run_smoke.ps1
```

## 9. Full reproduction

The complete experiments require external datasets, model weights, API access, and substantially more compute than the smoke test.

See:

* `REPRODUCIBILITY.md`
* `DATASETS.md`
* `MODELS.md`
* `ARTIFACT.md`

for the complete reproduction procedures.