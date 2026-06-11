# Validation Report

## Environment

- OS: Windows-11-10.0.26200-SP0 for local validation; target reference platform remains Ubuntu 22.04.
- Python: current shell Python 3.13.5; isolated validation environment Python 3.11.15.
- PyTorch: 2.1.2+cpu in the isolated validation environment; not installed in the current shell Python.
- Installation mode: isolated Conda prefix outside the release repository, then local repository import from the EVIQUE checkout.
- Validation date: 2026-06-11.

## Dependency Installation

| Check | Command | Status | Notes |
|---|---|---:|---|
| Core installation | `pip install -r requirements.txt` | Pass | Completed in the isolated Python 3.11.15 environment. |
| Reference PyTorch CPU installation | `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu` | Pass | Installed `torch==2.1.2+cpu`, `torchvision==0.16.2+cpu`, and `torchaudio==2.1.2+cpu`. |
| Vision installation | `pip install -r requirements-vision.txt` | Pass | Completed. Direct requirements include only `opencv-python-headless`; `ultralytics` pulled `opencv-python` transitively. No requirements file was changed. |
| Baseline installation | `pip install -r requirements-baselines.txt` | Fail | The pinned ImageBind commit declares `torch==1.13.1`, which cannot be resolved for the Python 3.11 validation environment. Baseline reproduction remains blocked until this baseline is isolated, patched, or documented with a compatible interpreter/runtime. |
| Dependency consistency | `python -m pip check` | Pass | No broken requirements found after core, vision, reference PyTorch, and dev dependency installation. |

## Imports

| Module | Command | Status | Notes |
|---|---|---:|---|
| EVIQUE | `python -c "import evique"` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |
| DB benchmark | `python -c "import db_benchmark"` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |
| PyTorch | `python -c "import torch"` | Pass | Passed in isolated environment with version `2.1.2+cpu`; current shell Python does not have PyTorch installed. |
| Transformers | `python -c "import transformers"` | Pass | Passed in isolated environment with version `4.37.1`; current shell Python does not have it installed. |
| Ultralytics | `python -c "import ultralytics"` | Pass | Passed in isolated environment with version `8.4.52`; current shell Python does not have it installed. |

## Command-Line Interfaces

| CLI | Status | Notes |
|---|---:|---|
| `run_db_baseline_comparison.py --help` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |
| `run_rag_comparison.py --help` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |
| `run_evique_ablation_db_rag.py --help` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |

## Static Checks

| Check | Command | Status | Notes |
|---|---|---:|---|
| Compile | `python -m compileall .` | Pass | Passed in current shell and isolated Python 3.11.15 environment. |
| Unit tests | `python -m pytest -q` | Pass | 4 tests passed in current shell and isolated Python 3.11.15 environment. |
| Credential/path broad scan | request-provided broad rg pattern | Review pass | Matches are placeholders, environment-variable reads, audit inventory rows, or excluded third-party provenance; no real credential or machine path was found by the stricter high-confidence scan. |
| Absolute-path scan | local/server path rg pattern | Pass | No current release-file hit for local Windows user path, server workdir path, or private endpoint. |
| Legacy-name scan | request-provided legacy-name rg pattern | Review pass | Matches are confined to migration/provenance files: `RENAME_REPORT.md`, `RENAMING_MAP.md`, and `REPOSITORY_FILE_INVENTORY.csv`. Public code and install/runtime docs have zero matches. |

## Smoke Test

- Command: `powershell -ExecutionPolicy Bypass -File scripts/smoke/run_smoke.ps1`
- Status: Pass
- Runtime: under 1 second on the current shell environment
- Output directory: `outputs/smoke` during validation; removed before final packaging cleanup
- Notes: Also passed with `python scripts/smoke/smoke_check.py` in the isolated Python 3.11.15 environment.

## Known Limitations

- The current shell Python is 3.13.5 and lacks PyTorch, Transformers, and Ultralytics; full dependency validation used the isolated Python 3.11.15 environment.
- Optional baseline installation is blocked by the pinned ImageBind commit requiring `torch==1.13.1`, which is incompatible with the Python 3.11 validation environment.
- `ultralytics==8.4.52` pulls `opencv-python` transitively even though EVIQUE directly specifies only `opencv-python-headless`.
- Full experiments still require external datasets, model checkpoints, API access, and substantially more compute than the smoke test.
- Public release remains blocked until the project license is selected and third-party baseline licensing/runtime limitations are reviewed.

## Release Decision

- Ready for public release: No
- Blocking issues: project license placeholder; optional baseline dependency conflict; external datasets/checkpoints/API credentials not distributed.
- Recommended actions: select the project license, decide whether ImageBind needs a separate compatible baseline environment, run full Linux GPU validation, and update this report after full baseline/runtime verification.