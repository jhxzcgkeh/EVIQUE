$ErrorActionPreference = "Stop"
python -m evique.cli.check_standalone
python scripts/smoke/smoke_check.py
