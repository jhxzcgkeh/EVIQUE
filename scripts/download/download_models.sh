#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO
YOLO('yolo11n.pt')
p=Path('yolo11n.pt')
if p.exists():
    q=Path('models')/'yolo11n.pt'; q.write_bytes(p.read_bytes()); p.unlink(); print(q)
PY
