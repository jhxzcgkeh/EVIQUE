$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force models | Out-Null
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
if (Test-Path yolo11n.pt) { Move-Item -Force yolo11n.pt models/yolo11n.pt }
