# Models

No weights are included. `yolo11n.pt` from the source working directory is excluded.

| Model | Role | Expected path | Included |
|---|---|---|---|
| YOLO11n | detector | `models/yolo11n.pt` | no |
| Caption model | captions | `models/<caption-model>/` or API | no |
| ASR model | transcripts | `models/faster-whisper-*` or API | no |
| Embedding model | retrieval | `.env`/CLI | no |
| LLM/Judge | answer/eval | `.env`/CLI | no |
