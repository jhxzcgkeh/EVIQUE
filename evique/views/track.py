from __future__ import annotations
from pathlib import Path
from typing import Any
from evique.schema import TrackViewRecord
from evique.utils import read_jsonl

def load_track_view(index_dir: str | Path) -> list[dict[str, Any]]:
    """Load `track_view.jsonl` from a built EVIQUE index directory."""
    return read_jsonl(Path(index_dir) / "track_view.jsonl")

__all__ = ["TrackViewRecord", "load_track_view"]
