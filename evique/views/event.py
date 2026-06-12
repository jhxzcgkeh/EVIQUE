from __future__ import annotations
from pathlib import Path
from typing import Any
from evique.schema import EventViewRecord
from evique.utils import read_jsonl

def load_event_view(index_dir: str | Path) -> list[dict[str, Any]]:
    """Load `event_view.jsonl` from a built EVIQUE index directory."""
    return read_jsonl(Path(index_dir) / "event_view.jsonl")

__all__ = ["EventViewRecord", "load_event_view"]
