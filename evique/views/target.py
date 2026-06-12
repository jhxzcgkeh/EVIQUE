from __future__ import annotations
from pathlib import Path
from typing import Any
from evique.schema import TargetViewRecord
from evique.utils import read_jsonl

def load_target_view(index_dir: str | Path) -> list[dict[str, Any]]:
    """Load `target_view.jsonl` from a built EVIQUE index directory."""
    return read_jsonl(Path(index_dir) / "target_view.jsonl")

__all__ = ["TargetViewRecord", "load_target_view"]
