from __future__ import annotations
from pathlib import Path
from typing import Any
from evique.schema import ScopeViewRecord
from evique.utils import read_jsonl

def load_scope_view(index_dir: str | Path) -> list[dict[str, Any]]:
    """Load `scope_view.jsonl` from a built EVIQUE index directory."""
    return read_jsonl(Path(index_dir) / "scope_view.jsonl")

__all__ = ["ScopeViewRecord", "load_scope_view"]
