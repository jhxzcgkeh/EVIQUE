from __future__ import annotations
from pathlib import Path
from typing import Any
from evique.builder import build_evique_from_segments
from evique.utils import read_json, read_jsonl

def build_evidence_graph(*, video_segments: dict[str, dict[str, dict[str, Any]]] | None = None, video_segments_path: str | Path | None = None, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return build_evique_from_segments(video_segments=video_segments, video_segments_path=Path(video_segments_path) if video_segments_path else None, output_dir=Path(output_dir), **kwargs)

def load_evidence_graph(index_dir: str | Path) -> dict[str, Any]:
    root = Path(index_dir)
    stats = root / "graph_stats.json"
    return {"nodes": read_jsonl(root / "evidence_nodes.jsonl"), "relations": read_jsonl(root / "evidence_relations.jsonl"), "stats": read_json(stats) if stats.exists() else {}}
