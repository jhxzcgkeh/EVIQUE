from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MIRIS_EXEC_PREDICATES = {
    "shibuya",
    "shibuya-crosswalk",
    "warsaw",
    "warsaw-brake",
    "beach-runner",
    "uav",
}

MIRIS_PREDICATE_ALIASES = {
    "shibuya": "shibuya",
    "shibuya-crosswalk": "shibuya-crosswalk",
    "shibuya crosswalk": "shibuya-crosswalk",
    "crosswalk stop": "shibuya-crosswalk",
    "warsaw": "warsaw",
    "warsaw-brake": "warsaw-brake",
    "warsaw brake": "warsaw-brake",
    "hard brake": "warsaw-brake",
    "hard braking": "warsaw-brake",
    "beach-runner": "beach-runner",
    "beach runner": "beach-runner",
    "uav": "uav",
}


def map_query_to_miris(query_text: str, query_type: str = "") -> dict[str, Any]:
    text = _normalize(query_text)
    query_type_l = _normalize(query_type)
    explicit = _extract_explicit_predicate(text)
    if explicit:
        return _supported_mapping(explicit, "explicit_miris_predicate")

    for phrase, predicate in sorted(MIRIS_PREDICATE_ALIASES.items(), key=lambda item: -len(item[0])):
        if _phrase_present(text, phrase):
            return _supported_mapping(predicate, f"miris_alias:{phrase}")

    if _looks_like_lava_db_query(text, query_type_l):
        return _unsupported_mapping(
            "LAVA class-level DB query is not one of MIRIS official built-in object-track predicates."
        )
    if any(word in text for word in ["cross", "enter", "leave", "trajectory", "turn", "brake", "speed", "move"]):
        return _unsupported_mapping(
            "Query is motion-like, but MIRIS official code only exposes fixed built-in predicates "
            "(shibuya, shibuya-crosswalk, warsaw, warsaw-brake, beach-runner, uav)."
        )
    return _unsupported_mapping(
        "Query text does not explicitly map to a MIRIS official predicate."
    )


def build_miris_query_input(
    *,
    query: dict[str, Any],
    dataset: str,
    video_id: str,
    video_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = map_query_to_miris(str(query.get("query") or ""), str(query.get("type") or ""))
    payload = {
        "source_repo": "official_miris",
        "query_id": str(query.get("query_id") or ""),
        "query": str(query.get("query") or ""),
        "query_type": str(query.get("type") or ""),
        "dataset": str(query.get("dataset") or dataset or ""),
        "video_id": str(query.get("video_id") or video_id or ""),
        "video_path": str(video_path or ""),
        "query_mapping": mapping,
        "miris_input_layout": {
            "required_detection_json": "data/<miris_dataset>/json/<segment>-detections.json",
            "required_track_json": "data/<miris_dataset>/json/<segment>-baseline.json",
            "required_frame_dir": "data/<miris_dataset>/frames/<segment>/",
            "detection_schema": ["frame_idx", "track_id", "left", "top", "right", "bottom", "score"],
        },
        "gt_ignored": True,
    }
    out_path = output_dir / f"{payload['query_id'] or 'query'}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["original_input_path"] = str(out_path)
    return payload


def convert_query_file(queries_path: Path, output_dir: Path, *, video_path: str = "") -> list[dict[str, Any]]:
    payload = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    dataset = str(payload.get("dataset") or "")
    video_id = str(payload.get("video_id") or "")
    queries = payload.get("queries") or []
    rows = []
    for query in queries:
        if not isinstance(query, dict):
            continue
        rows.append(
            build_miris_query_input(
                query=query,
                dataset=dataset,
                video_id=video_id,
                video_path=video_path,
                output_dir=Path(output_dir),
            )
        )
    return rows


def _supported_mapping(predicate: str, reason: str) -> dict[str, Any]:
    return {
        "supported": True,
        "miris_predicate": predicate,
        "supported_query_type": "official_miris_object_track_predicate",
        "mapping_reason": reason,
        "unsupported_reason": "",
    }


def _unsupported_mapping(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "miris_predicate": "",
        "supported_query_type": "",
        "mapping_reason": "",
        "unsupported_reason": reason,
    }


def _extract_explicit_predicate(text: str) -> str:
    patterns = [
        r"\bmiris\s+predicate\s+([a-z0-9_-]+)\b",
        r"\bpredicate\s+([a-z0-9_-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in MIRIS_EXEC_PREDICATES:
            return match.group(1)
    return ""


def _looks_like_lava_db_query(text: str, query_type: str) -> bool:
    if query_type in {"existence", "counting", "attribute_size", "spatial", "motion"}:
        return True
    lava_terms = [
        "visible",
        "appears",
        "multiple",
        "large",
        "near center",
        "closest to the center",
        "boat",
        "bicycle",
        "vehicle",
        "car",
        "truck",
        "traffic object",
    ]
    return any(term in text for term in lava_terms)


def _phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DB query text into conservative MIRIS adapter input manifests.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", default="")
    args = parser.parse_args()
    rows = convert_query_file(Path(args.queries), Path(args.output_dir), video_path=args.video)
    print(json.dumps({"query_count": len(rows), "output_dir": str(Path(args.output_dir))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
