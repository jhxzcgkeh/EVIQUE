#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from db_rag_pipeline_common import load_query_input, write_jsonl


DEFAULT_DATASET_CONFIGS = {
    "beach": {
        "video": "<repo>/Dataset/Beach/beach_52min.mp4",
        "run": "db_rag_beach_21q_9methods_v1",
    },
    "warsaw": {
        "video": "/root/autodl-tmp/Dataset/Beach/miris/ytstream-dataset/data/warsaw/videos/4.mp4",
        "run": "db_rag_warsaw_30q_9methods_v1",
    },
    "qvhighlights": {
        "video_dir": "<repo>/Dataset/Qvhighlights/videos",
        "run": "db_rag_qvhighlights_20q_9methods_v1",
    },
    "bellevue": {
        "video": "/root/autodl-tmp/Dataset/Bellevue/111Bellevue_150th_Eastgate__2017-09-10_19-08-25.mp4",
        "run": "db_rag_bellevue_25q_9methods_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize DB-RAG question files into queries.jsonl without generating new queries."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input txt/json/jsonl query file.")
    parser.add_argument("--output", required=True, type=Path, help="Output queries.jsonl path.")
    parser.add_argument("--dataset", required=True, help="Dataset name written to each query row.")
    parser.add_argument("--video", default="", help="Single video path for this query set.")
    parser.add_argument("--video-dir", default="", help="Video directory for multi-video query sets.")
    parser.add_argument("--video-id", default="", help="Optional video id override.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Optional smoke limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DEFAULT_DATASET_CONFIGS.get(args.dataset.lower(), {})
    video = args.video or str(defaults.get("video") or "")
    video_dir = args.video_dir or str(defaults.get("video_dir") or "")
    rows = load_query_input(
        args.input,
        dataset=args.dataset,
        video_path=video,
        video_dir=video_dir,
        video_id=args.video_id,
        limit=args.limit_queries or None,
    )
    if not rows:
        raise ValueError(f"no queries found in {args.input}")
    write_jsonl([row.to_json() for row in rows], args.output)
    print(f"wrote {len(rows)} queries to {args.output}")


if __name__ == "__main__":
    main()
