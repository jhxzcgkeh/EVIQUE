from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path

from evique.standalone_base_builder import build_evique_standalone_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EVIQUE standalone base JSON files.")
    parser.add_argument("--video", action="append", required=True, help="Video path. Can be passed multiple times.")
    parser.add_argument("--output-base-dir", default=None, help="Base output directory. Defaults to <evique-workdir>/base.")
    parser.add_argument("--evique-workdir", default=None, help="EVIQUE workdir used when --output-base-dir is omitted.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--fine-num-frames", type=int, default=15)
    parser.add_argument("--rough-num-frames", type=int, default=15)
    parser.add_argument("--segment-length", type=int, default=30)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate even if standalone base files already exist.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL"))
    parser.add_argument("--embedding-dim", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    args = parse_args()
    if args.output_base_dir:
        output_base_dir = Path(args.output_base_dir)
    elif args.evique_workdir:
        output_base_dir = Path(args.evique_workdir) / "base"
    else:
        raise SystemExit("Pass --output-base-dir or --evique-workdir.")
    result = build_evique_standalone_base(
        video_paths=args.video,
        output_base_dir=output_base_dir,
        dataset_name=args.dataset_name,
        chunk_token_size=args.chunk_token_size,
        fine_num_frames=args.fine_num_frames,
        rough_num_frames=args.rough_num_frames,
        segment_length=args.segment_length,
        rebuild=args.rebuild,
        model_name=args.model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )
    printable = {key: value for key, value in result.items() if key not in {"video_segments", "text_chunks"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

