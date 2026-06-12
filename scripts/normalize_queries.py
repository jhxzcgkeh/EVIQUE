from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evique.utils.query_io import (  # noqa: E402
    canonical_dataset,
    load_query_file,
    normalize_payload,
    read_raw_query_file,
    sha256_file,
    summarize_dataset,
    validate_query_dataset,
    write_query_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize legacy query files into the public EVIQUE query schema.")
    parser.add_argument("--input", required=True, type=Path, help="Source query file.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Warsaw.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected query count for strict validation.")
    parser.add_argument("--default-video-id", default=None, help="Default video id for single-video datasets.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on schema/count/id/text/video validation errors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw = read_raw_query_file(args.input)
        dataset = normalize_payload(
            raw.payload,
            dataset=canonical_dataset(args.dataset),
            source_format=raw.format,
            default_video_id=args.default_video_id,
        )
        errors = validate_query_dataset(dataset, expected_count=args.expected_count, strict=False)
        digest = write_query_dataset(dataset, args.output)
        summary = summarize_dataset(dataset)
        print(f"Input records: {raw.records}")
        print(f"Output records: {summary['records']}")
        print(f"Generated IDs: {summary['generated_ids']}")
        print(f"Duplicate IDs: {summary['duplicate_ids']}")
        print(f"Duplicate query texts: {summary['duplicate_query_texts']}")
        print(f"Missing video IDs: {summary['missing_video_ids']}")
        print(f"Empty queries: {summary['empty_queries']}")
        print(f"Output path: {args.output}")
        print(f"SHA-256: {digest}")
        if raw.parse_errors:
            print("Parser fallbacks: " + " | ".join(raw.parse_errors[:3]))
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"- {error}")
        if args.strict and (errors or raw.records == 0):
            return 1
        return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
