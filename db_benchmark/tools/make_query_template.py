from __future__ import annotations

import argparse
import json
from pathlib import Path

from db_benchmark.schema import default_query_payload
from db_benchmark.utils import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a DB benchmark query JSON template.")
    parser.add_argument("--output")
    parser.add_argument("--dataset", default="bellevue")
    parser.add_argument("--video-id", default="bellevue_11_090831")
    args = parser.parse_args(argv)

    payload = default_query_payload()
    payload["dataset"] = args.dataset
    payload["video_id"] = args.video_id
    if args.output:
        write_json(payload, Path(args.output))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

