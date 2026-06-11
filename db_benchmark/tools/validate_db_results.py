from __future__ import annotations

import argparse
from pathlib import Path

from db_benchmark.schema import normalize_result_record, validate_result_record
from db_benchmark.utils import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DB benchmark result JSONL files.")
    parser.add_argument("results", nargs="+")
    args = parser.parse_args(argv)

    total = 0
    errors = 0
    paths = _expand_paths(args.results)
    if not paths:
        print("no result files matched")
        return 1
    for path in paths:
        rows = read_jsonl(path)
        for idx, row in enumerate(rows, start=1):
            total += 1
            normalized = normalize_result_record(row)
            row_errors = validate_result_record(normalized)
            if row_errors:
                errors += 1
                print(f"{path}:{idx}: {'; '.join(row_errors)}")
    print(f"validated_rows={total} errors={errors}")
    return 1 if errors else 0


def _expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if any(char in value for char in "*?["):
            raw = Path(value)
            parent = raw.parent if str(raw.parent) else Path(".")
            matches = sorted(parent.glob(raw.name))
            paths.extend(path for path in matches if path.is_file())
        else:
            path = Path(value)
            if path.is_file():
                paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
