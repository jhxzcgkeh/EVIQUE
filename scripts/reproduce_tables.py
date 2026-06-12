from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, csv

def parse_args():
    p = argparse.ArgumentParser(description="Reproduce paper table CSVs from results/paper CSV files.")
    p.add_argument("--results-dir", type=Path, default=Path("results/paper"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    return p.parse_args()

def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["main_results", "ablation_results", "efficiency_results", "query_manifest"]:
        src = a.results_dir / f"{name}.csv"
        rows = list(csv.DictReader(src.open("r", encoding="utf-8"))) if src.exists() else []
        fields = list(rows[0]) if rows else (src.read_text(encoding="utf-8").splitlines()[0].split(",") if src.exists() else [])
        dst = a.output_dir / f"{name}.csv"
        with dst.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    print(a.output_dir)

if __name__ == "__main__": main()
