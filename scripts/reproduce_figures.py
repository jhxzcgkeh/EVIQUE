from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, csv
from pathlib import Path
def parse_args():
    p = argparse.ArgumentParser(description="Generate lightweight paper figures from efficiency_results.csv."); p.add_argument("--input", type=Path, default=Path("results/paper/efficiency_results.csv")); p.add_argument("--output-dir", type=Path, default=Path("outputs/figures")); return p.parse_args()
def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True); rows = list(csv.DictReader(a.input.open("r", encoding="utf-8"))) if a.input.exists() else []
    parts = []
    for i,row in enumerate(rows[:20]):
        try: value = float(row.get("value") or 0)
        except ValueError: value = 0.0
        y = 30 + i*24; width = max(1, min(360, int(value))); label = f"{row.get('figure','')}: {row.get('dataset','')} {row.get('method','')} {row.get('metric','')}"
        parts.append(f'<text x="10" y="{y}" font-size="10">{label}</text><rect x="260" y="{y-10}" width="{width}" height="12" fill="#3b82f6"/>')
    if not parts: parts.append('<text x="10" y="30" font-size="12">No efficiency rows available yet.</text>')
    out = a.output_dir / "efficiency_overview.svg"; out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="{max(80, 50+len(parts)*24)}">' + "".join(parts) + "</svg>", encoding="utf-8"); print(out)
if __name__ == "__main__": main()
