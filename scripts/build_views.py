from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json
from pathlib import Path
from evique.evidence.graph import build_evidence_graph

def parse_args():
    p = argparse.ArgumentParser(description="Build EVIQUE Scope/Target/Track/Event views and evidence graph.")
    p.add_argument("--segments-json", type=Path, default=Path("examples/demo_segments.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/demo_index"))
    p.add_argument("--event-window-seconds", type=int, default=120)
    p.add_argument("--track-gap-seconds", type=int, default=120)
    return p.parse_args()

def main():
    a = parse_args(); segments = json.loads(a.segments_json.read_text(encoding="utf-8"))
    manifest = build_evidence_graph(video_segments=segments, output_dir=a.output_dir, event_window_seconds=a.event_window_seconds, track_gap_seconds=a.track_gap_seconds)
    print(json.dumps({"index_dir": str(a.output_dir), "manifest": manifest}, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
