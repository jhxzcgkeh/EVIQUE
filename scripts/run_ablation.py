from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, os, subprocess, sys
from pathlib import Path
VARIANT_ENV = {"full":{}, "no_scope":{"EVIQUE_ABLATION_DISABLE_VIEWS":"scope"}, "no_target":{"EVIQUE_ABLATION_DISABLE_VIEWS":"target"}, "no_track":{"EVIQUE_ABLATION_DISABLE_VIEWS":"track"}, "no_event":{"EVIQUE_ABLATION_DISABLE_VIEWS":"event,adaptive_event,fixed_window_event,visual_event"}, "fixed_window_event":{"EVIQUE_ABLATION_EVENT_MODE":"fixed_window"}}
def parse_args():
    p = argparse.ArgumentParser(description="Run EVIQUE ablations over real first-party switches.")
    p.add_argument("--index-dir", type=Path, default=Path("outputs/demo_index")); p.add_argument("--query-file", type=Path, default=Path("examples/minimal_query.json")); p.add_argument("--output-dir", type=Path, default=Path("outputs/ablation")); p.add_argument("--variants", nargs="*", default=list(VARIANT_ENV))
    return p.parse_args()
def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        if v not in VARIANT_ENV: raise SystemExit(f"unknown variant: {v}")
        env = os.environ.copy(); env.update(VARIANT_ENV[v]); out = a.output_dir / f"{v}.json"
        subprocess.run([sys.executable, str(Path(__file__).with_name("run_evique.py")), "--index-dir", str(a.index_dir), "--query-file", str(a.query_file), "--output", str(out)], check=True, env=env); print(out)
if __name__ == "__main__": main()
