from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json
from pathlib import Path
from evique.retrieval import EvidenceRetriever
from evique.utils.query_io import QueryRecord, load_query_file

def parse_args():
    p = argparse.ArgumentParser(description="Run EVIQUE retrieval/planning/packing for a query.")
    p.add_argument("--index-dir", type=Path, default=Path("outputs/demo_index")); p.add_argument("--query", default="")
    p.add_argument("--query-file", type=Path); p.add_argument("--output", type=Path, default=Path("outputs/evique_result.json"))
    p.add_argument("--max-evidence", type=int, default=18); p.add_argument("--token-budget", type=int, default=12000)
    return p.parse_args()

def queries(a):
    if a.query_file:
        return load_query_file(a.query_file).queries
    if a.query:
        return [QueryRecord("adhoc_001", "adhoc", "", a.query)]
    raise SystemExit("provide --query or --query-file")

def main():
    a = parse_args(); retriever = EvidenceRetriever(a.index_dir, max_evidence=a.max_evidence, token_budget=a.token_budget); rows=[]
    for q in queries(a): rows.append({"query": q.to_dict(), "method": "EVIQUE", "evidence_package": retriever.retrieve(q.query, {"video_id": q.video_id, "dataset": q.dataset})})
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); print(a.output)
if __name__ == "__main__": main()
