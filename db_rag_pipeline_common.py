from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DB_RAG_METHODS = ["EVIQUE", "LOVO", "VOCAL", "MIRIS", "OTIF", "UMT", "VISA", "FiGO", "ZELDA"]
DB_RAG_METHOD_FIDELITY = {
    "EVIQUE": "native",
    "LOVO": "local_reproduction",
    "VOCAL": "third_party_proxy",
    "MIRIS": "third_party_proxy",
    "OTIF": "third_party_proxy",
    "UMT": "third_party_proxy",
    "VISA": "third_party_proxy",
    "FiGO": "local_reimplementation",
    "ZELDA": "local_reimplementation",
}
QUANT_METRICS = ["Comprehensiveness", "Empowerment", "Trustworthiness", "Depth", "Density", "Overall Score"]
WINRATE_METRICS = ["Comprehensiveness", "Empowerment", "Trustworthiness", "Depth", "Density", "Overall Winner"]


class ProgressReporter:
    def __init__(self, *, total: int, enabled: bool = True, desc: str = "progress", unit: str = "task"):
        self.total = max(int(total or 0), 0)
        self.enabled = bool(enabled)
        self.desc = desc
        self.unit = unit
        self.completed = 0
        self.start_time = time.perf_counter()
        self._bar = None
        if self.enabled and self.total > 0:
            try:
                from tqdm import tqdm  # type: ignore

                self._bar = tqdm(total=self.total, desc=desc, unit=unit, dynamic_ncols=True, file=sys.stderr)
            except Exception:
                self._bar = None

    def log(self, message: str) -> None:
        if self._bar is not None:
            try:
                self._bar.write(str(message), file=sys.stderr)
                sys.stderr.flush()
                return
            except Exception:
                pass
        print(str(message), flush=True)

    def update(self, n: int = 1, *, postfix: dict[str, Any] | None = None) -> None:
        self.completed = min(self.total if self.total else self.completed + n, self.completed + n)
        if self._bar is not None:
            if postfix:
                self._bar.set_postfix({key: _short_postfix_value(value) for key, value in postfix.items()}, refresh=False)
            self._bar.update(n)
            self._bar.refresh()
            return
        if self.enabled and self.total > 0:
            details = progress_details(self.completed, self.total, self.start_time)
            suffix = ""
            if postfix:
                suffix = " " + " ".join(f"{key}={_short_postfix_value(value)}" for key, value in postfix.items())
            print(
                f"[{self.desc}] completed={details['completed']}/{details['total']} "
                f"elapsed={details['elapsed']} avg={details['avg_seconds_per_item']:.3f}s/item "
                f"eta={details['eta']}{suffix}",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def details(self) -> dict[str, Any]:
        return progress_details(self.completed, self.total, self.start_time)


def progress_details(completed: int, total: int, start_time: float) -> dict[str, Any]:
    elapsed_sec = max(0.0, time.perf_counter() - start_time)
    completed = max(int(completed or 0), 0)
    total = max(int(total or 0), 0)
    avg = elapsed_sec / completed if completed else 0.0
    remaining = max(total - completed, 0)
    eta_sec = avg * remaining if completed else 0.0
    return {
        "completed": completed,
        "total": total,
        "elapsed": format_seconds(elapsed_sec),
        "elapsed_sec": elapsed_sec,
        "avg_seconds_per_item": avg,
        "eta": format_seconds(eta_sec) if completed else "unknown",
        "eta_sec": eta_sec,
    }


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _short_postfix_value(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


@dataclass
class LoadedQuery:
    dataset: str
    query_id: str
    question: str
    type: str = ""
    difficulty: str = ""
    video_id: str = ""
    video_path: str = ""
    video_dir: str = ""
    metadata: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        row = {
            "dataset": self.dataset,
            "query_id": self.query_id,
            "question": self.question,
            "type": self.type,
            "difficulty": self.difficulty,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "video_dir": self.video_dir,
        }
        if self.metadata:
            row["metadata"] = self.metadata
        return row


def slugify(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def parse_methods(value: str | None) -> list[str]:
    if not value:
        return list(DB_RAG_METHODS)
    aliases = {
        "evique-db": "EVIQUE",
        "evique_db": "EVIQUE",
        "evique": "EVIQUE",
        "equivocal": "VOCAL",
        "equi-vocal": "VOCAL",
        "videolisa": "VISA",
        "visa-videolisa": "VISA",
    }
    methods: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.lower().replace(" ", "")
        method = aliases.get(key) or aliases.get(slugify(token).replace("_", "-")) or token
        canonical = next((name for name in DB_RAG_METHODS if name.lower() == method.lower()), None)
        if not canonical:
            raise ValueError(f"unknown DB-RAG method: {token}")
        if canonical not in methods:
            methods.append(canonical)
    return methods


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def load_query_input(
    path: Path,
    *,
    dataset: str,
    video_path: str = "",
    video_dir: str = "",
    video_id: str = "",
    limit: int | None = None,
) -> list[LoadedQuery]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        raw_rows: list[Any] = read_jsonl(path)
    elif suffix == ".json":
        raw_rows = _extract_json_queries(read_json(path))
    else:
        raw_rows = _extract_text_queries(path)

    out: list[LoadedQuery] = []
    dataset_value = dataset or slugify(path.stem, "dataset")
    for idx, raw in enumerate(raw_rows, start=1):
        query = _normalize_raw_query(
            raw,
            idx=idx,
            dataset=dataset_value,
            default_video_path=video_path,
            default_video_dir=video_dir,
            default_video_id=video_id,
        )
        if query:
            out.append(query)
        if limit and len(out) >= limit:
            break
    return out


def load_queries_jsonl(path: Path, *, limit: int | None = None) -> list[LoadedQuery]:
    rows = read_jsonl(path)
    out: list[LoadedQuery] = []
    for idx, row in enumerate(rows, start=1):
        query = _normalize_raw_query(row, idx=idx, dataset=str(row.get("dataset") or ""), default_video_path="", default_video_dir="", default_video_id="")
        if query:
            out.append(query)
        if limit and len(out) >= limit:
            break
    return out


def _extract_json_queries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("queries"), list):
            return payload["queries"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("questions"), list):
            return payload["questions"]
        if any(key in payload for key in ("question", "query", "text", "prompt")):
            return [payload]
    raise ValueError("JSON query file must contain a query object, list, or a queries/data/questions list")


def _extract_text_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            query_id = ""
            question = text
            tab_parts = text.split("\t", 1)
            if len(tab_parts) == 2 and _looks_like_query_id(tab_parts[0]):
                query_id, question = tab_parts[0].strip(), tab_parts[1].strip()
            else:
                match = re.match(r"^\s*(?:query\s*)?([A-Za-z0-9_.-]{1,64})\s*[:：]\s+(.+)$", text, flags=re.I)
                if match and _looks_like_query_id(match.group(1)):
                    query_id, question = match.group(1).strip(), match.group(2).strip()
                else:
                    numbered = re.match(r"^\s*(\d{1,4})[\).、]\s+(.+)$", text)
                    if numbered:
                        query_id, question = f"q{int(numbered.group(1)):03d}", numbered.group(2).strip()
            rows.append({"query_id": query_id, "question": question, "line_no": line_no})
    return rows


def _looks_like_query_id(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 64:
        return False
    if " " in text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def _normalize_raw_query(
    raw: Any,
    *,
    idx: int,
    dataset: str,
    default_video_path: str,
    default_video_dir: str,
    default_video_id: str,
) -> LoadedQuery | None:
    if isinstance(raw, str):
        raw = {"question": raw}
    if not isinstance(raw, dict):
        return None
    question = str(raw.get("question") or raw.get("query") or raw.get("text") or raw.get("prompt") or "").strip()
    if not question:
        return None
    dataset_value = str(raw.get("dataset") or dataset or "").strip()
    query_id = str(raw.get("query_id") or raw.get("id") or raw.get("uid") or "").strip()
    if not query_id:
        query_id = f"{slugify(dataset_value, 'dataset')}_q{idx:03d}"
    qtype = str(raw.get("type") or raw.get("query_type") or infer_query_type(question)).strip()
    difficulty = str(raw.get("difficulty") or "").strip()
    video_path = str(raw.get("video_path") or raw.get("video") or default_video_path or "").strip()
    video_dir = str(raw.get("video_dir") or default_video_dir or "").strip()
    video_id = str(raw.get("video_id") or raw.get("source_vid") or default_video_id or "").strip()
    if not video_id:
        source = Path(video_path).stem if video_path else Path(video_dir).name if video_dir else dataset_value
        video_id = slugify(source, "video")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return LoadedQuery(
        dataset=dataset_value,
        query_id=query_id,
        question=question,
        type=qtype,
        difficulty=difficulty,
        video_id=video_id,
        video_path=video_path,
        video_dir=video_dir,
        metadata=dict(metadata),
    )


def infer_query_type(question: str) -> str:
    text = question.lower()
    tokens = set(re.findall(r"[a-z][a-z0-9]*", text))
    if tokens & {"multiple", "several", "many", "count", "counting", "crowd", "crowded"}:
        return "counting"
    if tokens & {"move", "moves", "moving", "across", "trajectory", "enter", "leave", "cross", "follow"}:
        return "motion"
    if "near center" in text or "closest to the center" in text or tokens & {"center", "centre", "middle"}:
        return "spatial"
    if tokens & {"large", "largest", "small", "smallest", "big", "biggest"}:
        return "attribute_size"
    if tokens & {"visible", "appears", "appear", "presence", "show", "shows", "find"}:
        return "existence"
    return ""


def to_db_query(query: LoadedQuery):
    from db_benchmark.schema import DBQuery

    metadata = dict(query.metadata or {})
    if query.video_path:
        metadata.setdefault("video_path", query.video_path)
    if query.video_dir:
        metadata.setdefault("video_dir", query.video_dir)
    return DBQuery(
        query_id=query.query_id,
        query=query.question,
        dataset=query.dataset,
        video_id=query.video_id,
        type=query.type,
        difficulty=query.difficulty,
        gt_windows=[],
        gt_boxes=[],
        metadata=metadata,
    )


def load_naiverag_module():
    module_name = "_db_rag_naiverag"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = PROJECT_ROOT / "RAG_Baselines" / "NaiveRAG.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import NaiveRAG from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        module = load_naiverag_module()
        return int(module.token_count(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def truncate_to_token_budget(text: str, max_tokens: int) -> tuple[str, int]:
    max_tokens = int(max_tokens or 0)
    current = estimate_tokens(text)
    if max_tokens <= 0 or current <= max_tokens:
        return text, current
    ratio = max_tokens / max(current, 1)
    limit = max(200, int(len(text) * ratio * 0.95))
    truncated = text[:limit].rstrip()
    marker = "\n\n[Evidence context truncated to fit the shared token budget.]"
    final = truncated + marker
    return final, estimate_tokens(final)


def answer_path(run_root: Path, method: str, query_id: str) -> Path:
    return Path(run_root) / f"answers-{method}" / f"answer_{safe_filename(query_id)}.md"


def evidence_json_path(run_root: Path, method: str, query_id: str) -> Path:
    return Path(run_root) / f"evidence-{method}" / f"evidence_{safe_filename(query_id)}.json"


def evidence_context_path(run_root: Path, method: str, query_id: str) -> Path:
    return Path(run_root) / f"evidence-{method}" / f"context_{safe_filename(query_id)}.md"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "item"


def rows_to_neutral_context(method: str, rows: list[dict[str, Any]], *, max_items: int | None = None) -> str:
    ok_rows = [row for row in rows if str(row.get("status") or "") == "ok"]
    selected = ok_rows[: max_items or len(ok_rows)]
    lines = [
        f"Method: {method}",
        "Evidence renderer: neutral DBEvidence temporal-window renderer.",
        "Only status=ok rows are used as evidence. Non-ok rows are omitted from the answer context.",
        "",
    ]
    if not selected:
        reasons = [str(row.get("reason") or row.get("adapter_status") or row.get("status") or "") for row in rows[:5]]
        lines.append("No usable evidence windows were retrieved.")
        if reasons:
            lines.append("Retrieval status notes: " + " | ".join(reason for reason in reasons if reason))
        return "\n".join(lines).strip()
    for row in selected:
        rank = row.get("rank")
        start = _fmt_float(row.get("start_time"))
        end = _fmt_float(row.get("end_time"))
        score = _fmt_float(row.get("score"))
        evidence_text = str(row.get("evidence_text") or "").strip()
        label = ""
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata:
            label = str(metadata.get("matched_label") or metadata.get("label") or metadata.get("source_view") or "")
        parts = [
            f"[{rank}] time={start}-{end}s",
            f"score={score}" if score else "",
            f"type={row.get('evidence_type') or ''}",
            f"track_id={row.get('track_id') or ''}",
            f"label/source={label}" if label else "",
        ]
        lines.append(" ".join(part for part in parts if part))
        if evidence_text:
            lines.append(f"support_text: {evidence_text}")
        bbox = row.get("bbox")
        if bbox:
            lines.append(f"bbox: {bbox}")
        lines.append("")
    return "\n".join(lines).strip()


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def query_to_naiverag_record(query: LoadedQuery):
    module = load_naiverag_module()
    return module.QueryRecord(
        uid=query.query_id,
        query_id=query.query_id,
        question=query.question,
        collection_id=query.dataset,
        collection_name=query.dataset,
        domain=query.dataset,
        video_id=query.video_id,
        source_vid=query.video_id,
        video_path=query.video_path,
        video_name=Path(query.video_path).name if query.video_path else query.video_id,
        original_video_path=query.video_path,
    )


def load_answer_sources(run_root: Path, methods: list[str], queries: list[LoadedQuery]) -> dict[str, dict[str, str]]:
    answer_sources: dict[str, dict[str, str]] = {}
    for method in methods:
        answers: dict[str, str] = {}
        for query in queries:
            path = answer_path(run_root, method, query.query_id)
            if path.exists():
                answers[query.query_id] = path.read_text(encoding="utf-8")
        if answers:
            answer_sources[method] = answers
    return answer_sources


def load_answer_metadata(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(run_root) / "answer_metadata.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[(str(row.get("method") or ""), str(row.get("query_id") or ""))] = row
    return out


def aggregate_quantitative_judgements(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not Path(path).exists():
        return {}
    payload = read_json(Path(path))
    accum: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, value in payload.items():
        parts = str(key).split("::")
        if len(parts) < 3 or not isinstance(value, dict):
            continue
        query_id, method = parts[0], parts[1]
        bucket = accum.setdefault((method, query_id), {metric: [] for metric in QUANT_METRICS})
        for metric in QUANT_METRICS:
            metric_payload = value.get(metric) if isinstance(value.get(metric), dict) else {}
            score = _safe_float(metric_payload.get("Score"))
            if score is not None:
                bucket[metric].append(score)
    return {
        key: {metric: _mean(values.get(metric, [])) for metric in QUANT_METRICS}
        for key, values in accum.items()
    }


def build_per_query_summary(
    *,
    run_root: Path,
    methods: list[str],
    queries: list[LoadedQuery],
    quant_scores: dict[tuple[str, str], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    answer_meta = load_answer_metadata(run_root)
    quant_scores = quant_scores or {}
    rows: list[dict[str, Any]] = []
    for query in queries:
        for method in methods:
            meta = answer_meta.get((method, query.query_id), {})
            scores = quant_scores.get((method, query.query_id), {})
            row: dict[str, Any] = {
                "dataset": query.dataset,
                "query_id": query.query_id,
                "method": method,
                "method_fidelity": meta.get("method_fidelity") or DB_RAG_METHOD_FIDELITY.get(method, ""),
                "question": query.question,
                "type": query.type,
                "difficulty": query.difficulty,
                "answer path": meta.get("answer_path") or str(answer_path(run_root, method, query.query_id)),
                "evidence chars": meta.get("evidence_chars", 0),
                "LLM input token estimate": meta.get("llm_input_token_estimate", 0),
                "retrieved evidence count": meta.get("retrieved_evidence_count", 0),
                "used evidence count": meta.get("used_evidence_count", 0),
                "avg query time": meta.get("query_time_sec", 0.0),
                "index size": meta.get("index_size_mb", 0.0),
            }
            for metric in QUANT_METRICS:
                row[metric] = scores.get(metric, "")
            rows.append(row)
    return rows


def build_comparison_summary(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_query_rows:
        grouped.setdefault(str(row.get("method") or ""), []).append(row)
    rows: list[dict[str, Any]] = []
    for method, bucket in grouped.items():
        row: dict[str, Any] = {
            "method": method,
            "method_fidelity": _first(bucket, "method_fidelity"),
            "query_count": len({str(item.get("query_id") or "") for item in bucket}),
            "answer_count": sum(1 for item in bucket if Path(str(item.get("answer path") or "")).exists()),
            "evidence chars": _mean_numeric(bucket, "evidence chars"),
            "LLM input token estimate": _mean_numeric(bucket, "LLM input token estimate"),
            "retrieved evidence count": _mean_numeric(bucket, "retrieved evidence count"),
            "used evidence count": _mean_numeric(bucket, "used evidence count"),
            "avg query time": _mean_numeric(bucket, "avg query time"),
            "index size": _mean_numeric(bucket, "index size"),
            "answer path": str(Path(_first(bucket, "answer path")).parent) if _first(bucket, "answer path") else "",
        }
        for metric in QUANT_METRICS:
            row[metric] = _mean_numeric(bucket, metric)
        rows.append(row)
    order = {method: idx for idx, method in enumerate(DB_RAG_METHODS)}
    return sorted(rows, key=lambda row: order.get(str(row.get("method")), 999))


def write_evaluation_summaries(run_root: Path, per_query_rows: list[dict[str, Any]]) -> None:
    eval_dir = Path(run_root) / "evaluation"
    comparison_rows = build_comparison_summary(per_query_rows)
    per_query_fields = [
        "dataset",
        "query_id",
        "method",
        "method_fidelity",
        "question",
        "type",
        "difficulty",
        *QUANT_METRICS,
        "evidence chars",
        "LLM input token estimate",
        "retrieved evidence count",
        "used evidence count",
        "avg query time",
        "index size",
        "answer path",
    ]
    comparison_fields = [
        "method",
        "method_fidelity",
        "query_count",
        "answer_count",
        *QUANT_METRICS,
        "evidence chars",
        "LLM input token estimate",
        "retrieved evidence count",
        "used evidence count",
        "avg query time",
        "index size",
        "answer path",
    ]
    write_csv(per_query_rows, eval_dir / "per_query_summary.csv", per_query_fields)
    write_csv(comparison_rows, eval_dir / "comparison_summary.csv", comparison_fields)


def aggregate_quant_table_from_per_query(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_query_rows:
        grouped.setdefault(str(row.get("method") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for method, bucket in grouped.items():
        row: dict[str, Any] = {"Model": method}
        for metric in QUANT_METRICS:
            row[metric] = f"{_mean_numeric(bucket, metric):.2f}"
        row["Queries"] = len({str(item.get("query_id") or "") for item in bucket})
        out.append(row)
    order = {method: idx for idx, method in enumerate(DB_RAG_METHODS)}
    return sorted(out, key=lambda row: order.get(str(row.get("Model")), 999))


def aggregate_winrate_judgements(paths: Iterable[Path]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for path in paths:
        if not Path(path).exists():
            continue
        payload = read_json(Path(path))
        for key, value in payload.items():
            parts = str(key).split("::")
            if len(parts) < 6 or parts[2] != "vs" or not isinstance(value, dict):
                continue
            model_a, model_b, order_name = parts[1], parts[3], parts[5]
            left_model, right_model = (model_b, model_a) if order_name == "rev" else (model_a, model_b)
            for metric in WINRATE_METRICS:
                winner = value.get(metric, {}).get("Winner") if isinstance(value.get(metric), dict) else None
                if winner not in {"Answer 1", "Answer 2"}:
                    continue
                winner_model = left_model if winner == "Answer 1" else right_model
                bucket = counts.setdefault((model_a, model_b, metric), {model_a: 0, model_b: 0, "total": 0})
                if winner_model in {model_a, model_b}:
                    bucket[winner_model] += 1
                    bucket["total"] += 1
    rows: list[dict[str, Any]] = []
    for (model_a, model_b, metric), bucket in sorted(counts.items()):
        total = bucket["total"]
        rows.append(
            {
                "Comparison": f"{model_a} vs {model_b}",
                "Metric": metric,
                f"{model_a} Win Rate (%)": f"{(bucket[model_a] / total * 100) if total else 0:.2f}",
                f"{model_b} Win Rate (%)": f"{(bucket[model_b] / total * 100) if total else 0:.2f}",
                f"{model_a} Wins": bucket[model_a],
                f"{model_b} Wins": bucket[model_b],
                "Judgements": total,
            }
        )
    return rows


def _safe_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    return round(_mean(numeric), 6) if numeric else 0.0


def _first(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""
