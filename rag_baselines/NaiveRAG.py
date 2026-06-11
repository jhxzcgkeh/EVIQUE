#!/usr/bin/env python3
"""
NaiveRAG baseline for VideoRAG-extracted text.

This script intentionally does not read raw videos. Its only corpus input is
VideoRAG's extracted visual descriptions plus ASR text, normally stored in:

    <workdir>/kv_store_video_segments.json

The chunking code below mirrors VideoRAG's chunking_by_video_segments protocol:
segments are tokenized with the gpt-4o tokenizer, each overlong segment is
truncated to chunk_token_size, and consecutive segments are packed into chunks
until chunk_token_size is reached.

Example:
    export OPENAI_API_KEY='your-key'
    export OPENAI_BASE_URL='${OPENAI_BASE_URL}'
    export OPENAI_MODEL='deepseek-ai/DeepSeek-V3.2'

    python Baselines/NaiveRAG.py \
        --workdir ./longervideos/videorag-workdir/4-rag-lecture \
        --questions ./longervideos/dataset.json \
        --collection 4 \
        --output-dir ./Baselines/runs/naiverag_4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is part of VideoRAG env.
    tiktoken = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency is part of VideoRAG env.
    OpenAI = None


MODEL_NAME = "NaiveRAG"
DEFAULT_CHUNK_TOKEN_SIZE = 1200
DEFAULT_MAX_CONTEXT_TOKENS = 12000
DEFAULT_FINE_NUM_FRAMES = 15
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3.2")

WINRATE_METRICS = [
    "Comprehensiveness",
    "Empowerment",
    "Trustworthiness",
    "Depth",
    "Density",
    "Overall Winner",
]

QUANT_METRICS = [
    "Comprehensiveness",
    "Empowerment",
    "Trustworthiness",
    "Depth",
    "Density",
    "Overall Score",
]

NAIVE_SYSTEM_PROMPT = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question.
Use only the retrieved VideoRAG text: visual descriptions and ASR transcripts.
If the retrieved text is insufficient, say so. Do not make up unsupported details.

---Target response length and format---

{response_type}

---Retrieved text chunks---

{content_data}
"""

WINRATE_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question based on these criteria: Comprehensiveness, Empowerment, Trustworthiness, Depth and Density.
"""

WINRATE_PROMPT = """You will evaluate two answers to the same question based on these criteria: Comprehensiveness, Empowerment, Trustworthiness, Depth and Density.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?
- Trustworthiness: Does the answer provide sufficient detail and align with common knowledge, enhancing its credibility?
- Depth: Does the answer provide in-depth analysis or details, rather than just superficial information?
- Density: Does the answer contain relevant information without less informative or redundant content?

For each criterion, choose the better answer, either Answer 1 or Answer 2, and explain why. Then select an overall winner based on these criteria.

Question:
{query}

Answer 1:
{answer1}

Answer 2:
{answer2}

Return JSON only in exactly this schema:
{{
  "Comprehensiveness": {{"Winner": "Answer 1", "Explanation": "..."}},
  "Empowerment": {{"Winner": "Answer 1", "Explanation": "..."}},
  "Trustworthiness": {{"Winner": "Answer 1", "Explanation": "..."}},
  "Depth": {{"Winner": "Answer 1", "Explanation": "..."}},
  "Density": {{"Winner": "Answer 1", "Explanation": "..."}},
  "Overall Winner": {{"Winner": "Answer 1", "Explanation": "..."}}
}}
"""

QUANT_BASELINE_SYSTEM_PROMPT = """---Role---
You are an expert evaluating an answer against a baseline answer based on these criteria: Comprehensiveness, Empowerment, Trustworthiness, Depth and Density.
"""

QUANT_BASELINE_PROMPT = """You are evaluating an answer against a baseline answer.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?
- Trustworthiness: Does the answer provide sufficient detail and align with common knowledge, enhancing its credibility?
- Depth: Does the answer provide in-depth analysis or details, rather than just superficial information?
- Density: Does the answer contain relevant information without less informative or redundant content?

For the evaluated answer, assign a score from 1 to 5 for each criterion compared with the baseline answer:
1 = strongly worse than the baseline answer
2 = weakly worse than the baseline answer
3 = comparable to the baseline answer
4 = weakly better than the baseline answer
5 = strongly better than the baseline answer

Question:
{query}

Baseline Answer:
{baseline_answer}

Evaluation Answer:
{evaluation_answer}

Return JSON only in exactly this schema:
{{
  "Comprehensiveness": {{"Score": 3, "Explanation": "..."}},
  "Empowerment": {{"Score": 3, "Explanation": "..."}},
  "Trustworthiness": {{"Score": 3, "Explanation": "..."}},
  "Depth": {{"Score": 3, "Explanation": "..."}},
  "Density": {{"Score": 3, "Explanation": "..."}},
  "Overall Score": {{"Score": 3, "Explanation": "..."}}
}}
"""

QUANT_REFERENCE_SYSTEM_PROMPT = """---Role---
You are an expert evaluating an answer against a reference answer based on these criteria: Comprehensiveness, Empowerment, Trustworthiness, Depth and Density.
"""

QUANT_REFERENCE_PROMPT = """You are evaluating an answer against a reference answer.

- Comprehensiveness: Does the answer cover all important aspects from the reference?
- Empowerment: Does the answer help the reader understand and make informed judgments?
- Trustworthiness: Is the answer faithful to the reference, sufficiently detailed, and consistent with common sense?
- Depth: Does the answer provide useful analysis instead of only surface-level facts?
- Density: Does the answer avoid redundant or irrelevant content?

Assign a score from 1 to 5 for each criterion:
1 = very poor
2 = weak
3 = acceptable
4 = good
5 = excellent

Question:
{query}

Reference Answer:
{reference_answer}

Evaluation Answer:
{evaluation_answer}

Return JSON only in exactly this schema:
{{
  "Comprehensiveness": {{"Score": 3, "Explanation": "..."}},
  "Empowerment": {{"Score": 3, "Explanation": "..."}},
  "Trustworthiness": {{"Score": 3, "Explanation": "..."}},
  "Depth": {{"Score": 3, "Explanation": "..."}},
  "Density": {{"Score": 3, "Explanation": "..."}},
  "Overall Score": {{"Score": 3, "Explanation": "..."}}
}}
"""


@dataclass
class QueryRecord:
    uid: str
    query_id: str
    question: str
    collection_id: str = ""
    collection_name: str = ""
    domain: str = ""
    video_id: str = ""
    source_vid: str = ""
    video_path: str = ""
    video_name: str = ""
    uploaded_filename: str = ""
    original_video_path: str = ""
    reference_answer: Optional[str] = None


@dataclass
class ChunkRecord:
    chunk_id: str
    content: str
    tokens: int
    chunk_order_index: int
    video_segment_id: list[str]


@dataclass
class RetrievalHit:
    item_id: str
    rank: int
    score: float
    text_score: Optional[float] = None
    visual_score: Optional[float] = None


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def slug(text: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-")
    return text or fallback


def make_openai_client() -> OpenAI:
    if OpenAI is None:
        raise SystemExit("Missing dependency: openai. Install VideoRAG requirements first.")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_encoder(model_name: str = "gpt-4o"):
    if tiktoken is None:
        raise SystemExit("Missing dependency: tiktoken. Install VideoRAG requirements first.")
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def token_count(text: str, encoder=None) -> int:
    encoder = encoder or get_encoder()
    return len(encoder.encode(text or ""))


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode()).hexdigest()


def extract_caption_and_transcript(segment: dict[str, Any], visual_field: Optional[str] = None) -> tuple[str, str]:
    """Return the visual-description channel and ASR channel from one VideoRAG segment."""
    content = str(segment.get("content") or "")
    transcript = str(segment.get("transcript") or segment.get("asr") or "")

    if visual_field and segment.get(visual_field):
        caption = str(segment[visual_field])
    else:
        caption = str(
            segment.get("fine_caption_15f")
            or segment.get("caption_15f")
            or segment.get("visual_description_15f")
            or segment.get("fine_caption")
            or segment.get("caption")
            or ""
        )

    if not caption and content:
        match = re.search(r"Caption:\s*(.*?)\s*Transcript:", content, flags=re.DOTALL | re.IGNORECASE)
        caption = match.group(1).strip() if match else content.strip()
    if not transcript and content:
        match = re.search(r"Transcript:\s*(.*)", content, flags=re.DOTALL | re.IGNORECASE)
        transcript = match.group(1).strip() if match else ""
    return caption.strip(), transcript.strip()


def segment_content(segment: dict[str, Any], visual_field: Optional[str] = None) -> str:
    """Build the unified text input: VideoRAG visual description + ASR transcript."""
    if not visual_field and segment.get("content"):
        return str(segment["content"]).strip()
    caption, transcript = extract_caption_and_transcript(segment, visual_field=visual_field)
    return f"Caption:\n{caption}\nTranscript:\n{transcript}\n\n".strip()


def load_video_segments(args: argparse.Namespace) -> dict[str, dict[str, dict[str, Any]]]:
    if args.video_segments_json:
        path = Path(args.video_segments_json)
    else:
        path = Path(args.workdir) / "kv_store_video_segments.json"
    if not path.exists():
        raise SystemExit(f"Cannot find VideoRAG segment store: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid video segment JSON: {path}")
    return data


def videorag_chunking_by_video_segments(
    video_segments: dict[str, dict[str, dict[str, Any]]],
    *,
    max_token_size: int = DEFAULT_CHUNK_TOKEN_SIZE,
    tiktoken_model_name: str = "gpt-4o",
    visual_field: Optional[str] = None,
) -> dict[str, ChunkRecord]:
    """Mirror videorag._op.get_chunks(..., chunking_by_video_segments).

    VideoRAG first tokenizes each segment's unified content, truncates an
    overlong single segment to max_token_size, then packs consecutive segments
    into a chunk until the chunk would exceed max_token_size.
    """
    encoder = get_encoder(tiktoken_model_name)
    chunks: dict[str, ChunkRecord] = {}

    for video_name in list(video_segments.keys()):
        segment_id_list = list(video_segments[video_name].keys())
        docs = [segment_content(video_segments[video_name][idx], visual_field) for idx in segment_id_list]
        doc_keys = [f"{video_name}_{idx}" for idx in segment_id_list]
        tokenized_docs = encoder.encode_batch(docs, num_threads=16)

        for index in range(len(tokenized_docs)):
            if len(tokenized_docs[index]) > max_token_size:
                tokenized_docs[index] = tokenized_docs[index][:max_token_size]

        chunk_tokens: list[int] = []
        chunk_segment_ids: list[str] = []
        chunk_order_index = 0
        for index, tokens in enumerate(tokenized_docs):
            if len(chunk_tokens) + len(tokens) <= max_token_size:
                chunk_tokens += list(tokens)
                chunk_segment_ids.append(doc_keys[index])
                continue

            decoded = encoder.decode(chunk_tokens).strip()
            if decoded:
                chunk_id = compute_mdhash_id(decoded, prefix="chunk-")
                chunks[chunk_id] = ChunkRecord(
                    chunk_id=chunk_id,
                    tokens=len(chunk_tokens),
                    content=decoded,
                    chunk_order_index=chunk_order_index,
                    video_segment_id=list(chunk_segment_ids),
                )
            chunk_tokens = list(tokens)
            chunk_segment_ids = [doc_keys[index]]
            chunk_order_index += 1

        if chunk_tokens:
            decoded = encoder.decode(chunk_tokens).strip()
            if decoded:
                chunk_id = compute_mdhash_id(decoded, prefix="chunk-")
                chunks[chunk_id] = ChunkRecord(
                    chunk_id=chunk_id,
                    tokens=len(chunk_tokens),
                    content=decoded,
                    chunk_order_index=chunk_order_index,
                    video_segment_id=list(chunk_segment_ids),
                )

    return chunks


def parse_time_range(value: Any) -> tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    parts = str(value).split("-")
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return ""
    value_int = int(value)
    h = value_int // 3600
    m = (value_int % 3600) // 60
    s = value_int % 60
    return f"{h}:{m:02d}:{s:02d}"


def build_segment_lookup(
    video_segments: dict[str, dict[str, dict[str, Any]]],
    *,
    visual_field: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for video_name, segments in video_segments.items():
        for index, segment in segments.items():
            segment_id = f"{video_name}_{index}"
            start, end = parse_time_range(segment.get("time"))
            caption, transcript = extract_caption_and_transcript(segment, visual_field=visual_field)
            frame_times = segment.get("frame_times") or []
            lookup[segment_id] = {
                "segment_id": segment_id,
                "video_name": video_name,
                "index": index,
                "start_time": format_seconds(start),
                "end_time": format_seconds(end),
                "caption": caption,
                "transcript": transcript,
                "content": segment_content(segment, visual_field=visual_field),
                "frame_count": len(frame_times) if isinstance(frame_times, list) else None,
            }
    return lookup


def load_queries(path: str, collection: Optional[str] = None) -> list[QueryRecord]:
    data = read_json(Path(path))
    records: list[QueryRecord] = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            qid = str(item.get("id", i))
            records.append(
                QueryRecord(
                    uid=qid,
                    query_id=qid,
                    question=str(item.get("question") or item.get("query") or ""),
                    reference_answer=item.get("answer") or item.get("reference_answer"),
                    domain=str(item.get("type") or item.get("domain") or ""),
                    video_id=str(item.get("video_id") or ""),
                    source_vid=str(item.get("source_vid") or ""),
                    video_path=str(item.get("video_path") or ""),
                    video_name=str(item.get("video_name") or ""),
                    uploaded_filename=str(item.get("uploaded_filename") or ""),
                    original_video_path=str(item.get("original_video_path") or ""),
                )
            )
        return [r for r in records if r.question]

    if not isinstance(data, dict):
        raise SystemExit(f"Unsupported question file format: {path}")

    selected_keys = list(data.keys())
    if collection:
        collection_id = str(collection).split("-")[0]
        if collection in data:
            selected_keys = [collection]
        elif collection_id in data:
            selected_keys = [collection_id]
        else:
            raise SystemExit(f"Collection {collection!r} was not found in {path}")

    for category_id in selected_keys:
        entries = data[category_id]
        if not entries:
            continue
        meta = entries[0]
        collection_name = str(meta.get("description") or category_id)
        domain = str(meta.get("type") or "")
        for i, item in enumerate(meta.get("questions", [])):
            qid = str(item.get("id", i))
            uid = f"{category_id}++{qid}"
            records.append(
                QueryRecord(
                    uid=uid,
                    query_id=qid,
                    question=str(item.get("question") or item.get("query") or ""),
                    collection_id=str(category_id),
                    collection_name=collection_name,
                    domain=domain,
                    reference_answer=item.get("answer") or item.get("reference_answer"),
                    video_id=str(item.get("video_id") or meta.get("video_id") or ""),
                    source_vid=str(item.get("source_vid") or meta.get("source_vid") or ""),
                    video_path=str(item.get("video_path") or meta.get("video_path") or ""),
                    video_name=str(item.get("video_name") or meta.get("video_name") or ""),
                    uploaded_filename=str(item.get("uploaded_filename") or meta.get("uploaded_filename") or ""),
                    original_video_path=str(item.get("original_video_path") or meta.get("original_video_path") or ""),
                )
            )
    return [r for r in records if r.question]


def load_reference_answers(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    data = read_json(Path(path))
    refs: dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                refs[str(key)] = str(value.get("answer") or value.get("reference_answer") or "")
            else:
                refs[str(key)] = str(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            qid = str(item.get("id", i))
            refs[qid] = str(item.get("answer") or item.get("reference_answer") or "")
    return {k: v for k, v in refs.items() if v}


def attach_reference_answers(queries: list[QueryRecord], references: dict[str, str]) -> list[QueryRecord]:
    if not references:
        return queries
    for query in queries:
        query.reference_answer = references.get(query.uid) or references.get(query.query_id) or query.reference_answer
    return queries


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    batch_size: int,
    cache_dir: Path,
    cache_name: str,
) -> np.ndarray:
    """Embed text with an on-disk cache keyed by model and text hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        json.dumps({"model": model, "texts": texts}, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{cache_name}_{digest}.npz"
    if cache_path.exists():
        return np.load(cache_path)["vectors"]

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch, encoding_format="float")
        batch_vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        vectors.append(batch_vectors)
    all_vectors = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(cache_path, vectors=all_vectors)
    return all_vectors


def rank_by_scores(ids: list[str], scores: np.ndarray, top_k: int) -> list[RetrievalHit]:
    if len(ids) == 0:
        return []
    top_k = min(top_k, len(ids))
    order = np.argsort(-scores)[:top_k]
    return [
        RetrievalHit(item_id=ids[int(i)], rank=rank + 1, score=float(scores[int(i)]))
        for rank, i in enumerate(order)
    ]


def query_embedding(client: OpenAI, query: str, model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=[query], encoding_format="float")
    return np.asarray(response.data[0].embedding, dtype=np.float32)


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        query_norm = 1.0
    normalized_query = query_vector / query_norm
    return normalized_rows(matrix) @ normalized_query


def truncate_context(parts: list[str], max_tokens: int) -> str:
    encoder = get_encoder()
    kept: list[str] = []
    total = 0
    for part in parts:
        n_tokens = len(encoder.encode(part))
        if total + n_tokens > max_tokens:
            break
        kept.append(part)
        total += n_tokens
    return "\n\n".join(kept)


def segment_refs(segment_ids: Iterable[str], lookup: dict[str, dict[str, Any]]) -> str:
    refs = []
    for sid in segment_ids:
        meta = lookup.get(sid, {})
        if meta:
            refs.append(f"{meta.get('video_name', '')}, {meta.get('start_time', '')}-{meta.get('end_time', '')}")
        else:
            refs.append(sid)
    return "; ".join(refs)


def format_chunk_context(
    chunks: dict[str, ChunkRecord],
    hits: list[RetrievalHit],
    segment_lookup: dict[str, dict[str, Any]],
    *,
    max_context_tokens: int,
) -> str:
    parts: list[str] = []
    for hit in hits:
        chunk = chunks[hit.item_id]
        refs = segment_refs(chunk.video_segment_id, segment_lookup)
        score_line = f"score={hit.score:.4f}"
        if hit.text_score is not None or hit.visual_score is not None:
            score_line = (
                f"combined={hit.score:.4f}, "
                f"text={hit.text_score if hit.text_score is not None else 0:.4f}, "
                f"visual={hit.visual_score if hit.visual_score is not None else 0:.4f}"
            )
        parts.append(
            f"-----New Chunk {hit.rank}-----\n"
            f"chunk_id: {hit.item_id}\n"
            f"segments: {refs}\n"
            f"{score_line}\n\n"
            f"{chunk.content}"
        )
    return truncate_context(parts, max_context_tokens)


def call_chat(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    current_messages = list(messages)
    for _ in range(max_retries):
        raw = call_chat(client, model=model, messages=current_messages, temperature=temperature)
        try:
            return extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001 - keep judge calls robust.
            last_error = exc
            current_messages = current_messages + [
                {
                    "role": "user",
                    "content": "The previous response was not valid JSON. Return only one valid JSON object that follows the requested schema.",
                }
            ]
    raise RuntimeError(f"Could not parse JSON judge response: {last_error}") from last_error


def answer_path(root: Path, query: QueryRecord, *, flat: bool) -> Path:
    if flat:
        return root / "answers" / f"answer_{query.query_id}.md"
    collection = slug(f"{query.collection_id}-{query.collection_name}", query.collection_id or "collection")
    return root / "answers" / collection / f"answer_{query.query_id}.md"


def result_path(root: Path, query: QueryRecord, *, flat: bool) -> Path:
    if flat:
        return root / "per_query" / f"query_{query.query_id}.json"
    collection = slug(f"{query.collection_id}-{query.collection_name}", query.collection_id or "collection")
    return root / "per_query" / collection / f"query_{query.query_id}.json"


def candidate_answer_paths(base_dir: Path, query: QueryRecord) -> list[Path]:
    collection = slug(f"{query.collection_id}-{query.collection_name}", query.collection_id or "collection")
    return [
        base_dir / f"answer_{query.query_id}.md",
        base_dir / f"{query.query_id}.md",
        base_dir / f"answer_{query.uid}.md",
        base_dir / f"{query.uid}.md",
        base_dir / collection / f"answer_{query.query_id}.md",
        base_dir / collection / f"{query.query_id}.md",
        base_dir / "answers" / f"answer_{query.query_id}.md",
        base_dir / "answers" / collection / f"answer_{query.query_id}.md",
    ]


def read_answer_from_dir(base_dir: Path, query: QueryRecord) -> Optional[str]:
    for path in candidate_answer_paths(base_dir, query):
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def collect_answer_sources(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    generated_answers: dict[str, str],
    *,
    current_model_name: str,
) -> dict[str, dict[str, str]]:
    source_dirs = {
        "EVIQUE": getattr(args, "evique_answers", None),
        "VideoRAG": args.videorag_answers,
        "NaiveRAG": args.naiverag_answers,
        "TextVideoRAG": args.textvideorag_answers,
    }
    sources: dict[str, dict[str, str]] = {}
    for model_name, dir_value in source_dirs.items():
        answers: dict[str, str] = {}
        if dir_value:
            base_dir = Path(dir_value)
            for query in queries:
                answer = read_answer_from_dir(base_dir, query)
                if answer is not None:
                    answers[query.uid] = answer
        if answers:
            sources[model_name] = answers

    if generated_answers:
        sources[current_model_name] = generated_answers
    return sources


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_default_pairs(available_models: Iterable[str], anchor_winrate_model: str | None = None) -> list[tuple[str, str]]:
    available = list(dict.fromkeys(available_models))
    available_set = set(available)
    preferred = [
        ("EVIQUE", "VideoRAG"),
        ("EVIQUE", "NaiveRAG"),
        ("EVIQUE", "TextVideoRAG"),
        ("VideoRAG", "NaiveRAG"),
        ("VideoRAG", "TextVideoRAG"),
    ]
    anchor_winrate_model = anchor_winrate_model.strip() if anchor_winrate_model else None
    if anchor_winrate_model:
        if anchor_winrate_model not in available_set:
            available_text = ", ".join(available) if available else "(none)"
            raise ValueError(f"anchor model not found: {anchor_winrate_model!r}. Available models: {available_text}")
        return [(anchor_winrate_model, model) for model in available if model != anchor_winrate_model]
    return [(a, b) for a, b in preferred if a in available_set and b in available_set]


def validate_winrate_result(data: dict[str, Any]) -> dict[str, Any]:
    for metric in WINRATE_METRICS:
        if metric not in data:
            raise ValueError(f"Missing metric {metric}")
        winner = data[metric].get("Winner")
        if winner not in {"Answer 1", "Answer 2"}:
            raise ValueError(f"Invalid winner for {metric}: {winner}")
    return data


def validate_quant_result(data: dict[str, Any]) -> dict[str, Any]:
    for metric in QUANT_METRICS:
        if metric not in data:
            raise ValueError(f"Missing metric {metric}")
        score = int(data[metric].get("Score"))
        if score < 1 or score > 5:
            raise ValueError(f"Invalid score for {metric}: {score}")
        data[metric]["Score"] = score
    return data


def run_winrate_eval(
    client: OpenAI,
    *,
    llm_model: str,
    queries: list[QueryRecord],
    answer_sources: dict[str, dict[str, str]],
    output_dir: Path,
    bidirectional: bool,
    eval_runs: int,
    anchor_winrate_model: str | None = None,
) -> list[dict[str, Any]]:
    pairs = choose_default_pairs(answer_sources.keys(), anchor_winrate_model=anchor_winrate_model)
    all_judgements: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    eval_runs = max(int(eval_runs), 1)

    for model_a, model_b in pairs:
        counts = {metric: {model_a: 0, model_b: 0} for metric in WINRATE_METRICS}
        total = {metric: 0 for metric in WINRATE_METRICS}

        for query in queries:
            answer_a = answer_sources.get(model_a, {}).get(query.uid)
            answer_b = answer_sources.get(model_b, {}).get(query.uid)
            if answer_a is None or answer_b is None:
                continue

            orders = [(model_a, answer_a, model_b, answer_b, "ori")]
            if bidirectional:
                orders.append((model_b, answer_b, model_a, answer_a, "rev"))

            for run_idx in range(eval_runs):
                for left_model, left_answer, right_model, right_answer, order_name in orders:
                    prompt = WINRATE_PROMPT.format(query=query.question, answer1=left_answer, answer2=right_answer)
                    data = chat_json(
                        client,
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": WINRATE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    data = validate_winrate_result(data)
                    judgement_key = f"{query.uid}::{model_a}::vs::{model_b}::run{run_idx + 1}::{order_name}"
                    all_judgements[judgement_key] = data

                    for metric in WINRATE_METRICS:
                        winner_label = data[metric]["Winner"]
                        winner_model = left_model if winner_label == "Answer 1" else right_model
                        if winner_model in counts[metric]:
                            counts[metric][winner_model] += 1
                            total[metric] += 1

        for metric in WINRATE_METRICS:
            metric_total = total[metric]
            a_wins = counts[metric][model_a]
            b_wins = counts[metric][model_b]
            table_rows.append(
                {
                    "Comparison": f"{model_a} vs {model_b}",
                    "Metric": metric,
                    f"{model_a} Win Rate (%)": f"{(a_wins / metric_total * 100) if metric_total else 0:.2f}",
                    f"{model_b} Win Rate (%)": f"{(b_wins / metric_total * 100) if metric_total else 0:.2f}",
                    f"{model_a} Wins": a_wins,
                    f"{model_b} Wins": b_wins,
                    "Judgements": metric_total,
                }
            )

    eval_dir = output_dir / "evaluation"
    write_json(all_judgements, eval_dir / "winrate_judgements.json")
    write_text(markdown_table(table_rows), eval_dir / "winrate_table.md")
    save_csv(table_rows, eval_dir / "winrate_table.csv")
    return table_rows


def run_quantitative_eval(
    client: OpenAI,
    *,
    llm_model: str,
    queries: list[QueryRecord],
    answer_sources: dict[str, dict[str, str]],
    output_dir: Path,
    baseline_model: str,
    eval_runs: int,
) -> list[dict[str, Any]]:
    use_reference = any(query.reference_answer for query in queries)
    all_scores: dict[str, Any] = {}
    accum: dict[str, dict[str, list[int]]] = {
        model_name: {metric: [] for metric in QUANT_METRICS} for model_name in answer_sources
    }
    eval_runs = max(int(eval_runs), 1)

    for model_name, answers in answer_sources.items():
        for query in queries:
            answer = answers.get(query.uid)
            if answer is None:
                continue

            for run_idx in range(eval_runs):
                if use_reference and query.reference_answer:
                    prompt = QUANT_REFERENCE_PROMPT.format(
                        query=query.question,
                        reference_answer=query.reference_answer,
                        evaluation_answer=answer,
                    )
                    system_prompt = QUANT_REFERENCE_SYSTEM_PROMPT
                else:
                    baseline_answer = answer_sources.get(baseline_model, {}).get(query.uid)
                    if baseline_answer is None:
                        continue
                    if model_name == baseline_model:
                        data = {
                            metric: {
                                "Score": 3,
                                "Explanation": "Baseline answer; score fixed at 3 by definition.",
                            }
                            for metric in QUANT_METRICS
                        }
                        all_scores[f"{query.uid}::{model_name}::run{run_idx + 1}"] = data
                        for metric in QUANT_METRICS:
                            accum[model_name][metric].append(3)
                        continue

                    prompt = QUANT_BASELINE_PROMPT.format(
                        query=query.question,
                        baseline_answer=baseline_answer,
                        evaluation_answer=answer,
                    )
                    system_prompt = QUANT_BASELINE_SYSTEM_PROMPT

                data = chat_json(
                    client,
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                data = validate_quant_result(data)
                all_scores[f"{query.uid}::{model_name}::run{run_idx + 1}"] = data
                for metric in QUANT_METRICS:
                    accum[model_name][metric].append(int(data[metric]["Score"]))

    rows: list[dict[str, Any]] = []
    preferred_order = ["EVIQUE", "VideoRAG", "NaiveRAG", "TextVideoRAG"]
    for model_name in preferred_order + sorted(name for name in accum if name not in preferred_order):
        if model_name not in accum:
            continue
        row: dict[str, Any] = {"Model": model_name}
        for metric in QUANT_METRICS:
            scores = accum[model_name][metric]
            row[metric] = f"{(sum(scores) / len(scores)) if scores else 0:.2f}"
        row["Queries"] = max((len(accum[model_name][m]) for m in QUANT_METRICS), default=0)
        rows.append(row)

    eval_dir = output_dir / "evaluation"
    write_json(all_scores, eval_dir / "quantitative_judgements.json")
    write_text(markdown_table(rows), eval_dir / "quantitative_table.md")
    save_csv(rows, eval_dir / "quantitative_table.csv")
    return rows


class NaiveRAGPipeline:
    def __init__(self, args: argparse.Namespace, client: OpenAI):
        self.args = args
        self.client = client
        self.output_dir = Path(args.output_dir)
        self.video_segments = load_video_segments(args)
        self.segment_lookup = build_segment_lookup(self.video_segments, visual_field=args.visual_field)
        self.chunks = videorag_chunking_by_video_segments(
            self.video_segments,
            max_token_size=args.chunk_token_size,
            visual_field=args.visual_field,
        )
        self.chunk_ids = list(self.chunks.keys())
        self.chunk_texts = [self.chunks[cid].content for cid in self.chunk_ids]
        self.chunk_embeddings = embed_texts(
            client,
            self.chunk_texts,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            cache_dir=self.output_dir / "cache",
            cache_name="naiverag_chunks",
        )

    def retrieve(self, query: str) -> list[RetrievalHit]:
        qvec = query_embedding(self.client, query, self.args.embedding_model)
        scores = cosine_scores(qvec, self.chunk_embeddings)
        return rank_by_scores(self.chunk_ids, scores, self.args.top_k)

    def answer_query(self, query: QueryRecord) -> dict[str, Any]:
        hits = self.retrieve(query.question)
        context = format_chunk_context(
            self.chunks,
            hits,
            self.segment_lookup,
            max_context_tokens=self.args.max_context_tokens,
        )
        system_prompt = NAIVE_SYSTEM_PROMPT.format(
            response_type=self.args.response_type,
            content_data=context or "No retrieved content.",
        )
        retrieved_count = len(hits)
        used_count = context.count("-----New Chunk")
        answer = call_chat(
            self.client,
            model=self.args.model,
            temperature=self.args.temperature,
            max_tokens=self.args.max_answer_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.question},
            ],
        )
        return {
            "model": MODEL_NAME,
            "query": asdict(query),
            "answer": answer,
            "retrieval": [asdict(hit) for hit in hits],
            "context": context,
            "metrics": {
                "retrieved_count": retrieved_count,
                "used_count": used_count,
                "support_ratio": (used_count / retrieved_count) if retrieved_count else 0.0,
                "evidence_chars": len(context),
                "llm_input_tokens_estimate": token_count(system_prompt) + token_count(query.question),
            },
        }


def add_common_arguments(parser: argparse.ArgumentParser, *, model_name: str) -> argparse.ArgumentParser:
    default_output = f"./Baselines/runs/{model_name.lower()}_{now_stamp()}"
    parser.add_argument("--workdir", default="./videorag-workdir", help="VideoRAG workdir containing kv_store_video_segments.json.")
    parser.add_argument("--video-segments-json", default=None, help="Direct path to kv_store_video_segments.json.")
    parser.add_argument("--questions", default="./longervideos/dataset.json", help="Question JSON. Supports LongerVideos dataset.json or a list of {id, question}.")
    parser.add_argument("--collection", default=None, help="Optional LongerVideos collection id, e.g. 4 or 4-rag-lecture.")
    parser.add_argument("--reference-answers", default=None, help="Optional JSON with standard/reference answers for quantitative scoring.")
    parser.add_argument("--visual-field", default=None, help="Optional segment field containing the unified 15-frame visual description.")
    parser.add_argument("--fine-num-frames", type=int, default=DEFAULT_FINE_NUM_FRAMES, help="Metadata guardrail: VideoRAG fine descriptions should use 15 sampled frames.")
    parser.add_argument("--chunk-token-size", type=int, default=DEFAULT_CHUNK_TOKEN_SIZE, help="VideoRAG chunk size. Default matches VideoRAG.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of retrieved chunks.")
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS, help="Token budget for retrieved context passed to the LLM.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model served by OPENAI_BASE_URL.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Answer/evaluator LLM model. Defaults to OPENAI_MODEL.")
    parser.add_argument("--judge-model", default=None, help="LLM judge model for paper-style evaluation. Defaults to --model.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of queries for a smoke run.")
    parser.add_argument("--skip-generation", action="store_true", help="Only run evaluation from answer folders.")
    parser.add_argument("--run-eval", action="store_true", help="Run win-rate and quantitative comparison after generation.")
    parser.add_argument("--single-pass-winrate", action="store_true", help="Use one answer order per query. Default uses original+reverse order like the paper reproduction code.")
    parser.add_argument("--eval-runs", type=int, default=5, help="Number of LLM-judge repetitions. Default 5 matches the paper reproduction protocol.")
    parser.add_argument("--anchor-winrate-model", default=None, help="Optional anchor model for win-rate evaluation; only compare this model against each other available model.")
    parser.add_argument("--videorag-answers", default=None, help="Answer directory for VideoRAG.")
    parser.add_argument("--evique-answers", default=None, help="Answer directory for EVIQUE.")
    parser.add_argument("--naiverag-answers", default=None, help="Answer directory for NaiveRAG.")
    parser.add_argument("--textvideorag-answers", default=None, help="Answer directory for TextVideoRAG.")
    parser.add_argument("--quant-baseline", default="NaiveRAG", choices=["EVIQUE", "VideoRAG", "NaiveRAG", "TextVideoRAG"], help="Baseline model for 1-5 quantitative comparison when no reference answers are supplied.")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NaiveRAG baseline on VideoRAG-extracted text.")
    return add_common_arguments(parser, model_name=MODEL_NAME)


def run_generation(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    pipeline_factory: Callable[[argparse.Namespace, OpenAI], Any],
    model_name: str,
) -> dict[str, str]:
    output_dir = Path(args.output_dir)
    flat = len({q.collection_id for q in queries}) <= 1
    client = make_openai_client()
    index_start = time.perf_counter()
    pipeline = pipeline_factory(args, client)
    index_build_seconds = time.perf_counter() - index_start

    write_json(
        {
            "model": model_name,
            "chunk_token_size": args.chunk_token_size,
            "fine_num_frames": args.fine_num_frames,
            "num_chunks": len(pipeline.chunks),
            "num_segments": len(pipeline.segment_lookup),
            "index_build_time_seconds": index_build_seconds,
            "note": "Corpus input is VideoRAG-extracted visual descriptions plus ASR text; raw videos are not read.",
        },
        output_dir / "run_config.json",
    )

    generated: dict[str, str] = {}
    all_results: dict[str, Any] = {}
    query_times: list[float] = []
    for query in queries:
        print(f"[{model_name}] query {query.uid}: {query.question}")
        query_start = time.perf_counter()
        result = pipeline.answer_query(query)
        query_seconds = time.perf_counter() - query_start
        query_times.append(query_seconds)
        result.setdefault("metrics", {})["query_time_seconds"] = query_seconds
        generated[query.uid] = result["answer"]
        all_results[query.uid] = result
        write_text(result["answer"], answer_path(output_dir, query, flat=flat))
        write_json(result, result_path(output_dir, query, flat=flat))

    write_json(all_results, output_dir / "all_query_results.json")
    result_metrics = [result.get("metrics", {}) for result in all_results.values()]
    write_json(
        {
            "model": model_name,
            "index_build_time_seconds": index_build_seconds,
            "index_size_mb": directory_size_bytes(output_dir / "cache") / (1024 * 1024),
            "avg_query_time_seconds": (sum(query_times) / len(query_times)) if query_times else 0.0,
            "avg_evidence_chars": (
                sum(float(m.get("evidence_chars", 0)) for m in result_metrics) / len(result_metrics)
            )
            if result_metrics
            else 0.0,
            "avg_llm_input_tokens_estimate": (
                sum(float(m.get("llm_input_tokens_estimate", 0)) for m in result_metrics) / len(result_metrics)
            )
            if result_metrics
            else 0.0,
            "avg_retrieved_count": (
                sum(float(m.get("retrieved_count", 0)) for m in result_metrics) / len(result_metrics)
            )
            if result_metrics
            else 0.0,
            "avg_used_count": (
                sum(float(m.get("used_count", 0)) for m in result_metrics) / len(result_metrics)
            )
            if result_metrics
            else 0.0,
            "avg_support_ratio": (
                sum(float(m.get("support_ratio", 0)) for m in result_metrics) / len(result_metrics)
            )
            if result_metrics
            else 0.0,
        },
        output_dir / "generation_metrics.json",
    )
    return generated


def run_evaluation_if_requested(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    generated_answers: dict[str, str],
    *,
    current_model_name: str,
) -> None:
    if not args.run_eval:
        return
    client = make_openai_client()
    sources = collect_answer_sources(args, queries, generated_answers, current_model_name=current_model_name)
    if len(sources) < 2:
        print("[eval] Need at least two answer sources. Skipping evaluation.")
        return
    print(f"[eval] available answer sources: {', '.join(sorted(sources))}")
    win_rows = run_winrate_eval(
        client,
        llm_model=args.judge_model or args.model,
        queries=queries,
        answer_sources=sources,
        output_dir=Path(args.output_dir),
        bidirectional=not args.single_pass_winrate,
        eval_runs=args.eval_runs,
        anchor_winrate_model=args.anchor_winrate_model,
    )
    quant_rows = run_quantitative_eval(
        client,
        llm_model=args.judge_model or args.model,
        queries=queries,
        answer_sources=sources,
        output_dir=Path(args.output_dir),
        baseline_model=args.quant_baseline,
        eval_runs=args.eval_runs,
    )
    print("\nWin-Rate Comparison")
    print(markdown_table(win_rows))
    print("\nQuantitative Comparison")
    print(markdown_table(quant_rows))


def main() -> None:
    args = build_arg_parser().parse_args()
    queries = load_queries(args.questions, args.collection)
    queries = attach_reference_answers(queries, load_reference_answers(args.reference_answers))
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit("No queries loaded.")

    generated: dict[str, str] = {}
    if not args.skip_generation:
        generated = run_generation(args, queries, pipeline_factory=NaiveRAGPipeline, model_name=MODEL_NAME)
    run_evaluation_if_requested(args, queries, generated, current_model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
