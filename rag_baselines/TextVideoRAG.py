#!/usr/bin/env python3
"""
TextVideoRAG baseline for VideoRAG-extracted text.

This baseline is text-first: it retrieves VideoRAG chunks by text embedding
similarity, then lightly reranks those chunks with a separate visual-description
channel built from the extracted captions. It never reads raw videos.

The intended input is the same VideoRAG segment store used by NaiveRAG:

    <workdir>/kv_store_video_segments.json

Example:
    export OPENAI_API_KEY='your-key'
    export OPENAI_BASE_URL='${OPENAI_BASE_URL}'
    export OPENAI_MODEL='deepseek-ai/DeepSeek-V3.2'

    python Baselines/TextVideoRAG.py \
        --workdir ./longervideos/videorag-workdir/4-rag-lecture \
        --questions ./longervideos/dataset.json \
        --collection 4 \
        --output-dir ./Baselines/runs/textvideorag_4
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from NaiveRAG import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    RetrievalHit,
    add_common_arguments,
    attach_reference_answers,
    build_segment_lookup,
    call_chat,
    cosine_scores,
    embed_texts,
    format_chunk_context,
    load_queries,
    load_reference_answers,
    load_video_segments,
    query_embedding,
    rank_by_scores,
    run_evaluation_if_requested,
    run_generation,
    segment_refs,
    truncate_context,
    token_count,
    videorag_chunking_by_video_segments,
)


MODEL_NAME = "TextVideoRAG"

TEXT_VIDEO_SYSTEM_PROMPT = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question.
Use only the retrieved VideoRAG inputs: ASR transcripts and visual descriptions.
Text chunks are the primary evidence. The visual-description channel is auxiliary evidence for scene details.
If the retrieved text is insufficient, say so. Do not make up unsupported details.

---Target response length and format---

{response_type}

---Retrieved visual-description channel---

{visual_data}

---Retrieved text chunks---

{chunk_data}
"""


def visual_channel_text(meta: dict[str, Any]) -> str:
    caption = str(meta.get("caption") or "").strip()
    transcript = str(meta.get("transcript") or "").strip()
    if caption:
        return f"Visual description:\n{caption}"
    return f"Visual description:\n{meta.get('content', '')}\nTranscript:\n{transcript}".strip()


def format_visual_context(
    visual_hits: list[RetrievalHit],
    segment_lookup: dict[str, dict[str, Any]],
    *,
    max_context_tokens: int,
) -> str:
    parts: list[str] = []
    for hit in visual_hits:
        meta = segment_lookup[hit.item_id]
        refs = segment_refs([hit.item_id], segment_lookup)
        parts.append(
            f"-----Retrieved Visual Description {hit.rank}-----\n"
            f"segment_id: {hit.item_id}\n"
            f"segment: {refs}\n"
            f"visual_score={hit.score:.4f}\n\n"
            f"Caption:\n{meta.get('caption', '')}\n"
            f"Transcript:\n{meta.get('transcript', '')}"
        )
    return truncate_context(parts, max_context_tokens)


class TextVideoRAGPipeline:
    def __init__(self, args: argparse.Namespace, client):
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
            cache_name="textvideorag_chunks",
        )

        self.segment_ids = list(self.segment_lookup.keys())
        self.visual_texts = [visual_channel_text(self.segment_lookup[sid]) for sid in self.segment_ids]
        self.visual_embeddings = embed_texts(
            client,
            self.visual_texts,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            cache_dir=self.output_dir / "cache",
            cache_name="textvideorag_visual_segments",
        )

    def retrieve(self, query: str) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
        qvec = query_embedding(self.client, query, self.args.embedding_model)
        text_scores = cosine_scores(qvec, self.chunk_embeddings)
        visual_scores = cosine_scores(qvec, self.visual_embeddings)

        visual_hits = rank_by_scores(self.segment_ids, visual_scores, self.args.visual_top_k)
        visual_by_segment = {sid: float(score) for sid, score in zip(self.segment_ids, visual_scores)}

        chunk_visual_scores: list[float] = []
        for cid in self.chunk_ids:
            segment_ids = self.chunks[cid].video_segment_id
            if not segment_ids:
                chunk_visual_scores.append(0.0)
            else:
                chunk_visual_scores.append(max(visual_by_segment.get(sid, 0.0) for sid in segment_ids))

        text_weight = max(float(self.args.text_weight), 0.0)
        visual_weight = max(float(self.args.visual_weight), 0.0)
        weight_sum = text_weight + visual_weight
        if weight_sum <= 0:
            text_weight, visual_weight, weight_sum = 1.0, 0.0, 1.0
        text_weight /= weight_sum
        visual_weight /= weight_sum

        chunk_visual_scores_array = np.asarray(chunk_visual_scores, dtype=np.float32)
        combined = text_weight * text_scores + visual_weight * chunk_visual_scores_array
        order = np.argsort(-combined)[: min(self.args.top_k, len(self.chunk_ids))]
        chunk_hits: list[RetrievalHit] = []
        for rank, idx in enumerate(order, start=1):
            chunk_hits.append(
                RetrievalHit(
                    item_id=self.chunk_ids[int(idx)],
                    rank=rank,
                    score=float(combined[int(idx)]),
                    text_score=float(text_scores[int(idx)]),
                    visual_score=float(chunk_visual_scores_array[int(idx)]),
                )
            )
        return chunk_hits, visual_hits

    def answer_query(self, query) -> dict[str, Any]:
        chunk_hits, visual_hits = self.retrieve(query.question)
        chunk_budget = int(self.args.max_context_tokens * self.args.chunk_context_ratio)
        visual_budget = max(self.args.max_context_tokens - chunk_budget, 0)
        chunk_context = format_chunk_context(
            self.chunks,
            chunk_hits,
            self.segment_lookup,
            max_context_tokens=chunk_budget,
        )
        visual_context = format_visual_context(
            visual_hits,
            self.segment_lookup,
            max_context_tokens=visual_budget,
        )
        system_prompt = TEXT_VIDEO_SYSTEM_PROMPT.format(
            response_type=self.args.response_type,
            visual_data=visual_context or "No auxiliary visual descriptions retrieved.",
            chunk_data=chunk_context or "No retrieved text chunks.",
        )
        retrieved_count = len(chunk_hits) + len(visual_hits)
        used_count = chunk_context.count("-----New Chunk") + visual_context.count("-----Retrieved Visual Description")
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
            "retrieval": {
                "text_first_reranked_chunks": [asdict(hit) for hit in chunk_hits],
                "auxiliary_visual_segments": [asdict(hit) for hit in visual_hits],
                "text_weight": self.args.text_weight,
                "visual_weight": self.args.visual_weight,
            },
            "context": {
                "visual_data": visual_context,
                "chunk_data": chunk_context,
            },
            "metrics": {
                "retrieved_count": retrieved_count,
                "used_count": used_count,
                "support_ratio": (used_count / retrieved_count) if retrieved_count else 0.0,
                "evidence_chars": len(visual_context) + len(chunk_context),
                "llm_input_tokens_estimate": token_count(system_prompt) + token_count(query.question),
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TextVideoRAG baseline on VideoRAG-extracted text.")
    add_common_arguments(parser, model_name=MODEL_NAME)
    parser.set_defaults(max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--visual-top-k", type=int, default=8, help="Number of auxiliary visual-description segments.")
    parser.add_argument("--text-weight", type=float, default=0.75, help="Primary text similarity weight for reranking.")
    parser.add_argument("--visual-weight", type=float, default=0.25, help="Auxiliary visual-description similarity weight for reranking.")
    parser.add_argument("--chunk-context-ratio", type=float, default=0.75, help="Share of context tokens reserved for text chunks.")
    return parser


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
        generated = run_generation(args, queries, pipeline_factory=TextVideoRAGPipeline, model_name=MODEL_NAME)
    run_evaluation_if_requested(args, queries, generated, current_model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
