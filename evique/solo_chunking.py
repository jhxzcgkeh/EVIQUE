"""Solo text chunking for EVIQUE standalone base generation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenEncoder:
    name: str
    encoder: Any | None = None

    def encode(self, text: str) -> list[int]:
        if self.encoder is not None:
            return list(self.encoder.encode(text or ""))
        return list(range(len(_fallback_tokens(text))))

    def decode(self, tokens: list[int], original_tokens: list[str] | None = None) -> str:
        if self.encoder is not None:
            return str(self.encoder.decode(tokens))
        if original_tokens is None:
            return ""
        return " ".join(original_tokens[: len(tokens)])


def compute_chunk_id(content: str) -> str:
    return "chunk-" + hashlib.md5(content.encode("utf-8")).hexdigest()


def _fallback_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\S+", text or "")
    if tokens:
        return tokens
    return list(text or "")


def get_token_encoder(model_name: str = "gpt-4o") -> TokenEncoder:
    try:
        import tiktoken

        return TokenEncoder(name="tiktoken", encoder=tiktoken.encoding_for_model(model_name))
    except Exception:
        return TokenEncoder(name="fallback", encoder=None)


def _segment_text(segment: dict[str, Any]) -> str:
    caption = str(segment.get("caption") or "")
    transcript = str(segment.get("transcript") or "")
    return f"Caption:\n{caption}\nTranscript:\n{transcript}\n\n".strip()


def _chunk_metadata(
    *,
    video_name: str,
    segment_ids: list[str],
    segments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first = segments[segment_ids[0]] if segment_ids else {}
    last = segments[segment_ids[-1]] if segment_ids else first
    metadata = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    return {
        "video_id": first.get("video_id") or video_name,
        "source_vid": first.get("source_vid") or video_name,
        "segment_id": list(segment_ids),
        "start_time": first.get("start_time"),
        "end_time": last.get("end_time"),
        "video_path": metadata.get("video_path") or first.get("video_path"),
        "source": "evique_solo_standalone",
    }


def build_text_chunks(
    video_segments: dict[str, dict[str, dict[str, Any]]],
    *,
    max_token_size: int = 1200,
    tiktoken_model_name: str = "gpt-4o",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if max_token_size <= 0:
        raise ValueError("max_token_size must be positive")
    encoder = get_token_encoder(tiktoken_model_name)
    chunks: dict[str, dict[str, Any]] = {}
    chunk_order_index = 0
    for video_name, segments in video_segments.items():
        segment_ids = list(segments.keys())
        current_tokens: list[int] = []
        current_words: list[str] = []
        current_segment_ids: list[str] = []
        current_parts: list[str] = []

        def flush() -> None:
            nonlocal chunk_order_index, current_tokens, current_words, current_segment_ids, current_parts
            if not current_segment_ids:
                return
            content = "\n".join(part for part in current_parts if part).strip()
            if not content:
                current_tokens = []
                current_words = []
                current_segment_ids = []
                current_parts = []
                return
            chunk_id = compute_chunk_id(content)
            chunks[chunk_id] = {
                "tokens": len(current_tokens),
                "content": content,
                "chunk_order_index": chunk_order_index,
                "video_segment_id": [f"{video_name}_{segment_id}" for segment_id in current_segment_ids],
                "metadata": _chunk_metadata(
                    video_name=video_name,
                    segment_ids=current_segment_ids,
                    segments=segments,
                ),
            }
            chunk_order_index += 1
            current_tokens = []
            current_words = []
            current_segment_ids = []
            current_parts = []

        for segment_id in segment_ids:
            text = _segment_text(segments[segment_id])
            tokens = encoder.encode(text)
            words = _fallback_tokens(text)
            if len(tokens) > max_token_size:
                tokens = tokens[:max_token_size]
                if encoder.encoder is not None:
                    text = encoder.decode(tokens).strip()
                else:
                    words = words[:max_token_size]
                    text = " ".join(words).strip()
            if current_segment_ids and len(current_tokens) + len(tokens) > max_token_size:
                flush()
            current_tokens += list(tokens)
            current_words += list(words)
            current_segment_ids.append(segment_id)
            current_parts.append(text)
        flush()
    stats = {
        "chunk_count": len(chunks),
        "tokenizer": encoder.name,
        "chunk_token_size": max_token_size,
    }
    return chunks, stats
