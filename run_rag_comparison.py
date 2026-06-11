#!/usr/bin/env python3
"""
Run the paper-style VideoRAG vs. RAG-baseline comparison on a custom video set.

The runner keeps the comparison protocol aligned with the VideoRAG paper:
1. Shared mode uses VideoRAG's grounded visual-caption + ASR transcript store.
2. Standalone EVIQUE can build an EVIQUE-owned compatible base store.
3. All RAG methods use the same VideoRAG segment-packing chunk protocol.
4. Video captions are generated with 15 sampled frames by default.
5. Evaluation uses LLM-as-judge win-rate and 1-5 quantitative scoring, repeated
   5 times by default, with NaiveRAG as the quantitative baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from string import Template
from typing import Any

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001 - progress bars are optional.
    _tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
RAG_BASELINES_DIR = PROJECT_ROOT / "rag_baselines"

EVIQUE_PACKAGE_DIR = PROJECT_ROOT / "evique"
VIDEORAG_ROOT = RAG_BASELINES_DIR / "VideoRAG-algorithm"
VIDEORAG_PACKAGE_DIR = VIDEORAG_ROOT / "videorag"
NAIVERAG_FILE = RAG_BASELINES_DIR / "NaiveRAG.py"
TEXTVIDEORAG_FILE = RAG_BASELINES_DIR / "TextVideoRAG.py"
LIGHTRAG_ROOT = RAG_BASELINES_DIR / "LightRAG"
LIGHTRAG_PACKAGE_DIR = LIGHTRAG_ROOT / "lightrag"
GRAPHRAG_ROOT = RAG_BASELINES_DIR / "GraphRAG"
GRAPHRAG_PACKAGE_ROOT = GRAPHRAG_ROOT / "packages" / "graphrag"
DEFAULT_LIGHTRAG_TIKTOKEN_MODEL_NAME = "gpt-4o-mini"

DATASET_DIR = PROJECT_ROOT / "Dataset"
DEFAULT_YOLO_MODEL = Path(os.getenv("EVIQUE_DETECTOR_MODEL", str(PROJECT_ROOT / "models" / "yolo11n.pt")))
MODELS_ROOT = Path(os.getenv("EVIQUE_MODELS_ROOT", str(PROJECT_ROOT / "models")))
DEFAULT_CAPTION_MODEL_PATH = MODELS_ROOT / "MiniCPM-V-2_6-int4"
DEFAULT_WHISPER_SMALL = MODELS_ROOT / "faster-whisper-small"
DEFAULT_WHISPER_TINY = MODELS_ROOT / "faster-whisper-tiny"
CAPTION_MODEL_PATH_ENV_VALUE = os.getenv("CAPTION_MODEL_PATH")
WHISPER_MODEL_ENV_VALUE = os.getenv("WHISPER_MODEL")

REPO_ROOT = PROJECT_ROOT
BASELINES_DIR = RAG_BASELINES_DIR
UNIFIED_BASE_SOURCE_EVIQUE = "evique"
VIDEORAG_COMPAT_BASE_DERIVATION_FILE = "unified_base_source.json"


def _prepend_sys_path(path: Path) -> None:
    path_text = str(path)
    sys.path[:] = [item for item in sys.path if item != path_text]
    sys.path.insert(0, path_text)


def _resolved_module_path(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None) if module is not None else None
    return Path(module_file).resolve() if module_file else None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_module_from(module_name: str, expected_path: Path, *, package_dir: bool = False) -> None:
    module_path = _resolved_module_path(module_name)
    if module_path is None:
        raise SystemExit(f"{module_name} import has no __file__; cannot verify local layout.")
    expected_path = expected_path.resolve()
    ok = _path_is_within(module_path, expected_path) if package_dir else module_path == expected_path
    if not ok:
        raise SystemExit(
            f"{module_name} was imported from {module_path}, expected "
            f"{expected_path if not package_dir else str(expected_path) + os.sep}"
        )


for import_path in (GRAPHRAG_PACKAGE_ROOT, VIDEORAG_ROOT, RAG_BASELINES_DIR, PROJECT_ROOT, LIGHTRAG_ROOT):
    _prepend_sys_path(import_path)

os.environ.setdefault("EVIQUE_DETECTOR_MODEL", str(DEFAULT_YOLO_MODEL))
os.environ.setdefault(
    "CAPTION_MODEL_PATH",
    DEFAULT_CAPTION_MODEL_PATH.as_posix() if not DEFAULT_CAPTION_MODEL_PATH.drive else str(DEFAULT_CAPTION_MODEL_PATH),
)
os.environ.setdefault(
    "WHISPER_MODEL",
    DEFAULT_WHISPER_SMALL.as_posix() if not DEFAULT_WHISPER_SMALL.drive else str(DEFAULT_WHISPER_SMALL),
)

from NaiveRAG import (  # noqa: E402
    DEFAULT_CHUNK_TOKEN_SIZE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    NAIVE_SYSTEM_PROMPT,
    QueryRecord,
    answer_path,
    attach_reference_answers,
    call_chat,
    collect_answer_sources,
    directory_size_bytes,
    format_chunk_context,
    load_queries,
    load_reference_answers,
    make_openai_client,
    markdown_table,
    read_answer_from_dir,
    read_json,
    result_path,
    run_generation,
    run_quantitative_eval,
    run_winrate_eval,
    save_csv,
    token_count,
    truncate_context,
    videorag_chunking_by_video_segments,
    write_json,
    write_text,
)
from TextVideoRAG import MODEL_NAME as TEXT_VIDEO_MODEL_NAME  # noqa: E402
from TextVideoRAG import TEXT_VIDEO_SYSTEM_PROMPT, TextVideoRAGPipeline  # noqa: E402
from NaiveRAG import MODEL_NAME as NAIVE_MODEL_NAME  # noqa: E402
from NaiveRAG import NaiveRAGPipeline  # noqa: E402
from evique import MODEL_NAME as EVIQUE_MODEL_NAME  # noqa: E402
from evique import EvidenceRetriever, build_evique_from_segments  # noqa: E402
from evique.cost_planner import get_cost_planner_config  # noqa: E402
from evique.evidence_packer import get_evidence_packer_config  # noqa: E402
from evique.event_segmenter import get_event_segmentation_config  # noqa: E402
from evique.retriever import build_prompt_package  # noqa: E402
from evique.standalone_base_builder import (  # noqa: E402
    build_evique_standalone_base,
    standalone_base_file_paths,
)
from evique.utils import CANONICAL_VISUAL_RELATION_FILE, LEGACY_VISUAL_RELATION_FILE  # noqa: E402
from evique.video_identity import EVIQUE_VERSION, EVIQUE_VERSION_LABEL  # noqa: E402
from evique.visual_compactor import (  # noqa: E402
    config_to_dict,
    get_visual_compactor_config,
    visual_compact_metadata,
)

_ensure_module_from("evique", EVIQUE_PACKAGE_DIR, package_dir=True)
_ensure_module_from("NaiveRAG", NAIVERAG_FILE)
_ensure_module_from("TextVideoRAG", TEXTVIDEORAG_FILE)


VIDEO_RAG_MODEL_NAME = "VideoRAG"
LIGHTRAG_MODEL_NAME = "LightRAG"
GRAPHRAG_LOCAL_MODEL_NAME = "GraphRAG-l"
GRAPHRAG_GLOBAL_MODEL_NAME = "GraphRAG-g"
GRAPHRAG_MODEL_NAMES = {GRAPHRAG_LOCAL_MODEL_NAME, GRAPHRAG_GLOBAL_MODEL_NAME}
GRAPHRAG_LITELLM_PROVIDER_PREFIXES = {
    "anthropic",
    "azure",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "huggingface",
    "mistral",
    "ollama",
    "openai",
    "openrouter",
    "siliconflow",
    "vertex_ai",
    "voyage",
}
GRAPHRAG_SENSITIVE_MODEL_FIELDS = {"api_key", "token", "password", "secret"}
GRAPHRAG_OPENAI_SDK_EMBEDDING_TYPE = "openai_sdk_embedding"
GRAPHRAG_UNSUPPORTED_MODEL_FIELDS = {"dimensions", "encoding_format", "deployment_name", "organization"}
DEFAULT_VIDEO = str(DATASET_DIR / "sample_video.mp4")
ALL_MODEL_NAMES = [EVIQUE_MODEL_NAME, VIDEO_RAG_MODEL_NAME, NAIVE_MODEL_NAME, TEXT_VIDEO_MODEL_NAME]
VISUAL_RETRIEVAL_METRIC_FIELDS = [
    "visual_used",
    "visual_chain_attempted",
    "visual_instance_chain_found",
    "visual_failure_reason",
    "visual_chain_evidence_count",
    "visual_intent_evidence_count",
    "visual_relations_enabled",
    "visual_relations_file_generated",
    "visual_object_candidates",
    "visual_track_candidates",
    "visual_relation_candidates",
    "visual_event_candidates",
    "caption_fallback_used",
    "caption_context_evidence_count",
    "temporal_relation_aligned_count",
    "temporal_relation_fallback_count",
    "insufficient_due_to_missing_visual_event",
    "nearby_object_context_used",
    "nearby_object_context_candidate_count",
    "nearby_object_context_in_final_count",
    "density_prompt_used",
    "caption_context_temporal_diversity",
    "cost_planner_enabled",
    "anchor_view",
    "view_order",
    "views_queried",
    "views_skipped",
    "stop_reason",
    "evidence_confidence",
    "evidence_coverage",
    "max_rows_per_view",
    "max_rows_total",
]
EVIDENCE_PACKING_METRIC_FIELDS = [
    "evidence_packer_enabled",
    "packing_strategy",
    "candidate_evidence_count",
    "packed_evidence_count",
    "dropped_evidence_count",
    "estimated_candidate_tokens",
    "estimated_packed_tokens",
    "estimated_candidate_chars",
    "estimated_packed_chars",
    "evidence_token_budget",
    "evidence_char_budget",
    "evidence_core_ratio",
    "evidence_support_ratio",
    "evidence_context_ratio",
    "evidence_min_core_items",
    "evidence_min_packed_items",
    "evidence_max_items",
    "evidence_dedup_threshold",
    "evidence_spatial_relation_min_items",
    "evidence_temporal_event_min_items",
    "evidence_temporal_aware_packing",
    "evidence_temporal_window_segments",
    "evidence_temporal_min_before",
    "evidence_temporal_min_focal",
    "evidence_temporal_min_after",
    "evidence_temporal_max_supplement",
    "packing_view_counts",
    "dropped_view_counts",
    "video_filter_source",
    "strict_video_filter_enabled",
    "packed_video_count",
    "dropped_cross_video_count",
    "spatial_relation_supplement_used",
    "spatial_relation_supplement_count",
    "temporal_event_supplement_used",
    "temporal_event_supplement_count",
    "relation_supplement_used",
    "relation_supplement_count",
    "event_supplement_used",
    "event_supplement_count",
    "temporal_aware_packing_used",
    "temporal_anchor_segment",
    "temporal_before_count",
    "temporal_focal_count",
    "temporal_after_count",
    "temporal_supplement_count",
    "pedestrian_crosswalk_expansion_used",
    "pedestrian_evidence_count",
    "crosswalk_evidence_count",
    "vehicle_near_pedestrian_evidence_count",
    "yielding_supported_by_visual_relation",
    "min_packed_items_target",
    "budget_fill_ratio",
    "mandatory_evidence_ids",
    "budget_exhausted",
]
SUMMARY_COLUMNS = [
    "数据集",
    "视频时长 (min)",
    "数据集大小 (MB)",
    "模型名称",
    "数据集基础索引构建时间 (秒)",
    "基础索引大小 (MB)",
    "方法增量索引大小 (MB)",
    "端到端索引大小 (MB)",
    "平均 query 临时索引大小 (MB)",
    "平均 query 临时索引时间 (秒)",
    "平均查询时间 (秒)",
    "平均准确率得分",
    "检索到的片段或项目数",
    "使用的支持片段或项目数",
    "使用 / 检索",
    "最终证据包平均大小 (字符)",
    "LLM 输入 tokens 平均估计值",
    "答案文件",
    "准确率文件",
]
INDEX_SIZE_POLICY = "query_time_persistent_index_excludes_raw_video_model_weights_answers_eval_cache"
RAW_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MODEL_WEIGHT_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors", ".onnx"}
TEMP_LOG_ARCHIVE_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".log",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
}
EXCLUDED_INDEX_DIR_NAMES = {
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "_cache",
    "answers",
    "cache",
    "evaluation",
    "judge",
    "judges",
    "llm_response_cache",
    "log",
    "logs",
    "report",
    "reports",
    "temp",
    "tmp",
}
LLM_RESPONSE_CACHE_NAMES = {"kv_store_llm_response_cache.json", "llm_response_cache"}
SHARED_DEPENDENCY_INDEX_FILES = [
    "kv_store_video_path.json",
    "kv_store_video_segments.json",
    "kv_store_text_chunks.json",
]
EVIQUE_BASE_METRIC_FIELDS = [
    "evique_base_mode",
    "evique_base_dir",
    "evique_base_generated",
    "evique_base_build_time_seconds",
    "evique_base_size_mb",
    "standalone_base_files",
]
SHARED_BASE_INDEX_BUILD_TIME_KEYS = [
    "shared_base_index_build_time_seconds",
    "shared_dependency_index_build_time_seconds",
    "base_index_build_time_seconds",
    "base_video_text_index_build_time_seconds",
    "video_text_segment_index_build_time_seconds",
    "video_segments_build_time_seconds",
    "text_chunks_build_time_seconds",
]
VIDEO_RAG_METHOD_INDEX_FILES = [
    "graph_chunk_entity_relation.graphml",
    "vdb_entities.json",
    "vdb_chunks.json",
    "vdb_video_segment_feature.json",
]
LIGHTRAG_METHOD_INDEX_FILES = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_full_entities.json",
    "kv_store_full_relations.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
    "kv_store_doc_status.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "vdb_chunks.json",
]
GRAPHRAG_METHOD_INDEX_FILES = [
    "documents.parquet",
    "text_units.parquet",
    "entities.parquet",
    "relationships.parquet",
    "communities.parquet",
    "community_reports.parquet",
    "covariates.parquet",
]
GRAPHRAG_METHOD_INDEX_DIRS = ["lancedb"]
EVIQUE_REQUIRED_METHOD_INDEX_FILES = [
    "scope_view.jsonl",
    "event_view.jsonl",
    "adaptive_event_view.jsonl",
    "target_view.jsonl",
    "track_view.jsonl",
    "evidence_nodes.jsonl",
    "evidence_relations.jsonl",
    "index_manifest.json",
    "graph_stats.json",
    "view_stats.json",
]
EVIQUE_OPTIONAL_METHOD_INDEX_FILES = [
    "visual_object_view.jsonl",
    "visual_track_view.jsonl",
    "visual_event_view.jsonl",
    "keyframe_view.jsonl",
]
VISUAL_COMPACT_METRIC_FIELDS = [
    "visual_compact_enabled",
    "visual_compact_level",
    "visual_relation_count_raw",
    "visual_relation_count_compact",
    "visual_relation_reduction_ratio",
    "visual_relations_enabled",
    "visual_relations_file_generated",
    "visual_relation_before",
    "visual_relation_after",
    "visual_object_count_raw",
    "visual_object_count_compact",
    "visual_track_count_raw",
    "visual_track_count_compact",
    "keyframe_count_raw",
    "keyframe_count_compact",
    "visual_track_point_reduction_ratio",
    "visual_index_size_bytes_raw",
    "visual_index_size_bytes_compact",
    "visual_index_size_reduction_ratio",
]
NAIVE_METHOD_INDEX_PATTERNS = ["cache/naiverag_chunks_*.npz", "naiverag_chunks_*.npz"]
TEXT_VIDEO_METHOD_INDEX_PATTERNS = [
    "cache/textvideorag_chunks_*.npz",
    "cache/textvideorag_visual_segments_*.npz",
    "textvideorag_chunks_*.npz",
    "textvideorag_visual_segments_*.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EVIQUE, VideoRAG, NaiveRAG, TextVideoRAG, LightRAG and paper-style LLM-judge evaluation."
    )
    parser.add_argument("--check-layout", action="store_true", help="Print local path/import checks and exit.")
    parser.add_argument("--check-lightrag", action="store_true", help="Print LightRAG static checks and exit without API calls.")
    parser.add_argument("--check-graphrag", action="store_true", help="Print GraphRAG static checks and exit without API calls.")
    parser.add_argument("--check-graphrag-embedding", action="store_true", help="Run one tiny GraphRAG/OpenAI-compatible embedding API smoke check and exit.")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", action="store_true", help="Show progress bars even when output is redirected.")
    progress_group.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--unified-base-source",
        default=None,
        choices=[UNIFIED_BASE_SOURCE_EVIQUE],
        help="Use one EVIQUE standalone base under <output-root>/evique-workdir/base for all selected baselines.",
    )
    parser.add_argument("--video", action="append", default=None, help="Video path. Can be passed multiple times.")
    parser.add_argument("--dataset-name", default=None, help="Name shown in summary tables.")
    parser.add_argument("--output-root", default=None, help="Output directory for this comparison run.")
    parser.add_argument("--workdir", default=None, help="VideoRAG workdir. Defaults to <output-root>/videorag-workdir.")
    parser.add_argument("--questions", default=None, help="Question JSON. If omitted, use --auto-generate-questions.")
    parser.add_argument("--auto-generate-questions", type=int, default=0, help="Generate N open-ended questions from VideoRAG text if --questions is omitted.")
    parser.add_argument("--reference-answers", default=None, help="Optional reference answers for quantitative scoring.")
    parser.add_argument("--skip-index", action="store_true", help="Reuse an existing VideoRAG workdir.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip answer generation and only evaluate existing outputs.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip LLM judge evaluation.")
    parser.add_argument(
        "--eval-only",
        choices=["all", "quantitative", "winrate"],
        default="all",
        help="Select evaluation stage to run: all, quantitative only, or winrate only.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated models to run: all (legacy four-model set), evique, videorag, naiverag, textvideorag, lightrag, graphrag-l, graphrag-g. "
        "VideoRAG preprocessing runs only when a selected model needs the shared base, or when EVIQUE uses --evique-base-mode shared.",
    )
    parser.add_argument("--videorag-llm-config", choices=["openai_compatible", "openai_4o_mini", "openai", "deepseek_bge"], default="openai_compatible")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Answer LLM for baselines and auto question generation.")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"), help="LLM judge model. Paper uses gpt-4o-mini.")
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--videorag-embedding-dim", type=int, default=None, help="Embedding dimension for VideoRAG vector DB. Inferred for common models.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--chunk-token-size", type=int, default=DEFAULT_CHUNK_TOKEN_SIZE)
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--segment-length", type=int, default=30)
    parser.add_argument("--fine-num-frames", type=int, default=15)
    parser.add_argument("--rough-num-frames", type=int, default=15, help="Set to 15 by default so the shared text store uses 15-frame captions.")
    parser.add_argument("--allow-non-15-frame-segments", action="store_true")
    parser.add_argument("--eval-runs", type=int, default=5)
    parser.add_argument("--single-pass-winrate", action="store_true")
    parser.add_argument(
        "--anchor-winrate-model",
        default=None,
        help="Optional anchor model for win-rate evaluation; only compare this model against each other evaluated model.",
    )
    parser.add_argument(
        "--winrate-reference-model",
        default=None,
        help="Optional reference model for win-rate evaluation; only compare this model against each other selected model.",
    )
    parser.add_argument(
        "--quant-baseline",
        default=NAIVE_MODEL_NAME,
        choices=[
            EVIQUE_MODEL_NAME,
            VIDEO_RAG_MODEL_NAME,
            NAIVE_MODEL_NAME,
            TEXT_VIDEO_MODEL_NAME,
            LIGHTRAG_MODEL_NAME,
            GRAPHRAG_LOCAL_MODEL_NAME,
            GRAPHRAG_GLOBAL_MODEL_NAME,
        ],
    )
    parser.add_argument("--visual-field", default=None, help="Optional segment field for baseline visual text.")
    parser.add_argument("--visual-top-k", type=int, default=8)
    parser.add_argument("--text-weight", type=float, default=0.75)
    parser.add_argument("--visual-weight", type=float, default=0.25)
    parser.add_argument("--chunk-context-ratio", type=float, default=0.75)
    parser.add_argument("--evique-workdir", default=None, help="EVIQUE four-view index directory. Defaults to <output-root>/evique-workdir.")
    parser.add_argument(
        "--evique-base-mode",
        choices=["shared", "standalone"],
        default="shared",
        help="shared reads VideoRAG base files from --workdir; standalone builds/reads EVIQUE base files under --evique-base-dir.",
    )
    parser.add_argument(
        "--evique-base-dir",
        default=None,
        help="Standalone EVIQUE base directory. Defaults to <evique-workdir>/base.",
    )
    parser.add_argument(
        "--evique-rebuild-base",
        action="store_true",
        help="Regenerate standalone EVIQUE base files instead of reusing an existing base.",
    )
    parser.add_argument("--skip-evique-index", action="store_true", help="Reuse an existing EVIQUE index if present.")
    parser.add_argument("--evique-event-window", type=int, default=120, help="Event View fixed-window size in seconds.")
    parser.add_argument("--evique-track-gap", type=int, default=120, help="Max gap in seconds for label-based pseudo-tracks.")
    parser.add_argument("--evique-max-evidence", type=int, default=18, help="Max compact evidence items sent to the answer LLM.")
    parser.add_argument("--evique-token-budget", type=int, default=12000, help="Planner token budget recorded in EVIQUE plans.")
    parser.add_argument("--debug-caption-model", action="store_true", help="Use only for debugging; real VideoRAG queries need the caption model.")
    return parser.parse_args()


def _exists_for_layout(path: Path, *, require_non_empty_dir: bool = False) -> bool:
    if require_non_empty_dir:
        return path.is_dir() and any(path.iterdir())
    return path.exists()


def _layout_path_text(path: Path) -> str:
    return path.as_posix() if not path.drive else str(path)


def _env_set_text(name: str) -> str:
    return "set" if os.getenv(name) else "unset"


def resolve_lightrag_tiktoken_model_name() -> str:
    return os.getenv("LIGHTRAG_TIKTOKEN_MODEL_NAME") or DEFAULT_LIGHTRAG_TIKTOKEN_MODEL_NAME


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{rem:02d}s"


def resolve_progress_enabled(args: argparse.Namespace) -> bool:
    if getattr(args, "no_progress", False):
        return False
    if getattr(args, "progress", False):
        return True
    return bool(sys.stderr.isatty() or sys.stdout.isatty())


def progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "_progress_enabled", False))


def _progress_total(iterable: Any) -> int | None:
    try:
        return len(iterable)
    except TypeError:
        return None


class _PrintProgressIter:
    def __init__(self, iterable: Any, *, desc: str, unit: str = "it") -> None:
        self._iterable = iterable
        self._desc = desc
        self._unit = unit
        self._total = _progress_total(iterable)

    def __iter__(self):
        start = time.perf_counter()
        count = 0
        for count, item in enumerate(self._iterable, start=1):
            elapsed = time.perf_counter() - start
            eta_text = "?"
            if self._total and count > 0:
                eta_text = _format_duration((elapsed / count) * max(self._total - count, 0))
            total_text = str(self._total) if self._total is not None else "?"
            print(
                f"[progress] {self._desc}: {count}/{total_text} {self._unit} "
                f"elapsed={_format_duration(elapsed)} eta={eta_text}",
                file=sys.stderr,
            )
            yield item
        elapsed = time.perf_counter() - start
        total_text = str(self._total) if self._total is not None else str(count)
        print(
            f"[progress] {self._desc}: done {count}/{total_text} {self._unit} "
            f"elapsed={_format_duration(elapsed)}",
            file=sys.stderr,
        )


def progress_iter(args: argparse.Namespace, iterable: Any, *, desc: str, unit: str = "it") -> Any:
    if not progress_enabled(args):
        return iterable
    total = _progress_total(iterable)
    if _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )
    return _PrintProgressIter(iterable, desc=desc, unit=unit)


def progress_stage_start(label: str) -> float:
    print(f"{label} started...")
    return time.perf_counter()


def progress_stage_finish(label: str, started_at: float) -> None:
    print(f"{label} finished in {_format_duration(time.perf_counter() - started_at)}")


class ModelProgress:
    def __init__(self, args: argparse.Namespace, models: list[str]) -> None:
        self._enabled = progress_enabled(args)
        self._models = models
        self._total = len(models)
        self._done = 0
        self._seen: set[str] = set()
        self._started_at = time.perf_counter()
        self._bar = None
        if self._enabled and _tqdm is not None:
            self._bar = _tqdm(
                total=self._total,
                desc="Models",
                unit="model",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=True,
            )

    def mark(self, model_name: str, stage: str = "done") -> None:
        if model_name in self._seen:
            return
        self._seen.add(model_name)
        self._done += 1
        elapsed = time.perf_counter() - self._started_at
        eta_text = "?"
        if self._done > 0:
            eta_text = _format_duration((elapsed / self._done) * max(self._total - self._done, 0))
        if self._bar is not None:
            self._bar.set_description(f"Models: {model_name}")
            self._bar.set_postfix_str(stage)
            self._bar.update(1)
        elif self._enabled:
            print(
                f"[progress] Models: {self._done}/{self._total} {model_name} "
                f"stage={stage} elapsed={_format_duration(elapsed)} eta={eta_text}",
                file=sys.stderr,
            )

    def mark_remaining(self, stage: str = "done") -> None:
        for model_name in self._models:
            self.mark(model_name, stage=stage)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def print_layout_check(args: argparse.Namespace | None = None) -> None:
    evique_module_path = _resolved_module_path("evique")
    naiverag_module_path = _resolved_module_path("NaiveRAG")
    textvideorag_module_path = _resolved_module_path("TextVideoRAG")
    videorag_spec = importlib.util.find_spec("videorag")
    videorag_spec_origin = Path(videorag_spec.origin).resolve() if videorag_spec and videorag_spec.origin else None
    lightrag_spec = importlib.util.find_spec("lightrag")
    lightrag_spec_origin = Path(lightrag_spec.origin).resolve() if lightrag_spec and lightrag_spec.origin else None
    lightrag_module_path: Path | None = None
    lightrag_version = "NO_VERSION"
    lightrag_import_error: str | None = None
    try:
        lightrag_module = __import__("lightrag")
        lightrag_module_path = _resolved_module_path("lightrag")
        lightrag_version = str(getattr(lightrag_module, "__version__", "NO_VERSION"))
    except Exception as exc:  # noqa: BLE001 - layout checks should be diagnostic only.
        lightrag_import_error = f"{type(exc).__name__}: {exc}"

    final_caption_model_path = Path(os.environ.get("CAPTION_MODEL_PATH", str(DEFAULT_CAPTION_MODEL_PATH)))
    final_whisper_model_path = Path(os.environ.get("WHISPER_MODEL", str(DEFAULT_WHISPER_SMALL)))
    unified_source = getattr(args, "unified_base_source", None) if args is not None else None
    layout_output_root = getattr(args, "output_root", None) if args is not None else None
    layout_workdir = getattr(args, "workdir", None) if args is not None else None
    layout_evique_base_dir = getattr(args, "evique_base_dir", None) if args is not None else None
    output_root_text = layout_output_root or "<output-root>"
    evique_base_text = layout_evique_base_dir or f"{output_root_text}/evique-workdir/base"
    videorag_base_text = layout_workdir or f"{output_root_text}/videorag-workdir"
    id_mapping_examples = "(not available until derived base is prepared)"
    nested_key_normalization_enabled = unified_source == UNIFIED_BASE_SOURCE_EVIQUE
    global_segment_id_format = "<video_name>_<index>" if unified_source == UNIFIED_BASE_SOURCE_EVIQUE else "(legacy)"
    removed_segment_token_in_global_ids = unified_source == UNIFIED_BASE_SOURCE_EVIQUE
    if layout_workdir:
        manifest_path = Path(layout_workdir).expanduser() / VIDEORAG_COMPAT_BASE_DERIVATION_FILE
    elif layout_output_root:
        manifest_path = Path(layout_output_root).expanduser() / "videorag-workdir" / VIDEORAG_COMPAT_BASE_DERIVATION_FILE
    else:
        manifest_path = None
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            nested_key_normalization_enabled = bool(manifest.get("nested_key_normalization_enabled"))
            global_segment_id_format = str(manifest.get("videorag_global_segment_id_format") or global_segment_id_format)
            removed_segment_token_in_global_ids = bool(manifest.get("removed_segment_token_in_global_ids"))
            examples = manifest.get("example_id_mappings") or []
            if examples:
                id_mapping_examples = json.dumps(examples[:10], ensure_ascii=False)
            else:
                id_mapping_examples = "(no id mappings recorded)"
        except Exception as exc:  # noqa: BLE001 - layout checks should stay diagnostic.
            id_mapping_examples = f"(unreadable manifest: {type(exc).__name__}: {exc})"

    print("[layout] project_root=" + str(PROJECT_ROOT))
    print(f"[layout] unified_base_source={unified_source or '(legacy)'}")
    print(f"[layout] evique base path={evique_base_text}")
    print(f"[layout] VideoRAG-compatible base path={videorag_base_text}")
    print(f"[layout] VideoRAG-compatible base derived from evique={unified_source == UNIFIED_BASE_SOURCE_EVIQUE}")
    print(f"[layout] VideoRAG-compatible id normalization enabled={unified_source == UNIFIED_BASE_SOURCE_EVIQUE}")
    print(f"[layout] nested key normalization enabled={nested_key_normalization_enabled}")
    print(f"[layout] VideoRAG-compatible global segment id format={global_segment_id_format}")
    print(f"[layout] removed segment token in global ids={removed_segment_token_in_global_ids}")
    print(f"[layout] example id mapping if exists={id_mapping_examples}")
    print(f"[layout] models_root={_layout_path_text(MODELS_ROOT)} exists={MODELS_ROOT.exists()}")
    print(
        f"[layout] default caption model path exists={DEFAULT_CAPTION_MODEL_PATH.exists()} "
        f"path={_layout_path_text(DEFAULT_CAPTION_MODEL_PATH)}"
    )
    print(
        f"[layout] default whisper small path exists={DEFAULT_WHISPER_SMALL.exists()} "
        f"path={_layout_path_text(DEFAULT_WHISPER_SMALL)}"
    )
    print(
        f"[layout] default whisper tiny path exists={DEFAULT_WHISPER_TINY.exists()} "
        f"path={_layout_path_text(DEFAULT_WHISPER_TINY)}"
    )
    print(f"[layout] CAPTION_MODEL_PATH env value={CAPTION_MODEL_PATH_ENV_VALUE or '(unset)'}")
    print(f"[layout] WHISPER_MODEL env value={WHISPER_MODEL_ENV_VALUE or '(unset)'}")
    print(
        f"[layout] final resolved caption model path exists={final_caption_model_path.exists()} "
        f"path={_layout_path_text(final_caption_model_path)}"
    )
    print(
        f"[layout] final resolved whisper model path exists={final_whisper_model_path.exists()} "
        f"path={_layout_path_text(final_whisper_model_path)}"
    )
    print(f"[layout] evique package dir exists={EVIQUE_PACKAGE_DIR.exists()} path={EVIQUE_PACKAGE_DIR}")
    print(f"[layout] evique import path={evique_module_path}")
    print(f"[layout] evique import local={bool(evique_module_path and _path_is_within(evique_module_path, EVIQUE_PACKAGE_DIR))}")
    print(f"[layout] EVIQUE version={EVIQUE_VERSION} / {EVIQUE_VERSION_LABEL}")
    print(f"[layout] videorag root exists={VIDEORAG_ROOT.exists()} path={VIDEORAG_ROOT}")
    print(f"[layout] videorag package exists={VIDEORAG_PACKAGE_DIR.exists()} path={VIDEORAG_PACKAGE_DIR}")
    print(f"[layout] VideoRAG sys.path root={VIDEORAG_ROOT}")
    print(f"[layout] videorag import spec origin={videorag_spec_origin}")
    print(f"[layout] videorag import local={bool(videorag_spec_origin and _path_is_within(videorag_spec_origin, VIDEORAG_PACKAGE_DIR))}")
    print(f"[layout] NaiveRAG.py exists={NAIVERAG_FILE.exists()} path={NAIVERAG_FILE}")
    print(f"[layout] NaiveRAG import path={naiverag_module_path}")
    print(f"[layout] TextVideoRAG.py exists={TEXTVIDEORAG_FILE.exists()} path={TEXTVIDEORAG_FILE}")
    print(f"[layout] TextVideoRAG import path={textvideorag_module_path}")
    print(f"[layout] LightRAG root exists={LIGHTRAG_ROOT.exists()} path={LIGHTRAG_ROOT}")
    print(f"[layout] lightrag package exists={LIGHTRAG_PACKAGE_DIR.exists()} path={LIGHTRAG_PACKAGE_DIR}")
    print(f"[layout] lightrag import spec origin={lightrag_spec_origin}")
    print(f"[layout] lightrag import path={lightrag_module_path}")
    print(f"[layout] lightrag import local={bool(lightrag_module_path and _path_is_within(lightrag_module_path, LIGHTRAG_PACKAGE_DIR))}")
    print(f"[layout] LightRAG version={lightrag_version}")
    if lightrag_import_error:
        print(f"[layout] lightrag import error={lightrag_import_error}")
    print(f"[layout] Dataset dir exists={DATASET_DIR.exists()} path={DATASET_DIR}")
    print(f"[layout] yolo11n.pt exists={DEFAULT_YOLO_MODEL.exists()} path={DEFAULT_YOLO_MODEL}")
    print(f"[layout] EVIQUE_DETECTOR_MODEL={os.environ.get('EVIQUE_DETECTOR_MODEL')}")
    print(f"[layout] CAPTION_MODEL_PATH={os.environ.get('CAPTION_MODEL_PATH')}")
    print(f"[layout] WHISPER_MODEL={os.environ.get('WHISPER_MODEL')}")


def print_lightrag_check() -> None:
    lightrag_module_path: Path | None = None
    lightrag_version = "NO_VERSION"
    lightrag_import_error: str | None = None
    light_cls_error: str | None = None
    query_param_error: str | None = None
    missing_import_names: set[str] = set()
    light_cls_ok = False
    query_param_ok = False

    try:
        lightrag_module = __import__("lightrag")
        lightrag_module_path = _resolved_module_path("lightrag")
        lightrag_version = str(getattr(lightrag_module, "__version__", "NO_VERSION"))
    except ModuleNotFoundError as exc:
        lightrag_import_error = f"{type(exc).__name__}: {exc}"
        if exc.name:
            missing_import_names.add(exc.name)
    except Exception as exc:  # noqa: BLE001 - static check should report instead of fail.
        lightrag_import_error = f"{type(exc).__name__}: {exc}"

    try:
        from lightrag import LightRAG as _LightRAG  # noqa: F401

        light_cls_ok = True
    except ModuleNotFoundError as exc:
        light_cls_error = f"{type(exc).__name__}: {exc}"
        if exc.name:
            missing_import_names.add(exc.name)
    except Exception as exc:  # noqa: BLE001 - dependencies may be missing.
        light_cls_error = f"{type(exc).__name__}: {exc}"

    try:
        from lightrag import QueryParam as _QueryParam  # noqa: F401

        query_param_ok = True
    except ModuleNotFoundError as exc:
        query_param_error = f"{type(exc).__name__}: {exc}"
        if exc.name:
            missing_import_names.add(exc.name)
    except Exception as exc:  # noqa: BLE001 - dependencies may be missing.
        query_param_error = f"{type(exc).__name__}: {exc}"

    wrapper_names = [
        "get_lightrag_api",
        "make_lightrag",
        "run_lightrag_answers",
        "load_lightrag_documents",
    ]
    dependency_names = [
        "openai",
        "tiktoken",
        "nano_vectordb",
        "networkx",
        "pandas",
        "numpy",
        "aiohttp",
        "pipmaster",
    ]
    missing_dependencies: list[str] = []

    print(f"[lightrag-check] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[lightrag-check] LIGHTRAG_ROOT={LIGHTRAG_ROOT}")
    print(f"[lightrag-check] lightrag package path={LIGHTRAG_PACKAGE_DIR}")
    print(f"[lightrag-check] lightrag.__file__={lightrag_module_path}")
    print(f"[lightrag-check] lightrag version={lightrag_version}")
    print(f"[lightrag-check] import local={bool(lightrag_module_path and _path_is_within(lightrag_module_path, LIGHTRAG_PACKAGE_DIR))}")
    if lightrag_import_error:
        print(f"[lightrag-check] import lightrag error={lightrag_import_error}")
    print(f"[lightrag-check] import LightRAG={light_cls_ok}")
    if light_cls_error:
        print(f"[lightrag-check] import LightRAG error={light_cls_error}")
    print(f"[lightrag-check] import QueryParam={query_param_ok}")
    if query_param_error:
        print(f"[lightrag-check] import QueryParam error={query_param_error}")

    for wrapper_name in wrapper_names:
        print(f"[lightrag-check] wrapper {wrapper_name} exists={callable(globals().get(wrapper_name))}")

    for dependency_name in dependency_names:
        exists = importlib.util.find_spec(dependency_name) is not None
        if not exists:
            missing_dependencies.append(dependency_name)
        print(f"[lightrag-check] dependency {dependency_name} exists={exists}")

    print(f"[lightrag-check] OPENAI_BASE_URL={_env_set_text('OPENAI_BASE_URL')}")
    print(f"[lightrag-check] OPENAI_MODEL={_env_set_text('OPENAI_MODEL')}")
    print(f"[lightrag-check] OPENAI_EMBEDDING_MODEL={_env_set_text('OPENAI_EMBEDDING_MODEL')}")
    print(f"[lightrag-check] OPENAI_API_KEY={_env_set_text('OPENAI_API_KEY')}")
    print(f"[lightrag-check] LIGHTRAG_TIKTOKEN_MODEL_NAME env value={os.getenv('LIGHTRAG_TIKTOKEN_MODEL_NAME') or '(unset)'}")
    print(f"[lightrag-check] resolved LightRAG tiktoken model name={resolve_lightrag_tiktoken_model_name()}")

    missing_dependency_names = sorted(set(missing_dependencies) | missing_import_names)
    if missing_dependency_names:
        print(f"[WARN] missing dependencies: {', '.join(missing_dependency_names)}")
    elif lightrag_import_error or light_cls_error or query_param_error:
        print("[WARN] LightRAG API import failed; see import errors above")
    else:
        print("[OK] LightRAG static check passed")



def graphrag_version_from_pyproject() -> str:
    pyproject_path = GRAPHRAG_PACKAGE_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return "NO_VERSION"
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else "NO_VERSION"


def _graphrag_pythonpath_entries() -> list[str]:
    packages_dir = GRAPHRAG_ROOT / "packages"
    entries = [str(path) for path in packages_dir.iterdir() if path.is_dir()] if packages_dir.exists() else []
    if str(GRAPHRAG_PACKAGE_ROOT) not in entries:
        entries.insert(0, str(GRAPHRAG_PACKAGE_ROOT))
    return entries


def _graphrag_env(model_name: str | None = None, embedding_model: str | None = None, adapter_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = _graphrag_pythonpath_entries()
    if adapter_dir is not None:
        pythonpath_entries.insert(0, str(adapter_dir))
    local_pythonpath = os.pathsep.join(pythonpath_entries)
    env["PYTHONPATH"] = local_pythonpath if not existing_pythonpath else local_pythonpath + os.pathsep + existing_pythonpath

    api_key = env.get("GRAPHRAG_API_KEY") or env.get("OPENAI_API_KEY")
    if api_key:
        env["GRAPHRAG_API_KEY"] = api_key

    llm_model = env.get("GRAPHRAG_LLM_MODEL") or model_name or env.get("OPENAI_MODEL")
    if llm_model:
        env["GRAPHRAG_LLM_MODEL"] = llm_model

    resolved_embedding_model = env.get("GRAPHRAG_EMBEDDING_MODEL") or embedding_model or env.get("OPENAI_EMBEDDING_MODEL")
    if resolved_embedding_model:
        env["GRAPHRAG_EMBEDDING_MODEL"] = resolved_embedding_model
    api_base = env.get("GRAPHRAG_API_BASE") or env.get("OPENAI_BASE_URL")
    if api_base:
        env["GRAPHRAG_API_BASE"] = api_base

    base_url = env.get("GRAPHRAG_BASE_URL") or env.get("OPENAI_BASE_URL")
    if base_url:
        env["GRAPHRAG_BASE_URL"] = base_url
    return env


def _graphrag_template_placeholders(text: str) -> list[str]:
    names: set[str] = set()
    for match in re.finditer(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<named>[A-Za-z_][A-Za-z0-9_]*))", text):
        names.add(match.group("braced") or match.group("named"))
    return sorted(names)


def _line_col_for_index(text: str, index: int) -> tuple[int, int]:
    line = text.count("\n", 0, index) + 1
    line_start = text.rfind("\n", 0, index) + 1
    return line, index - line_start + 1


def _line_col_for_placeholder(text: str, placeholder_name: str) -> tuple[int, int]:
    match = re.search(r"\$(?:\{" + re.escape(placeholder_name) + r"\}|" + re.escape(placeholder_name) + r"\b)", text)
    if match:
        return _line_col_for_index(text, match.start())
    return 0, 0


def validate_graphrag_settings_template(settings_path: Path, env: dict[str, str]) -> None:
    if not settings_path.exists():
        return
    text = settings_path.read_text(encoding="utf-8")
    placeholders = _graphrag_template_placeholders(text)
    try:
        Template(text).substitute(env)
    except KeyError as exc:
        missing_name = str(exc.args[0]) if exc.args else "UNKNOWN"
        line, col = _line_col_for_placeholder(text, missing_name)
        location = f"{settings_path}:{line}:{col}" if line else str(settings_path)
        raise SystemExit(
            f"GraphRAG settings template validation failed at {location}: missing environment variable {missing_name!r}. "
            "The runner fills GRAPHRAG_API_KEY from OPENAI_API_KEY for GraphRAG subprocesses; set one of them before indexing/querying."
        ) from exc
    except ValueError as exc:
        match = re.search(r"line (\d+), col (\d+)", str(exc))
        location = str(settings_path)
        if match:
            location = f"{settings_path}:{match.group(1)}:{match.group(2)}"
        raise SystemExit(
            f"GraphRAG settings template validation failed at {location}: {exc}. "
            "This often means settings.yaml contains a bare '$' in a regex; use a pattern such as '.*[.]csv'."
        ) from exc
    placeholder_text = ", ".join(placeholders) if placeholders else "(none)"
    print(f"[GraphRAG] settings template validation OK: {settings_path} placeholders={placeholder_text}")



def _split_graphrag_litellm_model(value: str | None, *, default_provider: str = "openai") -> tuple[str, str]:
    model = (value or "").strip()
    if not model:
        return default_provider, model
    prefix, sep, rest = model.partition("/")
    if sep and prefix.lower() in GRAPHRAG_LITELLM_PROVIDER_PREFIXES and rest:
        return prefix, rest
    return default_provider, model


def _graphrag_litellm_model_name(provider: str, model: str) -> str:
    return f"{provider}/{model}" if provider and model else model


def _resolve_graphrag_completion_settings_fields(model_name: str | None) -> tuple[str, str]:
    return _split_graphrag_litellm_model(model_name, default_provider="openai")


def _resolve_graphrag_embedding_settings_fields(embedding_model: str | None, env: dict[str, str] | None = None) -> tuple[str, str]:
    local_env = env or os.environ
    explicit_litellm_model = local_env.get("GRAPHRAG_LITELLM_EMBEDDING_MODEL")
    if explicit_litellm_model:
        return _split_graphrag_litellm_model(explicit_litellm_model, default_provider="openai")
    return _split_graphrag_litellm_model(embedding_model, default_provider="openai")


def _find_yaml_model_block(lines: list[str], section_name: str, model_id: str) -> tuple[int, int] | None:
    section_line = f"{section_name}:"
    model_line = f"  {model_id}:"
    for idx, line in enumerate(lines):
        if line != section_line:
            continue
        j = idx + 1
        while j < len(lines):
            current = lines[j]
            if current and not current.startswith(" "):
                break
            if current == model_line:
                start = j + 1
                end = start
                while end < len(lines):
                    candidate = lines[end]
                    if not candidate.strip():
                        break
                    indent = len(candidate) - len(candidate.lstrip(" "))
                    if indent <= 2:
                        break
                    end += 1
                return start, end
            j += 1
    return None


def _set_yaml_model_field(lines: list[str], start: int, end: int, field_name: str, value: str, *, after_field: str | None = None) -> int:
    prefix = f"    {field_name}:"
    for idx in range(start, end):
        if lines[idx].startswith(prefix):
            lines[idx] = f"    {field_name}: {value}"
            return end
    insert_at = end
    if after_field:
        after_prefix = f"    {after_field}:"
        for idx in range(start, end):
            if lines[idx].startswith(after_prefix):
                insert_at = idx + 1
                break
    lines.insert(insert_at, f"    {field_name}: {value}")
    return end + 1


def _remove_yaml_model_fields(lines: list[str], start: int, end: int, field_names: set[str]) -> int:
    idx = start
    while idx < end:
        stripped = lines[idx].strip()
        field_name = stripped.split(":", 1)[0] if ":" in stripped else ""
        indent = len(lines[idx]) - len(lines[idx].lstrip(" "))
        if indent == 4 and field_name in field_names:
            del lines[idx]
            end -= 1
            continue
        idx += 1
    return end


def _configure_graphrag_model_block(
    text: str,
    *,
    section_name: str,
    model_id: str,
    model_provider: str | None = None,
    model_name: str | None = None,
    api_base_placeholder: bool = False,
    remove_unsupported_fields: bool = False,
) -> str:
    lines = text.splitlines()
    trailing_newline = text.endswith("\n")
    block = _find_yaml_model_block(lines, section_name, model_id)
    if block is None:
        return text
    start, end = block
    if remove_unsupported_fields:
        end = _remove_yaml_model_fields(lines, start, end, GRAPHRAG_UNSUPPORTED_MODEL_FIELDS)
    if model_provider:
        end = _set_yaml_model_field(lines, start, end, "model_provider", model_provider)
    if model_name:
        end = _set_yaml_model_field(lines, start, end, "model", model_name, after_field="model_provider")
    if api_base_placeholder:
        end = _set_yaml_model_field(lines, start, end, "api_base", "${GRAPHRAG_API_BASE}", after_field="api_key")
    updated = "\n".join(lines)
    return updated + "\n" if trailing_newline else updated


def _yaml_model_block_fields(text: str, section_name: str, model_id: str) -> dict[str, str] | None:
    lines = text.splitlines()
    block = _find_yaml_model_block(lines, section_name, model_id)
    if block is None:
        return None
    start, end = block
    fields: dict[str, str] = {}
    for line in lines[start:end]:
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def _redact_graphrag_settings_value(field_name: str, value: str) -> str:
    lowered = field_name.lower()
    if any(secret in lowered for secret in GRAPHRAG_SENSITIVE_MODEL_FIELDS):
        return "set" if value else "unset"
    return value or "(empty)"


def print_graphrag_settings_summary(settings_path: Path) -> None:
    if not settings_path.exists():
        print(f"[GraphRAG] settings summary path={settings_path} exists=False")
        return
    text = settings_path.read_text(encoding="utf-8")
    print(f"[GraphRAG] settings summary path={settings_path} exists=True")
    specs = [
        ("completion_models", "default_completion_model"),
        ("completion_models", "default_chat_model"),
        ("embedding_models", "default_embedding_model"),
    ]
    for section_name, model_id in specs:
        fields = _yaml_model_block_fields(text, section_name, model_id)
        if fields is None:
            print(f"[GraphRAG] {section_name}.{model_id}: exists=False")
            continue
        print(f"[GraphRAG] {section_name}.{model_id}: exists=True")
        for key in ("model_provider", "model", "api_base", "base_url", "api_key", "type"):
            if key in fields:
                print(f"[GraphRAG]   {key}={_redact_graphrag_settings_value(key, fields[key])}")
        unsupported = sorted(key for key in fields if key in GRAPHRAG_UNSUPPORTED_MODEL_FIELDS)
        print(f"[GraphRAG]   unsupported_fields={', '.join(unsupported) if unsupported else '(none)'}")


def _safe_api_error(exc: Exception, api_key: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if api_key:
        message = message.replace(api_key, "<redacted>")
    message = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-<redacted>", message)
    return message[:800]


def _embedding_response_dimension(response: Any) -> int | None:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        return None
    first = data[0]
    embedding = getattr(first, "embedding", None)
    if embedding is None and isinstance(first, dict):
        embedding = first.get("embedding")
    return len(embedding) if embedding is not None else None


def print_graphrag_embedding_smoke(args: argparse.Namespace) -> None:
    graph_env = _graphrag_env(model_name=args.model, embedding_model=args.embedding_model)
    api_key = graph_env.get("OPENAI_API_KEY") or graph_env.get("GRAPHRAG_API_KEY")
    base_url = graph_env.get("OPENAI_BASE_URL") or graph_env.get("GRAPHRAG_API_BASE") or graph_env.get("GRAPHRAG_BASE_URL")
    raw_embedding_model = graph_env.get("GRAPHRAG_EMBEDDING_MODEL") or args.embedding_model or graph_env.get("OPENAI_EMBEDDING_MODEL")
    provider, model_part = _resolve_graphrag_embedding_settings_fields(raw_embedding_model, graph_env)
    litellm_embedding_model = graph_env.get("GRAPHRAG_LITELLM_EMBEDDING_MODEL") or _graphrag_litellm_model_name(provider, model_part)

    print(f"[graphrag-embedding-smoke] OPENAI_BASE_URL={'set' if base_url else 'unset'}")
    print(f"[graphrag-embedding-smoke] OPENAI_API_KEY={'set' if api_key else 'unset'}")
    print(f"[graphrag-embedding-smoke] raw embedding model={raw_embedding_model}")
    print(f"[graphrag-embedding-smoke] LiteLLM embedding model={litellm_embedding_model}")
    print(f"[graphrag-embedding-smoke] GraphRAG index/query embedding adapter type={GRAPHRAG_OPENAI_SDK_EMBEDDING_TYPE}")
    if not api_key:
        print("[WARN] OPENAI_API_KEY/GRAPHRAG_API_KEY is unset; skipping API smoke calls")
        return

    successes: list[str] = []

    try:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.embeddings.create(model=raw_embedding_model, input=["hello"])
        print(f"[graphrag-embedding-smoke] OpenAI SDK model={raw_embedding_model} success=True dim={_embedding_response_dimension(response)}")
        successes.append("openai-sdk")
    except Exception as exc:  # noqa: BLE001 - explicit smoke diagnostics.
        print(f"[graphrag-embedding-smoke] OpenAI SDK model={raw_embedding_model} success=False error={_safe_api_error(exc, api_key)}")

    try:
        import litellm

        response = litellm.embedding(model=raw_embedding_model, input=["hello"], api_key=api_key, api_base=base_url)
        print(f"[graphrag-embedding-smoke] LiteLLM model={raw_embedding_model} success=True dim={_embedding_response_dimension(response)}")
        successes.append("litellm-raw")
    except Exception as exc:  # noqa: BLE001 - explicit smoke diagnostics.
        print(f"[graphrag-embedding-smoke] LiteLLM model={raw_embedding_model} success=False error={_safe_api_error(exc, api_key)}")

    try:
        import litellm

        response = litellm.embedding(model=litellm_embedding_model, input=["hello"], api_key=api_key, api_base=base_url)
        print(f"[graphrag-embedding-smoke] LiteLLM model={litellm_embedding_model} success=True dim={_embedding_response_dimension(response)}")
        successes.append("litellm-provider")
    except Exception as exc:  # noqa: BLE001 - explicit smoke diagnostics.
        print(f"[graphrag-embedding-smoke] LiteLLM model={litellm_embedding_model} success=False error={_safe_api_error(exc, api_key)}")

    if successes:
        print(f"[OK] GraphRAG embedding smoke passed via {', '.join(successes)}")
    else:
        raise SystemExit("[WARN] GraphRAG embedding smoke failed for all tested call styles")


def ensure_graphrag_openai_sdk_embedding_adapter(graphrag_workdir: Path) -> Path:
    adapter_dir = graphrag_workdir / ".runner_adapters" / "openai_sdk_embedding"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = adapter_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        """
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.embedding.embedding_factory import register_embedding
from graphrag_llm.types import LLMEmbeddingResponse


class OpenAISDKEmbedding(LLMEmbedding):
    _metrics_store: Any
    _tokenizer: Any

    def __init__(
        self,
        *,
        model_id: str,
        model_config: Any,
        tokenizer: Any,
        metrics_store: Any,
        metrics_processor: Any | None = None,
        rate_limiter: Any | None = None,
        retrier: Any | None = None,
        cache: Any | None = None,
        cache_key_creator: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._rate_limiter = rate_limiter
        self._retrier = retrier
        self._cache = cache
        self._cache_key_creator = cache_key_creator
        self._model = model_config.model
        self._api_key = model_config.api_key
        self._base_url = model_config.api_base or getattr(model_config, "base_url", None)
        self._client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            self._client_kwargs["base_url"] = self._base_url

    def _clean_embedding_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.pop("metrics", None)
        kwargs.pop("model", None)
        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        clean_kwargs.pop("dimensions", None)
        clean_kwargs.pop("encoding_format", None)
        return clean_kwargs

    def embedding(self, /, **kwargs: Any) -> LLMEmbeddingResponse:
        clean_kwargs = self._clean_embedding_kwargs(dict(kwargs))
        client = OpenAI(**self._client_kwargs)
        response = client.embeddings.create(model=self._model, **clean_kwargs)
        return LLMEmbeddingResponse(**response.model_dump())

    async def embedding_async(self, /, **kwargs: Any) -> LLMEmbeddingResponse:
        clean_kwargs = self._clean_embedding_kwargs(dict(kwargs))
        client = AsyncOpenAI(**self._client_kwargs)
        response = await client.embeddings.create(model=self._model, **clean_kwargs)
        return LLMEmbeddingResponse(**response.model_dump())

    @property
    def metrics_store(self) -> Any:
        return self._metrics_store

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


register_embedding("openai_sdk_embedding", OpenAISDKEmbedding, scope="singleton")
""".lstrip(),
        encoding="utf-8",
    )
    return adapter_dir


def _graphrag_python_command() -> list[str]:
    return [sys.executable, "-m", "graphrag"]


def _run_graphrag_static_command(command: list[str], timeout_seconds: int = 15) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=_graphrag_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics only.
        return False, f"{type(exc).__name__}: {exc}"
    output = (proc.stdout or proc.stderr or "").strip().splitlines()
    interesting_line = output[-1] if output and output[0].startswith("Traceback") else (output[0] if output else "")
    return proc.returncode == 0, interesting_line[:240]


def _graphrag_keyword_hits() -> dict[str, bool]:
    patterns = {
        "local_search": "local_search",
        "global_search": "global_search",
        "LocalSearch": "LocalSearch",
        "GlobalSearch": "GlobalSearch",
        "--method": "--method",
        "community_reports": "community_reports",
    }
    hits = {key: False for key in patterns}
    package_dir = GRAPHRAG_PACKAGE_ROOT / "graphrag"
    if not package_dir.exists():
        return hits
    for path in package_dir.rglob("*.py"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for key, pattern in patterns.items():
            if not hits[key] and pattern in content:
                hits[key] = True
        if all(hits.values()):
            break
    return hits


def print_graphrag_check() -> None:
    graphrag_module_path: Path | None = None
    graphrag_import_error: str | None = None
    missing_import_names: set[str] = set()
    try:
        graphrag_module = __import__("graphrag")
        module_file = getattr(graphrag_module, "__file__", None)
        graphrag_module_path = Path(module_file).resolve() if module_file else None
    except ModuleNotFoundError as exc:
        graphrag_import_error = f"{type(exc).__name__}: {exc}"
        if exc.name:
            missing_import_names.add(exc.name)
    except Exception as exc:  # noqa: BLE001 - static check should report instead of fail.
        graphrag_import_error = f"{type(exc).__name__}: {exc}"

    dependency_specs = {
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "lancedb": "lancedb",
        "numpy": "numpy",
        "networkx": "networkx",
        "litellm": "litellm",
        "pyyaml": "yaml",
    }
    missing_dependencies: list[str] = []
    python_help_ok, python_help_note = _run_graphrag_static_command(_graphrag_python_command() + ["--help"])
    graphrag_exe = shutil.which("graphrag")
    exe_help_ok = False
    exe_help_note = "not found"
    if graphrag_exe:
        exe_help_ok, exe_help_note = _run_graphrag_static_command([graphrag_exe, "--help"])
    for help_note in (python_help_note, exe_help_note):
        missing_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", help_note or "")
        if missing_match:
            missing_import_names.add(missing_match.group(1))
    keyword_hits = _graphrag_keyword_hits()
    effective_env = _graphrag_env()

    print(f"[graphrag-check] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[graphrag-check] GRAPHRAG_ROOT={GRAPHRAG_ROOT}")
    print(f"[graphrag-check] GRAPHRAG_PACKAGE_ROOT={GRAPHRAG_PACKAGE_ROOT}")
    print(f"[graphrag-check] package path exists={GRAPHRAG_PACKAGE_ROOT.exists()} path={GRAPHRAG_PACKAGE_ROOT}")
    print(f"[graphrag-check] import graphrag={graphrag_module_path is not None}")
    print(f"[graphrag-check] graphrag.__file__={graphrag_module_path}")
    print(f"[graphrag-check] import local={bool(graphrag_module_path and _path_is_within(graphrag_module_path, GRAPHRAG_PACKAGE_ROOT / 'graphrag'))}")
    print(f"[graphrag-check] GraphRAG version={graphrag_version_from_pyproject()}")
    if graphrag_import_error:
        print(f"[graphrag-check] import error={graphrag_import_error}")
    print(f"[graphrag-check] python -m graphrag --help available={python_help_ok} note={python_help_note}")
    print(f"[graphrag-check] graphrag --help available={exe_help_ok} path={graphrag_exe or '(not found)'} note={exe_help_note}")
    for key, hit in keyword_hits.items():
        print(f"[graphrag-check] keyword {key} found={hit}")
    for label, module_name in dependency_specs.items():
        exists = importlib.util.find_spec(module_name) is not None
        if not exists:
            missing_dependencies.append(label)
        print(f"[graphrag-check] dependency {label} exists={exists}")
    print(f"[graphrag-check] OPENAI_BASE_URL={_env_set_text('OPENAI_BASE_URL')}")
    print(f"[graphrag-check] OPENAI_MODEL={_env_set_text('OPENAI_MODEL')}")
    print(f"[graphrag-check] OPENAI_EMBEDDING_MODEL={_env_set_text('OPENAI_EMBEDDING_MODEL')}")
    print(f"[graphrag-check] OPENAI_API_KEY={_env_set_text('OPENAI_API_KEY')}")
    print(f"[graphrag-check] effective GRAPHRAG_API_KEY={'set' if effective_env.get('GRAPHRAG_API_KEY') else 'unset'}")
    print(f"[graphrag-check] effective GRAPHRAG_LLM_MODEL={'set' if effective_env.get('GRAPHRAG_LLM_MODEL') else 'unset'}")
    print(f"[graphrag-check] effective GRAPHRAG_EMBEDDING_MODEL={'set' if effective_env.get('GRAPHRAG_EMBEDDING_MODEL') else 'unset'}")
    print(f"[graphrag-check] effective GraphRAG embedding adapter type={GRAPHRAG_OPENAI_SDK_EMBEDDING_TYPE}")
    print(f"[graphrag-check] effective GRAPHRAG_API_BASE={'set' if effective_env.get('GRAPHRAG_API_BASE') else 'unset'}")
    print(f"[graphrag-check] effective GRAPHRAG_BASE_URL={'set' if effective_env.get('GRAPHRAG_BASE_URL') else 'unset'}")

    smoke_settings_path = PROJECT_ROOT / "comparison_runs" / "short_dance_lightrag_smoke_q3" / "graphrag-workdir" / "settings.yaml"
    if smoke_settings_path.exists():
        print_graphrag_settings_summary(smoke_settings_path)

    missing = sorted(set(missing_dependencies) | missing_import_names)
    if missing:
        print(f"[WARN] missing dependencies: {', '.join(missing)}")
    elif graphrag_import_error:
        print("[WARN] GraphRAG import failed; see import error above")
    elif not (python_help_ok or exe_help_ok):
        print("[WARN] GraphRAG CLI help is unavailable; see notes above")
    else:
        print("[OK] GraphRAG static check passed")


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return value or "dataset"


def parse_model_selection(value: str) -> list[str]:
    aliases = {
        "evique": EVIQUE_MODEL_NAME,
        "evi": EVIQUE_MODEL_NAME,
        "videorag": VIDEO_RAG_MODEL_NAME,
        "video": VIDEO_RAG_MODEL_NAME,
        "naiverag": NAIVE_MODEL_NAME,
        "naive": NAIVE_MODEL_NAME,
        "textvideorag": TEXT_VIDEO_MODEL_NAME,
        "textvideo": TEXT_VIDEO_MODEL_NAME,
        "text": TEXT_VIDEO_MODEL_NAME,
        "lightrag": LIGHTRAG_MODEL_NAME,
        "light": LIGHTRAG_MODEL_NAME,
        "graphrag-l": GRAPHRAG_LOCAL_MODEL_NAME,
        "graphrag_l": GRAPHRAG_LOCAL_MODEL_NAME,
        "graphragl": GRAPHRAG_LOCAL_MODEL_NAME,
        "graph-l": GRAPHRAG_LOCAL_MODEL_NAME,
        "graphlocal": GRAPHRAG_LOCAL_MODEL_NAME,
        "graphrag-local": GRAPHRAG_LOCAL_MODEL_NAME,
        "graphrag-g": GRAPHRAG_GLOBAL_MODEL_NAME,
        "graphrag_g": GRAPHRAG_GLOBAL_MODEL_NAME,
        "graphragg": GRAPHRAG_GLOBAL_MODEL_NAME,
        "graph-g": GRAPHRAG_GLOBAL_MODEL_NAME,
        "graphglobal": GRAPHRAG_GLOBAL_MODEL_NAME,
        "graphrag-global": GRAPHRAG_GLOBAL_MODEL_NAME,
    }
    raw = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw or raw == ["all"]:
        return list(ALL_MODEL_NAMES)
    selected: list[str] = []
    for item in raw:
        if item == "all":
            return list(ALL_MODEL_NAMES)
        if item not in aliases:
            valid = ", ".join(["all"] + sorted(aliases))
            raise SystemExit(f"Unknown model in --models: {item!r}. Valid values: {valid}")
        model_name = aliases[item]
        if model_name not in selected:
            selected.append(model_name)
    return selected


def resolve_winrate_reference_model(args: argparse.Namespace) -> str | None:
    reference = args.winrate_reference_model.strip() if args.winrate_reference_model else None
    anchor = args.anchor_winrate_model.strip() if args.anchor_winrate_model else None
    if reference and anchor and reference != anchor:
        raise SystemExit(
            "Conflicting win-rate reference models: "
            f"--winrate-reference-model={reference!r} and --anchor-winrate-model={anchor!r}."
        )
    return reference or anchor


def infer_embedding_dim(model_name: str, explicit_dim: int | None) -> int:
    if explicit_dim:
        return explicit_dim
    normalized = model_name.lower()
    if "bge-m3" in normalized:
        return 1024
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-3-small" in normalized:
        return 1536
    raise SystemExit(
        f"Cannot infer embedding dimension for {model_name!r}. "
        "Pass --videorag-embedding-dim, e.g. 1024 for BAAI/bge-m3."
    )


def get_videorag_llm_config(args: argparse.Namespace):
    from videorag._llm import (
        LLMConfig,
        deepseek_bge_config,
        gpt_4o_mini_complete,
        openai_4o_mini_config,
        openai_config,
        openai_embedding,
    )

    name = args.videorag_llm_config
    if name == "openai_compatible":
        return LLMConfig(
            embedding_func_raw=openai_embedding,
            embedding_model_name=args.embedding_model,
            embedding_dim=infer_embedding_dim(args.embedding_model, args.videorag_embedding_dim),
            embedding_max_token_size=8192,
            embedding_batch_num=args.embedding_batch_size,
            embedding_func_max_async=16,
            query_better_than_threshold=0.2,
            best_model_func_raw=gpt_4o_mini_complete,
            best_model_name=args.model,
            best_model_max_token_size=32768,
            best_model_max_async=16,
            cheap_model_func_raw=gpt_4o_mini_complete,
            cheap_model_name=args.model,
            cheap_model_max_token_size=32768,
            cheap_model_max_async=16,
        )
    if name == "openai_4o_mini":
        return openai_4o_mini_config
    if name == "openai":
        return openai_config
    if name == "deepseek_bge":
        return deepseek_bge_config
    raise ValueError(name)


def get_videorag_api():
    from videorag import QueryParam, VideoRAG

    _ensure_module_from("videorag", VIDEORAG_PACKAGE_DIR, package_dir=True)
    return QueryParam, VideoRAG


def get_lightrag_api():
    import lightrag
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    _ensure_module_from("lightrag", LIGHTRAG_PACKAGE_DIR, package_dir=True)
    return (
        LightRAG,
        QueryParam,
        openai_complete_if_cache,
        openai_embed,
        EmbeddingFunc,
        str(getattr(lightrag, "__version__", "NO_VERSION")),
    )


def load_video_metadata(video_paths: list[Path]) -> dict[str, float]:
    size_bytes = sum(path.stat().st_size for path in video_paths if path.exists())
    duration_seconds = 0.0
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        for path in video_paths:
            with VideoFileClip(str(path)) as clip:
                duration_seconds += float(clip.duration or 0.0)
    except Exception as exc:  # noqa: BLE001 - metadata should not block the run.
        print(f"[warn] Could not read video duration with moviepy: {exc}")
    return {
        "duration_min": duration_seconds / 60.0,
        "size_mb": size_bytes / (1024 * 1024),
    }


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "n/a", "na", "none", "null", "unknown", "unavailable"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float_or_unknown(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "unknown"


def _format_float_or_na(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def safe_existing_paths(paths: list[Path]) -> list[Path]:
    return _dedupe_paths([path for path in paths if path.exists()])


def _missing_paths(paths: list[Path]) -> list[str]:
    return sorted(str(path) for path in _dedupe_paths(paths) if not path.exists())


def default_index_exclude_predicate(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if suffix in RAW_VIDEO_EXTENSIONS:
        return True
    if suffix in MODEL_WEIGHT_EXTENSIONS:
        return True
    if suffix in TEMP_LOG_ARCHIVE_EXTENSIONS:
        return True
    if name in LLM_RESPONSE_CACHE_NAMES or name.startswith("kv_store_llm_response_cache"):
        return True
    if any(part.startswith("answers-") for part in parts):
        return True
    return any(part in EXCLUDED_INDEX_DIR_NAMES for part in parts)


def _directory_size_bytes_filtered(path: Path, exclude_predicate=None) -> int:
    if not path.exists():
        return 0
    if exclude_predicate and exclude_predicate(path):
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        if exclude_predicate and exclude_predicate(root_path):
            dirs[:] = []
            continue
        if exclude_predicate:
            dirs[:] = [dirname for dirname in dirs if not exclude_predicate(root_path / dirname)]
        for filename in files:
            file_path = root_path / filename
            if exclude_predicate and exclude_predicate(file_path):
                continue
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def directory_size_mb(path: Path, exclude_predicate=default_index_exclude_predicate) -> float:
    return _directory_size_bytes_filtered(path, exclude_predicate) / (1024 * 1024)


def file_size_mb(paths: list[Path]) -> float:
    total = 0
    for path in safe_existing_paths(paths):
        try:
            if path.is_file():
                total += path.stat().st_size
            elif path.is_dir():
                total += _directory_size_bytes_filtered(path, default_index_exclude_predicate)
        except OSError:
            continue
    return total / (1024 * 1024)


def _resolve_evique_base_dir(evique_workdir: Path, evique_base_dir: Path | None = None) -> Path:
    return Path(evique_base_dir).resolve() if evique_base_dir is not None else evique_workdir / "base"


def _shared_base_size_mb(base_dir: Path) -> float:
    return file_size_mb([base_dir / name for name in SHARED_DEPENDENCY_INDEX_FILES])


def _evique_base_metrics(
    *,
    base_mode: str,
    base_dir: Path,
    base_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    generated = bool(base_result.get("generated")) if base_result else False
    build_time_seconds = float(base_result.get("build_time_seconds") or 0.0) if base_result else 0.0
    if base_result and base_result.get("size_mb") is not None:
        size_mb = float(base_result.get("size_mb") or 0.0)
    elif base_mode == "standalone":
        size_mb = directory_size_mb(base_dir)
    else:
        size_mb = _shared_base_size_mb(base_dir)
    if base_mode == "standalone":
        files = (
            [str(path) for path in base_result.get("standalone_base_files", [])]
            if base_result
            else [str(path) for path in standalone_base_file_paths(base_dir)]
        )
    else:
        files = []
    return {
        "evique_base_mode": base_mode,
        "evique_base_dir": str(base_dir),
        "evique_base_generated": generated,
        "evique_base_build_time_seconds": build_time_seconds,
        "evique_base_size_mb": size_mb,
        "standalone_base_files": files,
    }


def _glob_existing_paths(base_dir: Path | None, patterns: list[str]) -> list[Path]:
    if base_dir is None or not base_dir.exists():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in base_dir.glob(pattern) if path.exists())
    return safe_existing_paths(paths)


def _evique_optional_index_paths(evique_workdir: Path) -> list[Path]:
    canonical = evique_workdir / CANONICAL_VISUAL_RELATION_FILE
    manifest_path = evique_workdir / "index_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    include_visual_relations = bool(manifest.get("visual_relations_file_generated"))
    paths = [evique_workdir / name for name in EVIQUE_OPTIONAL_METHOD_INDEX_FILES]
    if include_visual_relations:
        paths.append(canonical)
    globbed = _glob_existing_paths(evique_workdir, ["*_view.jsonl", "evidence_*.jsonl"])
    for path in globbed:
        if path.name == LEGACY_VISUAL_RELATION_FILE and not include_visual_relations:
            continue
        if path.name == LEGACY_VISUAL_RELATION_FILE and canonical.exists():
            continue
        paths.append(path)
    return safe_existing_paths(paths)


def _query_index_paths_for_model(
    model_name: str,
    output_root: Path,
    workdir: Path | None,
    evique_workdir: Path | None,
    model_answer_dir: Path | None = None,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
) -> tuple[list[Path], list[Path], list[str]]:
    output_root = Path(output_root)
    workdir = Path(workdir) if workdir is not None else output_root / "videorag-workdir"
    evique_workdir = (
        Path(evique_workdir) if evique_workdir is not None else output_root / "evique-workdir"
    )

    base_dir = _resolve_evique_base_dir(evique_workdir, evique_base_dir)
    method_required_paths: list[Path] = []
    method_optional_paths: list[Path] = []
    shared_required_paths: list[Path] = []

    if model_name == VIDEO_RAG_MODEL_NAME:
        method_required_paths = [workdir / name for name in VIDEO_RAG_METHOD_INDEX_FILES]
        method_optional_paths = (
            _glob_existing_paths(workdir, ["graph_*.graphml", "vdb_*.json", "*_hnsw.index", "*_hnsw_metadata.pkl"])
        )
        shared_required_paths = _video_text_shared_dependency_paths(workdir, evique_workdir, evique_base_dir)
    elif model_name == EVIQUE_MODEL_NAME:
        method_required_paths = [evique_workdir / name for name in EVIQUE_REQUIRED_METHOD_INDEX_FILES]
        method_optional_paths = _evique_optional_index_paths(evique_workdir)
        if evique_base_mode == "standalone":
            shared_required_paths = standalone_base_file_paths(
                _resolve_evique_base_dir(evique_workdir, evique_base_dir)
            )
        else:
            shared_required_paths = [workdir / name for name in SHARED_DEPENDENCY_INDEX_FILES]
    elif model_name == NAIVE_MODEL_NAME:
        method_optional_paths = _glob_existing_paths(model_answer_dir, NAIVE_METHOD_INDEX_PATTERNS)
        shared_required_paths = _video_text_shared_dependency_paths(workdir, evique_workdir, evique_base_dir)
    elif model_name == TEXT_VIDEO_MODEL_NAME:
        method_optional_paths = _glob_existing_paths(model_answer_dir, TEXT_VIDEO_METHOD_INDEX_PATTERNS)
        shared_required_paths = _video_text_shared_dependency_paths(workdir, evique_workdir, evique_base_dir)
    elif model_name == LIGHTRAG_MODEL_NAME:
        lightrag_workdir = output_root / "lightrag-workdir"
        method_optional_paths = [lightrag_workdir / name for name in LIGHTRAG_METHOD_INDEX_FILES]
        standalone_base_dir = base_dir
        if _evique_standalone_base_has_lightrag_input(standalone_base_dir):
            shared_required_paths = standalone_base_file_paths(standalone_base_dir)
        elif any((workdir / name).exists() for name in SHARED_DEPENDENCY_INDEX_FILES):
            shared_required_paths = [workdir / name for name in SHARED_DEPENDENCY_INDEX_FILES]
    elif model_name in GRAPHRAG_MODEL_NAMES:
        graphrag_output_dir = output_root / "graphrag-workdir" / "output"
        method_optional_paths = [graphrag_output_dir / name for name in GRAPHRAG_METHOD_INDEX_FILES]
        method_optional_paths.extend(graphrag_output_dir / name for name in GRAPHRAG_METHOD_INDEX_DIRS)
        method_optional_paths.extend(_glob_existing_paths(graphrag_output_dir, ["*.parquet", "**/*.parquet"]))
        standalone_base_dir = base_dir
        if _evique_standalone_base_has_lightrag_input(standalone_base_dir):
            shared_required_paths = standalone_base_file_paths(standalone_base_dir)
        elif any((workdir / name).exists() for name in SHARED_DEPENDENCY_INDEX_FILES):
            shared_required_paths = [workdir / name for name in SHARED_DEPENDENCY_INDEX_FILES]

    method_paths = safe_existing_paths(method_required_paths + method_optional_paths)
    shared_paths = safe_existing_paths(shared_required_paths)
    missing = _missing_paths(method_required_paths + shared_required_paths)
    return method_paths, shared_paths, missing


def _legacy_index_size_mb_for_model(
    model_name: str,
    *,
    workdir: Path,
    evique_workdir: Path,
    model_answer_dir: Path | None,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
) -> float:
    if model_name == VIDEO_RAG_MODEL_NAME:
        base_dir = _resolve_evique_base_dir(evique_workdir, evique_base_dir)
        if _videorag_workdir_is_evique_derived(workdir, base_dir):
            output_root = model_answer_dir.parent if model_answer_dir is not None else workdir.parent
            method_paths, _, _ = _query_index_paths_for_model(
                model_name,
                output_root,
                workdir,
                evique_workdir,
                model_answer_dir=model_answer_dir,
                evique_base_mode=evique_base_mode,
                evique_base_dir=evique_base_dir,
            )
            return file_size_mb(method_paths)
        return directory_size_bytes(workdir) / (1024 * 1024)
    if model_name == EVIQUE_MODEL_NAME:
        size_mb = directory_size_bytes(evique_workdir) / (1024 * 1024)
        if evique_base_mode == "standalone" and evique_base_dir is not None:
            resolved_base = Path(evique_base_dir).resolve()
            try:
                resolved_base.relative_to(evique_workdir.resolve())
            except ValueError:
                size_mb += directory_size_mb(resolved_base)
        return size_mb
    if model_name in {NAIVE_MODEL_NAME, TEXT_VIDEO_MODEL_NAME} and model_answer_dir is not None:
        return directory_size_bytes(model_answer_dir / "cache") / (1024 * 1024)
    if model_name == LIGHTRAG_MODEL_NAME:
        if model_answer_dir is not None:
            return directory_size_bytes(model_answer_dir.parent / "lightrag-workdir") / (1024 * 1024)
        return 0.0
    if model_name in GRAPHRAG_MODEL_NAMES:
        if model_answer_dir is not None:
            return directory_size_bytes(model_answer_dir.parent / "graphrag-workdir" / "output") / (1024 * 1024)
        return 0.0
    if model_answer_dir is not None:
        return directory_size_bytes(model_answer_dir) / (1024 * 1024)
    return 0.0


def compute_query_index_size_metrics(
    model_name: str,
    output_root: Path,
    workdir: Path | None,
    evique_workdir: Path | None,
    model_answer_dir: Path | None = None,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
) -> dict[str, Any]:
    method_paths, shared_paths, missing = _query_index_paths_for_model(
        model_name,
        output_root,
        workdir,
        evique_workdir,
        model_answer_dir=model_answer_dir,
        evique_base_mode=evique_base_mode,
        evique_base_dir=evique_base_dir,
    )
    method_size_mb = file_size_mb(method_paths)
    shared_size_mb = file_size_mb(shared_paths)
    metrics: dict[str, Any] = {
        "method_specific_index_size_mb": method_size_mb,
        "shared_dependency_index_size_mb": shared_size_mb,
        "end_to_end_query_index_size_mb": method_size_mb + shared_size_mb,
        "method_incremental_index_size_mb": method_size_mb,
        "base_index_size_mb": shared_size_mb,
        "end_to_end_index_size_mb": method_size_mb + shared_size_mb,
        "index_size_policy": INDEX_SIZE_POLICY,
    }
    if missing:
        metrics["index_size_missing_files"] = missing
    return metrics


def load_visual_compact_metrics(evique_workdir: Path) -> dict[str, Any]:
    manifest_path = evique_workdir / "index_manifest.json"
    graph_stats_path = evique_workdir / "graph_stats.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    graph_stats = read_json(graph_stats_path) if graph_stats_path.exists() else {}
    stats = manifest.get("visual_compact_stats") or graph_stats.get("visual_compact_stats") or {}
    metrics = visual_compact_metadata(stats)
    for key in VISUAL_COMPACT_METRIC_FIELDS:
        if key in manifest:
            metrics[key] = manifest.get(key)
        elif key in graph_stats:
            metrics[key] = graph_stats.get(key)
    return {key: metrics.get(key) for key in VISUAL_COMPACT_METRIC_FIELDS if key in metrics}


def update_comparison_config_with_visual_compact(output_root: Path, evique_workdir: Path) -> None:
    config_path = output_root / "comparison_config.json"
    if not config_path.exists():
        return
    config = read_json(config_path)
    config["visual_compactor"] = config_to_dict(get_visual_compactor_config())
    config["visual_compact_metrics"] = load_visual_compact_metrics(evique_workdir)
    write_json(config, config_path)


def enrich_generation_metrics_with_index_sizes(
    metrics: dict[str, Any],
    model_name: str,
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    model_answer_dir: Path | None,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
    evique_base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(metrics)
    if "index_size_mb" not in enriched:
        enriched["index_size_mb"] = _legacy_index_size_mb_for_model(
            model_name,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=model_answer_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
        )
    legacy_index_size_mb = _float_or_zero(enriched.get("index_size_mb"))
    enriched["legacy_index_size_mb"] = legacy_index_size_mb
    enriched.update(
        compute_query_index_size_metrics(
            model_name,
            output_root,
            workdir,
            evique_workdir,
            model_answer_dir=model_answer_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
        )
    )
    if evique_base_metrics and _model_uses_evique_base_dependency(
        model_name,
        workdir=workdir,
        evique_workdir=evique_workdir,
        evique_base_dir=evique_base_dir,
    ):
        for key, value in evique_base_metrics.items():
            enriched.setdefault(key, value)
    if model_name == EVIQUE_MODEL_NAME:
        enriched.setdefault("evique_version", EVIQUE_VERSION)
        enriched.setdefault("model_version", EVIQUE_VERSION_LABEL)
        base_dir = _resolve_evique_base_dir(evique_workdir, evique_base_dir)
        fallback_base_metrics = _evique_base_metrics(
            base_mode=evique_base_mode,
            base_dir=base_dir if evique_base_mode == "standalone" else workdir,
            base_result=None,
        )
        for key, value in {**fallback_base_metrics, **(evique_base_metrics or {})}.items():
            enriched.setdefault(key, value)
        enriched.update(load_visual_compact_metrics(evique_workdir))
    return enriched


def enrich_generation_metrics_file(
    model_name: str,
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    model_answer_dir: Path,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
    evique_base_metrics: dict[str, Any] | None = None,
) -> None:
    metrics_path = model_answer_dir / "generation_metrics.json"
    if not metrics_path.exists():
        return
    metrics = read_json(metrics_path)
    enriched = enrich_generation_metrics_with_index_sizes(
        metrics,
        model_name,
        output_root=output_root,
        workdir=workdir,
        evique_workdir=evique_workdir,
        model_answer_dir=model_answer_dir,
        evique_base_mode=evique_base_mode,
        evique_base_dir=evique_base_dir,
        evique_base_metrics=evique_base_metrics,
    )
    if enriched != metrics:
        write_json(enriched, metrics_path)


def read_video_segments(workdir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    candidates = [
        workdir / "kv_store_video_segments.json",
        workdir / "evique_video_segments.json",
    ]
    for segment_path in candidates:
        if segment_path.exists():
            return read_json(segment_path)
    raise SystemExit(f"Cannot find video segment store. Checked: {[str(path) for path in candidates]}")


def read_video_path_map(workdir: Path) -> dict[str, str]:
    path = workdir / "kv_store_video_path.json"
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def _evique_standalone_base_has_lightrag_input(base_dir: Path) -> bool:
    return any(
        (base_dir / name).exists()
        for name in (
            "evique_video_segments.json",
            "evique_text_chunks.json",
            "kv_store_video_segments.json",
            "kv_store_text_chunks.json",
        )
    )

def use_unified_evique_base(args: argparse.Namespace) -> bool:
    return getattr(args, "unified_base_source", None) == UNIFIED_BASE_SOURCE_EVIQUE


def _evique_base_source_for_compat_file(base_dir: Path, target_name: str) -> Path:
    direct = base_dir / target_name
    if direct.exists():
        return direct
    fallback_names = {
        "kv_store_video_path.json": "evique_video_path.json",
        "kv_store_video_segments.json": "evique_video_segments.json",
        "kv_store_text_chunks.json": "evique_text_chunks.json",
    }
    fallback = base_dir / fallback_names.get(target_name, target_name)
    return fallback


def _normalize_videorag_global_segment_id(value: str) -> str:
    text = str(value)
    match = re.match(r"^(.+)_segment_0*(\d+)$", text)
    if match:
        video_name, suffix = match.groups()
        return f"{video_name}_{int(suffix)}"
    return text


def _normalize_videorag_numeric_suffix(value: str) -> str:
    text = _normalize_videorag_global_segment_id(str(value))
    if text != str(value):
        return text
    match = re.match(r"^(.+?)([_*])0+(\d+)$", text)
    if match:
        prefix, separator, suffix = match.groups()
        return f"{prefix}{separator}{int(suffix)}"
    if text.isdigit() and len(text) > 1 and text.startswith("0"):
        return str(int(text))
    return text


def _videorag_segment_index_from_key(value: str) -> str:
    text = str(value)
    match = re.match(r"^segment_0*(\d+)$", text)
    if match:
        return str(int(match.group(1)))
    if text.isdigit():
        return str(int(text))
    match = re.match(r"^.+[_*]0*(\d+)$", text)
    if match:
        return str(int(match.group(1)))
    return _normalize_videorag_numeric_suffix(text)


def _videorag_local_segment_id_from_key(value: str) -> str:
    index = _videorag_segment_index_from_key(value)
    return f"segment_{index}" if str(index).isdigit() else _normalize_videorag_numeric_suffix(value)


def _videorag_record_id_mapping(old_value: str, new_value: str, id_map: dict[str, str]) -> None:
    if old_value == new_value:
        return
    existing = id_map.get(old_value)
    if existing is not None and existing != new_value:
        raise SystemExit(
            f"Unified base VideoRAG id normalization conflict for {old_value!r}: "
            f"{existing!r} vs {new_value!r}."
        )
    id_map[old_value] = new_value


def _normalize_videorag_nested_ids(
    value: Any,
    *,
    seed_replacements: dict[str, str] | None = None,
    id_map: dict[str, str] | None = None,
    context: str = "unified base",
) -> tuple[Any, dict[str, str]]:
    replacements = seed_replacements or {}
    observed = id_map if id_map is not None else {}

    def normalize_string(text: str) -> str:
        if text in replacements:
            normalized = replacements[text]
        else:
            normalized = _normalize_videorag_numeric_suffix(text)
        _videorag_record_id_mapping(text, normalized, observed)
        return normalized

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return normalize_string(item)
        if isinstance(item, list):
            return [walk(child) for child in item]
        if isinstance(item, tuple):
            return tuple(walk(child) for child in item)
        if isinstance(item, dict):
            normalized_dict: dict[Any, Any] = {}
            for key, child in item.items():
                normalized_key = normalize_string(key) if isinstance(key, str) else key
                if normalized_key in normalized_dict and normalized_key != key:
                    raise SystemExit(
                        f"Unified base VideoRAG id normalization collision in {context}: "
                        f"{key!r} -> {normalized_key!r}."
                    )
                normalized_dict[normalized_key] = walk(child)
            return normalized_dict
        return item

    return walk(value), observed


def _normalize_videorag_segment_store_ids(video_segments: Any) -> tuple[Any, dict[str, str]]:
    if not isinstance(video_segments, dict):
        return video_segments, {}
    normalized_store: dict[str, Any] = {}
    id_map: dict[str, str] = {}
    for video_name, segments in video_segments.items():
        if not isinstance(segments, dict):
            normalized_store[video_name] = segments
            continue
        normalized_segments: dict[str, Any] = {}
        normalized_store[video_name] = normalized_segments
        for old_key, segment in segments.items():
            old_key_text = str(old_key)
            index = _videorag_segment_index_from_key(old_key_text)
            local_id = _videorag_local_segment_id_from_key(old_key_text)
            global_id = f"{video_name}_{index}" if str(index).isdigit() else f"{video_name}_{local_id}"
            _videorag_record_id_mapping(old_key_text, local_id, id_map)
            _videorag_record_id_mapping(local_id, local_id, id_map)
            _videorag_record_id_mapping(f"{video_name}_{old_key_text}", global_id, id_map)
            _videorag_record_id_mapping(f"{video_name}_{local_id}", global_id, id_map)
            if index in normalized_segments and index != old_key_text:
                raise SystemExit(
                    f"Unified base VideoRAG segment key collision in {video_name}: "
                    f"{old_key_text!r} -> {index!r}."
                )
            normalized_segment, id_map = _normalize_videorag_nested_ids(
                segment,
                seed_replacements=id_map,
                id_map=id_map,
                context="kv_store_video_segments.json",
            )
            normalized_segments[index] = normalized_segment
    return normalized_store, id_map


def _normalize_videorag_text_chunk_ids(text_chunks: Any, replacements: dict[str, str]) -> tuple[Any, dict[str, str]]:
    normalized, id_map = _normalize_videorag_nested_ids(
        text_chunks,
        seed_replacements=replacements,
        context="kv_store_text_chunks.json",
    )
    return normalized, id_map


def _videorag_json_store_non_empty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(data, dict):
        if "data" in data and isinstance(data.get("data"), list):
            return bool(data.get("data"))
        return bool(data)
    if isinstance(data, list):
        return bool(data)
    return False


def videorag_unified_workdir_validation_errors(workdir: Path) -> list[str]:
    workdir = Path(workdir)
    errors: list[str] = []
    graph_path = workdir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists() or graph_path.stat().st_size <= 0:
        errors.append(f"missing or empty {graph_path.name}")
    for name in ("vdb_chunks.json", "vdb_entities.json"):
        if not _videorag_json_store_non_empty(workdir / name):
            errors.append(f"missing or empty {name}")
    for name in ("kv_store_text_chunks.json", "kv_store_video_segments.json"):
        if not (workdir / name).exists():
            errors.append(f"missing {name}")
    manifest_path = workdir / VIDEORAG_COMPAT_BASE_DERIVATION_FILE
    if not manifest_path.exists():
        errors.append(f"missing {VIDEORAG_COMPAT_BASE_DERIVATION_FILE}")
        return errors
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable {VIDEORAG_COMPAT_BASE_DERIVATION_FILE}: {exc}")
        return errors
    required_flags = {
        "id_normalization_enabled": True,
        "nested_key_normalization_enabled": True,
        "removed_segment_token_in_global_ids": True,
    }
    for key, expected in required_flags.items():
        if manifest.get(key) is not expected:
            errors.append(f"{VIDEORAG_COMPAT_BASE_DERIVATION_FILE} has {key}={manifest.get(key)!r}, expected {expected!r}")
    if manifest.get("videorag_global_segment_id_format") != "<video_name>_<index>":
        errors.append(
            f"{VIDEORAG_COMPAT_BASE_DERIVATION_FILE} has videorag_global_segment_id_format="
            f"{manifest.get('videorag_global_segment_id_format')!r}, expected '<video_name>_<index>'"
        )
    return errors


def is_valid_videorag_unified_workdir(workdir: Path) -> bool:
    return not videorag_unified_workdir_validation_errors(workdir)


def prepare_videorag_compatible_base_from_evique(
    evique_base_dir: Path,
    workdir: Path,
    *,
    skip_index: bool = False,
) -> dict[str, Any]:
    evique_base_dir = Path(evique_base_dir).resolve()
    workdir = Path(workdir).resolve()
    if skip_index:
        validation_errors = videorag_unified_workdir_validation_errors(workdir)
        if validation_errors:
            print(
                "[VideoRAG][WARN] unified-base skip-index workdir is missing or inconsistent: " + "; ".join(validation_errors)
            )
            raise RuntimeError(
                "VideoRAG unified-base method index is missing or inconsistent. "
                "Re-run VideoRAG without --skip-index."
            )
        print(f"[VideoRAG] skip-index: using existing unified-base method index {workdir}")
        return read_json(workdir / VIDEORAG_COMPAT_BASE_DERIVATION_FILE)
    workdir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    missing: list[str] = []
    source_paths = {
        target_name: _evique_base_source_for_compat_file(evique_base_dir, target_name)
        for target_name in SHARED_DEPENDENCY_INDEX_FILES
    }
    for target_name, source in source_paths.items():
        if not source.exists():
            missing.append(str(source))
    if missing:
        raise SystemExit(
            "Unified base source evique was requested, but required EVIQUE base files are missing: "
            f"{missing}. Build or upload evique-workdir/base first."
        )

    video_path_source = source_paths["kv_store_video_path.json"]
    video_path_target = workdir / "kv_store_video_path.json"
    shutil.copy2(video_path_source, video_path_target)
    copied.append({"source": str(video_path_source), "target": str(video_path_target), "mode": "copy"})

    video_segments_source = source_paths["kv_store_video_segments.json"]
    video_segments, segment_id_map = _normalize_videorag_segment_store_ids(read_json(video_segments_source))
    video_segments_target = workdir / "kv_store_video_segments.json"
    write_json(video_segments, video_segments_target)
    copied.append({"source": str(video_segments_source), "target": str(video_segments_target), "mode": "normalized-copy"})

    text_chunks_source = source_paths["kv_store_text_chunks.json"]
    text_chunks, chunk_id_map = _normalize_videorag_text_chunk_ids(read_json(text_chunks_source), segment_id_map)
    text_chunks_target = workdir / "kv_store_text_chunks.json"
    write_json(text_chunks, text_chunks_target)
    copied.append({"source": str(text_chunks_source), "target": str(text_chunks_target), "mode": "normalized-copy"})

    full_id_map = {**segment_id_map, **chunk_id_map}
    example_mappings = [
        {"old": old, "new": new}
        for old, new in list(full_id_map.items())[:10]
    ]
    manifest = {
        "unified_base_source": UNIFIED_BASE_SOURCE_EVIQUE,
        "derived_from": str(evique_base_dir),
        "videorag_compatible_base": str(workdir),
        "base_files_are_copies": True,
        "copied_files": copied,
        "id_normalization_enabled": True,
        "segment_id_map_size": len(segment_id_map),
        "text_chunk_id_map_size": len(chunk_id_map),
        "nested_key_normalization_enabled": True,
        "videorag_global_segment_id_format": "<video_name>_<index>",
        "removed_segment_token_in_global_ids": True,
        "example_id_mappings": example_mappings,
    }
    write_json(manifest, workdir / VIDEORAG_COMPAT_BASE_DERIVATION_FILE)
    print(
        f"[unified-base] prepared VideoRAG-compatible base from {evique_base_dir} -> {workdir}; "
        f"normalized_id_mappings={len(full_id_map)}"
    )
    return manifest


def _videorag_workdir_is_evique_derived(workdir: Path | None, evique_base_dir: Path | None) -> bool:
    if workdir is None or evique_base_dir is None:
        return False
    manifest_path = Path(workdir) / VIDEORAG_COMPAT_BASE_DERIVATION_FILE
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("unified_base_source") != UNIFIED_BASE_SOURCE_EVIQUE:
        return False
    derived_from = manifest.get("derived_from")
    if not derived_from:
        return True
    try:
        return Path(derived_from).resolve() == Path(evique_base_dir).resolve()
    except OSError:
        return True


def _video_text_shared_dependency_paths(
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path | None,
) -> list[Path]:
    base_dir = _resolve_evique_base_dir(evique_workdir, evique_base_dir)
    if _videorag_workdir_is_evique_derived(workdir, base_dir):
        return standalone_base_file_paths(base_dir)
    return [workdir / name for name in SHARED_DEPENDENCY_INDEX_FILES]


def _model_uses_evique_base_dependency(
    model_name: str,
    *,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path | None,
) -> bool:
    base_dir = _resolve_evique_base_dir(evique_workdir, evique_base_dir)
    if model_name == EVIQUE_MODEL_NAME:
        return True
    if model_name in {LIGHTRAG_MODEL_NAME, *GRAPHRAG_MODEL_NAMES}:
        return _evique_standalone_base_has_lightrag_input(base_dir)
    if model_name in {VIDEO_RAG_MODEL_NAME, NAIVE_MODEL_NAME, TEXT_VIDEO_MODEL_NAME}:
        return _videorag_workdir_is_evique_derived(workdir, base_dir)
    return False


def _stringify_lightrag_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify_lightrag_value(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _first_lightrag_field(record: dict[str, Any], keys: list[str]) -> Any:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for source in (record, metadata):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _looks_like_lightrag_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    fields = {
        "caption",
        "transcript",
        "content",
        "text",
        "chunk",
        "chunk_text",
        "video_segment_id",
        "segment_id",
        "chunk_id",
        "tokens",
        "start_time",
        "end_time",
        "time",
    }
    return any(key in value for key in fields)


def _iter_lightrag_records(data: Any) -> list[tuple[str | None, str, dict[str, Any]]]:
    records: list[tuple[str | None, str, dict[str, Any]]] = []
    if isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                records.append((None, str(item.get("id") or item.get("chunk_id") or index), item))
            elif item not in (None, ""):
                records.append((None, str(index), {"content": item}))
        return records
    if not isinstance(data, dict):
        if data not in (None, ""):
            return [(None, "0", {"content": data})]
        return []
    if _looks_like_lightrag_record(data):
        return [(None, str(data.get("id") or data.get("chunk_id") or data.get("segment_id") or "0"), data)]
    for outer_key, outer_value in data.items():
        if isinstance(outer_value, dict) and _looks_like_lightrag_record(outer_value):
            records.append((None, str(outer_key), outer_value))
        elif isinstance(outer_value, dict):
            for inner_key, item in outer_value.items():
                if isinstance(item, dict):
                    records.append((str(outer_key), str(inner_key), item))
                elif item not in (None, ""):
                    records.append((str(outer_key), str(inner_key), {"content": item}))
        elif outer_value not in (None, ""):
            records.append((None, str(outer_key), {"content": outer_value}))
    return records


def _format_lightrag_time(record: dict[str, Any]) -> str:
    time_value = _first_lightrag_field(record, ["time", "timestamp", "time_range"])
    if time_value not in (None, ""):
        return _stringify_lightrag_value(time_value)
    start = _first_lightrag_field(record, ["start_time", "start", "start_sec", "begin_time"])
    end = _first_lightrag_field(record, ["end_time", "end", "end_sec", "finish_time"])
    if start not in (None, "") and end not in (None, ""):
        return f"{start}-{end}"
    if start not in (None, ""):
        return str(start)
    return ""


def _lightrag_record_to_document(
    record: dict[str, Any],
    *,
    source_kind: str,
    fallback_video_id: str | None,
    fallback_record_id: str,
) -> tuple[str, str]:
    video_id = _first_lightrag_field(
        record,
        ["video_id", "video_name", "source_vid", "source_video", "video_path", "collection_id"],
    ) or fallback_video_id
    segment_id = _first_lightrag_field(record, ["segment_id", "video_segment_id", "segment", "segment_index"])
    chunk_id = _first_lightrag_field(record, ["chunk_id", "id", "full_doc_id", "chunk_order_index"])
    record_id = _stringify_lightrag_value(segment_id or chunk_id or fallback_record_id) or fallback_record_id
    time_text = _format_lightrag_time(record)

    header_parts = []
    if video_id:
        header_parts.append(f"video_id={_stringify_lightrag_value(video_id)}")
    if segment_id not in (None, ""):
        header_parts.append(f"segment_id={_stringify_lightrag_value(segment_id)}")
    elif chunk_id not in (None, ""):
        header_parts.append(f"chunk_id={_stringify_lightrag_value(chunk_id)}")
    if time_text:
        header_parts.append(f"time={time_text}")
    lines = [f"[{' '.join(header_parts)}]"] if header_parts else []

    caption = _first_lightrag_field(
        record,
        ["caption", "visual_caption", "visual_description", "video_caption", "image_caption", "description"],
    )
    transcript = _first_lightrag_field(record, ["transcript", "asr", "speech", "audio_transcript"])
    content = _first_lightrag_field(record, ["content", "text", "chunk", "chunk_text", "document", "raw_text"])
    if source_kind == "segments" and caption in (None, "") and content not in (None, ""):
        caption = content
        content = None

    if caption not in (None, ""):
        lines.append(f"Caption: {_stringify_lightrag_value(caption).strip()}")
    if transcript not in (None, ""):
        lines.append(f"Transcript: {_stringify_lightrag_value(transcript).strip()}")
    if content not in (None, ""):
        content_text = _stringify_lightrag_value(content).strip()
        if content_text and content_text not in {_stringify_lightrag_value(caption).strip(), _stringify_lightrag_value(transcript).strip()}:
            lines.append(f"Text: {content_text}")

    document = "\n".join(line for line in lines if line.strip()).strip()
    return record_id, document


def _make_unique_lightrag_doc_id(source_name: str, record_id: str, used: set[str]) -> str:
    base = safe_slug(f"{source_name}-{record_id}")[:160] or f"{source_name}-doc"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _convert_lightrag_documents(
    data: Any,
    *,
    source_name: str,
    source_path: Path,
    source_kind: str,
) -> dict[str, Any]:
    documents: list[str] = []
    ids: list[str] = []
    file_paths: list[str] = []
    used_ids: set[str] = set()
    for index, (video_id, fallback_id, record) in enumerate(_iter_lightrag_records(data)):
        record_id, document = _lightrag_record_to_document(
            record,
            source_kind=source_kind,
            fallback_video_id=video_id,
            fallback_record_id=fallback_id or str(index),
        )
        if not document:
            continue
        documents.append(document)
        ids.append(_make_unique_lightrag_doc_id(source_name, record_id or str(index), used_ids))
        file_paths.append(f"{source_path}#{record_id or index}")
    return {
        "documents": documents,
        "ids": ids,
        "file_paths": file_paths,
        "source_name": source_name,
        "source_path": str(source_path),
        "source_kind": source_kind,
        "document_count": len(documents),
    }


def _load_generic_lightrag_input(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return rows
    if path.suffix.lower() == ".json":
        return read_json(path)
    return [{"content": path.read_text(encoding="utf-8")}]


def load_lightrag_documents(
    *,
    output_root: Path,
    workdir: Path,
    evique_base_dir: Path,
    required: bool,
) -> dict[str, Any]:
    candidates: list[tuple[str, Path, str]] = [
        ("evique_video_segments", evique_base_dir / "evique_video_segments.json", "segments"),
        ("evique_text_chunks", evique_base_dir / "evique_text_chunks.json", "chunks"),
        ("evique_compat_video_segments", evique_base_dir / "kv_store_video_segments.json", "segments"),
        ("evique_compat_text_chunks", evique_base_dir / "kv_store_text_chunks.json", "chunks"),
        ("shared_video_segments", workdir / "kv_store_video_segments.json", "segments"),
        ("shared_text_chunks", workdir / "kv_store_text_chunks.json", "chunks"),
    ]
    generic_names = [
        "lightrag_input.jsonl",
        "lightrag_input.json",
        "shared_text_input.jsonl",
        "shared_text_input.json",
        "textual_input.jsonl",
        "textual_input.json",
        "input.jsonl",
        "input.json",
        "input.txt",
    ]
    candidates.extend((f"generic_{path.name}", path, "chunks") for base in (output_root, workdir) for path in (base / name for name in generic_names))

    checked: list[str] = []
    for source_name, path, source_kind in candidates:
        checked.append(str(path))
        if not path.exists():
            continue
        try:
            data = _load_generic_lightrag_input(path) if source_name.startswith("generic_") else read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[LightRAG] skipping unreadable input {path}: {exc}")
            continue
        bundle = _convert_lightrag_documents(
            data,
            source_name=source_name,
            source_path=path,
            source_kind=source_kind,
        )
        if bundle["documents"]:
            bundle["checked_paths"] = checked
            return bundle

    if required:
        raise SystemExit(f"Cannot find usable LightRAG text input. Checked: {checked}")
    return {
        "documents": [],
        "ids": [],
        "file_paths": [],
        "source_name": "existing_lightrag_workdir",
        "source_path": "",
        "source_kind": "existing",
        "document_count": 0,
        "checked_paths": checked,
    }


def validate_frame_counts(video_segments: dict[str, dict[str, dict[str, Any]]], expected: int) -> None:
    bad: list[str] = []
    for video_name, segments in video_segments.items():
        for index, segment in segments.items():
            frame_times = segment.get("frame_times") or []
            if isinstance(frame_times, list) and len(frame_times) != expected:
                bad.append(f"{video_name}_{index}:{len(frame_times)}")
    if bad:
        sample = ", ".join(bad[:10])
        raise SystemExit(
            f"Found segments not using {expected} sampled frames ({sample}). "
            "Rebuild the VideoRAG workdir with --rough-num-frames 15 --fine-num-frames 15, "
            "or pass --allow-non-15-frame-segments only for debugging."
        )


def normalize_generated_questions(raw: Any, limit: int) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        if "questions" in raw:
            raw = raw["questions"]
        else:
            raw = list(raw.values())
    if not isinstance(raw, list):
        raise ValueError("Generated question response is not a list.")
    records: list[dict[str, str]] = []
    for i, item in enumerate(raw[:limit], start=1):
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("query") or "")
        else:
            question = ""
        question = question.strip()
        if question:
            records.append({"id": str(i), "question": question})
    return records


def generate_questions_from_segments(
    video_segments: dict[str, dict[str, dict[str, Any]]],
    *,
    output_path: Path,
    count: int,
    model: str,
    chunk_token_size: int,
    visual_field: str | None,
) -> Path:
    client = make_openai_client()
    chunks = videorag_chunking_by_video_segments(
        video_segments,
        max_token_size=chunk_token_size,
        visual_field=visual_field,
    )
    parts = [f"-----Video Text Chunk {i}-----\n{chunk.content}" for i, chunk in enumerate(chunks.values(), start=1)]
    corpus = truncate_context(parts, max_tokens=14000)
    prompt = f"""You are creating an open-ended QA evaluation set for a long-video RAG comparison.

Use only the following VideoRAG-extracted visual descriptions and ASR transcripts.
Generate {count} diverse, answerable questions that require understanding the video content.
Prefer questions about events, scene changes, spatial details, visible actions, spoken content, and temporal order.

Return JSON only in this schema:
{{"questions": [{{"question": "..."}}, ...]}}

VideoRAG extracted text:
{corpus}
"""
    raw = call_chat(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Return only valid JSON for the requested schema."},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    questions = normalize_generated_questions(data, count)
    write_json(questions, output_path)
    return output_path


def prepare_queries(args: argparse.Namespace, workdir: Path, output_root: Path) -> list[QueryRecord]:
    questions_path = Path(args.questions) if args.questions else None
    video_segments = read_video_segments(workdir)
    if questions_path is None:
        if args.auto_generate_questions <= 0:
            raise SystemExit("Pass --questions, or pass --auto-generate-questions N to create a local question set.")
        questions_path = output_root / "generated_questions.json"
        print(f"[questions] generating {args.auto_generate_questions} questions -> {questions_path}")
        generate_questions_from_segments(
            video_segments,
            output_path=questions_path,
            count=args.auto_generate_questions,
            model=args.model,
            chunk_token_size=args.chunk_token_size,
            visual_field=args.visual_field,
        )
    queries = load_queries(str(questions_path))
    queries = attach_reference_answers(queries, load_reference_answers(args.reference_answers))
    if not queries:
        raise SystemExit(f"No queries loaded from {questions_path}")
    return queries


def make_videorag(args: argparse.Namespace, workdir: Path):
    _, videorag_cls = get_videorag_api()
    return videorag_cls(
        llm=get_videorag_llm_config(args),
        working_dir=str(workdir),
        video_segment_length=args.segment_length,
        rough_num_frames_per_segment=args.rough_num_frames,
        fine_num_frames_per_segment=args.fine_num_frames,
        chunk_token_size=args.chunk_token_size,
    )


def build_videorag_index(
    args: argparse.Namespace,
    workdir: Path,
    video_paths: list[Path],
    *,
    evique_base_dir: Path | None = None,
) -> tuple[VideoRAG, float]:
    metrics_path = workdir / "index_build_metrics.json"
    if args.skip_index and use_unified_evique_base(args):
        validation_errors = videorag_unified_workdir_validation_errors(workdir)
        if validation_errors:
            print(
                "[VideoRAG][WARN] unified-base skip-index workdir is missing or inconsistent: " + "; ".join(validation_errors)
            )
            raise RuntimeError(
                "VideoRAG unified-base method index is missing or inconsistent. "
                "Re-run VideoRAG without --skip-index."
            )
        print(f"[VideoRAG] skip-index: using existing unified-base method index {workdir}")
    rag = make_videorag(args, workdir)
    if args.skip_index:
        existing = read_json(metrics_path) if metrics_path.exists() else {}
        return rag, float(existing.get("index_build_time_seconds", 0.0))

    if use_unified_evique_base(args):
        start = progress_stage_start("[VideoRAG] build method index from unified base")
        video_segments = read_video_segments(workdir)
        # The compatibility workdir already has text chunks copied from the unified base.
        # Clear only the in-memory view so VideoRAG can build its method index from the same segments.
        if hasattr(rag.text_chunks, "_data"):
            rag.text_chunks._data = {}
        asyncio.run(rag.ainsert(video_segments))
        elapsed = time.perf_counter() - start
        progress_stage_finish("[VideoRAG] build method index from unified base", start)
    else:
        start = progress_stage_start("[VideoRAG] build index")
        rag.insert_video(video_path_list=[str(path) for path in video_paths])
        elapsed = time.perf_counter() - start
        progress_stage_finish("[VideoRAG] build index", start)
    write_json(
        {
            "index_build_time_seconds": elapsed,
            "video_paths": [str(path) for path in video_paths],
            "segment_length": args.segment_length,
            "rough_num_frames_per_segment": args.rough_num_frames,
            "fine_num_frames_per_segment": args.fine_num_frames,
            "chunk_token_size": args.chunk_token_size,
            "unified_base_source": getattr(args, "unified_base_source", None) or "legacy",
            "evique_base_dir": str(evique_base_dir) if evique_base_dir is not None else "",
        },
        metrics_path,
    )
    return rag, elapsed


def run_videorag_answers(
    args: argparse.Namespace,
    rag,
    queries: list[QueryRecord],
    *,
    output_dir: Path,
    index_build_seconds: float,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path,
    evique_base_metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = len({q.collection_id for q in queries}) <= 1
    rag.load_caption_model(debug=args.debug_caption_model)
    query_param_cls, _ = get_videorag_api()
    generated: dict[str, str] = {}
    all_results: dict[str, Any] = {}
    query_times: list[float] = []

    for query in progress_iter(args, queries, desc="VideoRAG queries", unit="query"):
        print(f"[VideoRAG] query {query.uid}: {query.question}")
        param = query_param_cls(mode="videorag")
        param.wo_reference = True
        param.return_details = True
        param.response_type = args.response_type
        param.top_k = args.top_k
        param.naive_max_token_for_text_unit = args.max_context_tokens

        start = time.perf_counter()
        raw_response = rag.query(query=query.question, param=param)
        query_seconds = time.perf_counter() - start
        query_times.append(query_seconds)

        if isinstance(raw_response, dict):
            answer = str(raw_response.get("answer") or "")
            details = raw_response.get("details") or {}
        else:
            answer = str(raw_response)
            details = {}

        retrieved_chunks = details.get("retrieved_chunk_ids") or []
        retrieved_segments = details.get("retrieved_segments") or []
        used_segments = details.get("used_segments") or []
        used_chunk_count = int(details.get("used_chunk_count") or 0)
        retrieved_count = len(retrieved_chunks) + len(retrieved_segments)
        used_count = used_chunk_count + len(used_segments)
        video_context = str(details.get("video_context") or "")
        chunk_context = str(details.get("chunk_context") or "")
        system_prompt = str(details.get("system_prompt") or "")
        metrics = {
            "query_time_seconds": query_seconds,
            "retrieved_count": retrieved_count,
            "used_count": used_count,
            "support_ratio": (used_count / retrieved_count) if retrieved_count else 0.0,
            "evidence_chars": len(video_context) + len(chunk_context),
            "llm_input_tokens_estimate": token_count(system_prompt) + token_count(query.question),
        }
        result = {
            "model": VIDEO_RAG_MODEL_NAME,
            "query": asdict(query),
            "answer": answer,
            "retrieval": details,
            "context": {
                "video_data": video_context,
                "chunk_data": chunk_context,
            },
            "metrics": metrics,
        }
        generated[query.uid] = answer
        all_results[query.uid] = result
        write_text(answer, answer_path(output_dir, query, flat=flat))
        write_json(result, result_path(output_dir, query, flat=flat))

    write_json(all_results, output_dir / "all_query_results.json")
    result_metrics = [result["metrics"] for result in all_results.values()]
    generation_metrics = aggregate_generation_metrics(
        VIDEO_RAG_MODEL_NAME,
        result_metrics,
        index_build_seconds=index_build_seconds,
        index_size_mb=_legacy_index_size_mb_for_model(
            VIDEO_RAG_MODEL_NAME,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
        index_size_metrics=compute_query_index_size_metrics(
            VIDEO_RAG_MODEL_NAME,
            output_dir.parent,
            workdir,
            evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
    )
    if evique_base_metrics and _model_uses_evique_base_dependency(
        VIDEO_RAG_MODEL_NAME,
        workdir=workdir,
        evique_workdir=evique_workdir,
        evique_base_dir=evique_base_dir,
    ):
        generation_metrics.update(evique_base_metrics)
    write_json(generation_metrics, output_dir / "generation_metrics.json")
    write_json(
        {
            "model": VIDEO_RAG_MODEL_NAME,
            "workdir": str(workdir),
            "index_build_time_seconds": index_build_seconds,
            "segment_length": args.segment_length,
            "rough_num_frames_per_segment": args.rough_num_frames,
            "fine_num_frames_per_segment": args.fine_num_frames,
            "chunk_token_size": args.chunk_token_size,
            "unified_base_source": getattr(args, "unified_base_source", None) or "legacy",
            "evique_base_dir": str(evique_base_dir),
        },
        output_dir / "run_config.json",
    )
    return generated


def build_evique_index(
    args: argparse.Namespace,
    *,
    evique_workdir: Path,
    video_segments: dict[str, dict[str, dict[str, Any]]],
    workdir: Path | None = None,
    video_path_map: dict[str, str] | None = None,
    video_paths: list[Path] | None = None,
    queries: list[QueryRecord] | None = None,
    evique_base_metrics: dict[str, Any] | None = None,
) -> tuple[Path, float]:
    manifest_path = evique_workdir / "index_manifest.json"
    if (args.skip_index or args.skip_evique_index) and manifest_path.exists():
        manifest = read_json(manifest_path)
        if evique_base_metrics:
            manifest.update(evique_base_metrics)
            write_json(manifest, manifest_path)
        return evique_workdir, float(manifest.get("index_build_time_seconds", 0.0))

    stage_start = progress_stage_start("[EVIQUE] visual index build")
    manifest = build_evique_from_segments(
        video_segments,
        video_paths=video_paths,
        video_path_map=(
            video_path_map
            if video_path_map is not None
            else (read_video_path_map(workdir) if workdir is not None else None)
        ),
        question_records=[asdict(query) for query in queries or []],
        output_dir=evique_workdir,
        event_window_seconds=args.evique_event_window,
        track_gap_seconds=args.evique_track_gap,
        visual_field=args.visual_field,
    )
    if evique_base_metrics:
        manifest.update(evique_base_metrics)
        write_json(manifest, manifest_path)
    progress_stage_finish("[EVIQUE] visual index build", stage_start)
    return evique_workdir, float(manifest.get("index_build_time_seconds", 0.0))


def run_evique_answers(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_dir: Path,
    workdir: Path,
    evique_workdir: Path,
    index_build_seconds: float,
    evique_base_dir: Path,
    evique_base_metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = len({q.collection_id for q in queries}) <= 1
    client = make_openai_client()
    retriever = EvidenceRetriever(
        evique_workdir,
        max_evidence=args.evique_max_evidence,
        token_budget=args.evique_token_budget,
    )
    generated: dict[str, str] = {}
    all_results: dict[str, Any] = {}
    result_metrics: list[dict[str, Any]] = []

    for query in progress_iter(args, queries, desc=f"{EVIQUE_MODEL_NAME} queries", unit="query"):
        print(f"[{EVIQUE_MODEL_NAME}] query {query.uid}: {query.question}")
        start = time.perf_counter()
        package = retriever.retrieve(query.question, query_metadata=asdict(query))
        context = retriever.format_package(package)
        system_prompt = build_prompt_package(package, context, response_type=args.response_type)
        answer = call_chat(
            client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_answer_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.question},
            ],
        )
        query_seconds = time.perf_counter() - start
        retrieved_count = int(package.get("retrieved_count") or 0)
        used_count = int(package.get("used_count") or 0)
        metrics = {
            "query_time_seconds": query_seconds,
            "retrieved_count": retrieved_count,
            "used_count": used_count,
            "support_ratio": (used_count / retrieved_count) if retrieved_count else 0.0,
            "evidence_chars": len(context),
            "llm_input_tokens_estimate": token_count(system_prompt) + token_count(query.question),
        }
        for field in VISUAL_RETRIEVAL_METRIC_FIELDS:
            metrics[field] = package.get(field)
        for field in EVIDENCE_PACKING_METRIC_FIELDS:
            metrics[field] = package.get(field)
        result = {
            "model": EVIQUE_MODEL_NAME,
            "model_version": EVIQUE_VERSION_LABEL,
            "query": asdict(query),
            "answer": answer,
            "retrieval": package,
            "context": {
                "evidence_package": context,
            },
            "metrics": metrics,
        }
        generated[query.uid] = answer
        all_results[query.uid] = result
        result_metrics.append(metrics)
        write_text(answer, answer_path(output_dir, query, flat=flat))
        write_json(result, result_path(output_dir, query, flat=flat))

    write_json(all_results, output_dir / "all_query_results.json")
    generation_metrics = aggregate_generation_metrics(
        EVIQUE_MODEL_NAME,
        result_metrics,
        index_build_seconds=index_build_seconds,
        index_size_mb=_legacy_index_size_mb_for_model(
            EVIQUE_MODEL_NAME,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
        index_size_metrics=compute_query_index_size_metrics(
            EVIQUE_MODEL_NAME,
            output_dir.parent,
            workdir,
            evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
    )
    generation_metrics.update(evique_base_metrics)
    generation_metrics["evique_version"] = EVIQUE_VERSION
    generation_metrics["model_version"] = EVIQUE_VERSION_LABEL
    generation_metrics.update(load_visual_compact_metrics(evique_workdir))
    write_json(generation_metrics, output_dir / "generation_metrics.json")
    visual_compact_metrics = load_visual_compact_metrics(evique_workdir)
    write_json(
        {
            "model": EVIQUE_MODEL_NAME,
            "evique_version": EVIQUE_VERSION,
            "model_version": EVIQUE_VERSION_LABEL,
            "evique_workdir": str(evique_workdir),
            **evique_base_metrics,
            "index_build_time_seconds": index_build_seconds,
            "event_window_seconds": args.evique_event_window,
            "track_gap_seconds": args.evique_track_gap,
            "event_segmentation": get_event_segmentation_config(),
            "cost_based_view_planner": get_cost_planner_config(),
            "evidence_packer": get_evidence_packer_config(),
            "visual_compactor": config_to_dict(get_visual_compactor_config()),
            "visual_compact_metrics": visual_compact_metrics,
            "max_evidence": args.evique_max_evidence,
            "token_budget": args.evique_token_budget,
            **visual_compact_metrics,
        },
        output_dir / "run_config.json",
    )
    return generated


def aggregate_generation_metrics(
    model_name: str,
    result_metrics: list[dict[str, Any]],
    *,
    index_build_seconds: float,
    index_size_mb: float,
    index_size_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [_float_or_none(metric.get(key)) for metric in result_metrics]
        valid_values = [value for value in values if value is not None]
        return (sum(valid_values) / len(valid_values)) if valid_values else None

    def truthy_count(key: str) -> int:
        return sum(1 for metric in result_metrics if bool(metric.get(key)))

    def sum_int(key: str) -> int:
        total = 0
        for metric in result_metrics:
            try:
                total += int(metric.get(key) or 0)
            except (TypeError, ValueError):
                continue
        return total

    def value_counts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric in result_metrics:
            value = metric.get(key)
            if value is None or value == "":
                continue
            text = str(value)
            counts[text] = counts.get(text, 0) + 1
        return counts

    def merge_count_dicts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric in result_metrics:
            value = metric.get(key)
            if not isinstance(value, dict):
                continue
            for item_key, item_value in value.items():
                try:
                    counts[str(item_key)] = counts.get(str(item_key), 0) + int(item_value or 0)
                except (TypeError, ValueError):
                    continue
        return counts

    def avg_len(key: str) -> float:
        if not result_metrics:
            return 0.0
        total = 0
        for metric in result_metrics:
            value = metric.get(key)
            total += len(value) if isinstance(value, list) else (1 if value else 0)
        return total / len(result_metrics)

    metrics = {
        "model": model_name,
        "index_build_time_seconds": index_build_seconds,
        "index_size_mb": index_size_mb,
        "legacy_index_size_mb": index_size_mb,
        "method_specific_index_size_mb": 0.0,
        "shared_dependency_index_size_mb": 0.0,
        "end_to_end_query_index_size_mb": 0.0,
        "index_size_policy": INDEX_SIZE_POLICY,
        "avg_query_time_seconds": avg("query_time_seconds"),
        "avg_evidence_chars": avg("evidence_chars"),
        "avg_llm_input_tokens_estimate": avg("llm_input_tokens_estimate"),
        "avg_retrieved_count": avg("retrieved_count"),
        "avg_used_count": avg("used_count"),
        "avg_support_ratio": avg("support_ratio"),
        "metric_source_counts": value_counts("metric_source"),
    }
    if index_size_metrics:
        metrics.update(index_size_metrics)

    if any(any(field in metric for field in VISUAL_RETRIEVAL_METRIC_FIELDS) for metric in result_metrics):
        metrics.update(
            {
                "visual_used_count": truthy_count("visual_used"),
                "visual_chain_attempted_count": truthy_count("visual_chain_attempted"),
                "visual_instance_chain_found_count": truthy_count("visual_instance_chain_found"),
                "avg_visual_chain_evidence_count": avg("visual_chain_evidence_count"),
                "avg_visual_intent_evidence_count": avg("visual_intent_evidence_count"),
                "visual_relations_enabled": bool(truthy_count("visual_relations_enabled")),
                "visual_relations_file_generated": bool(truthy_count("visual_relations_file_generated")),
                "avg_visual_object_candidates": avg("visual_object_candidates"),
                "avg_visual_track_candidates": avg("visual_track_candidates"),
                "avg_visual_relation_candidates": avg("visual_relation_candidates"),
                "avg_visual_event_candidates": avg("visual_event_candidates"),
                "visual_failure_reason_counts": value_counts("visual_failure_reason"),
                "caption_fallback_used_count": truthy_count("caption_fallback_used"),
                "caption_context_evidence_count": sum_int("caption_context_evidence_count"),
                "temporal_relation_aligned_count": sum_int("temporal_relation_aligned_count"),
                "temporal_relation_fallback_count": sum_int("temporal_relation_fallback_count"),
                "insufficient_due_to_missing_visual_event_count": truthy_count("insufficient_due_to_missing_visual_event"),
                "nearby_object_context_used_count": truthy_count("nearby_object_context_used"),
                "nearby_object_context_candidate_count": sum_int("nearby_object_context_candidate_count"),
                "nearby_object_context_in_final_count": sum_int("nearby_object_context_in_final_count"),
                "density_prompt_used_count": truthy_count("density_prompt_used"),
                "caption_context_temporal_diversity_avg": avg("caption_context_temporal_diversity"),
            }
        )

    if any("cost_planner_enabled" in metric for metric in result_metrics):
        metrics.update(
            {
                "cost_based_view_planner": "on" if truthy_count("cost_planner_enabled") else "off",
                "cost_planner_enabled_count": truthy_count("cost_planner_enabled"),
                "anchor_view_counts": value_counts("anchor_view"),
                "stop_reason_counts": value_counts("stop_reason"),
                "avg_views_queried": avg_len("views_queried"),
                "avg_views_skipped": avg_len("views_skipped"),
                "avg_evidence_confidence": avg("evidence_confidence"),
                "avg_evidence_coverage": avg("evidence_coverage"),
                "max_rows_total_config": max(
                    (int(metric.get("max_rows_total") or 0) for metric in result_metrics),
                    default=0,
                ),
            }
        )

    if any("evidence_packer_enabled" in metric for metric in result_metrics):
        strategy_counts = value_counts("packing_strategy")
        metrics.update(
            {
                "evidence_packer_enabled": bool(truthy_count("evidence_packer_enabled")),
                "evidence_packer_enabled_count": truthy_count("evidence_packer_enabled"),
                "evidence_token_budget": max(
                    (int(metric.get("evidence_token_budget") or 0) for metric in result_metrics),
                    default=0,
                ),
                "evidence_char_budget": max(
                    (int(metric.get("evidence_char_budget") or 0) for metric in result_metrics),
                    default=0,
                ),
                "evidence_max_items": max(
                    (int(metric.get("evidence_max_items") or 0) for metric in result_metrics),
                    default=0,
                ),
                "candidate_evidence_count": sum_int("candidate_evidence_count"),
                "packed_evidence_count": sum_int("packed_evidence_count"),
                "dropped_evidence_count": sum_int("dropped_evidence_count"),
                "estimated_candidate_tokens": sum_int("estimated_candidate_tokens"),
                "estimated_packed_tokens": sum_int("estimated_packed_tokens"),
                "estimated_candidate_chars": sum_int("estimated_candidate_chars"),
                "estimated_packed_chars": sum_int("estimated_packed_chars"),
                "avg_candidate_evidence_count": avg("candidate_evidence_count"),
                "avg_packed_evidence_count": avg("packed_evidence_count"),
                "avg_dropped_evidence_count": avg("dropped_evidence_count"),
                "avg_estimated_candidate_tokens": avg("estimated_candidate_tokens"),
                "avg_estimated_packed_tokens": avg("estimated_packed_tokens"),
                "avg_budget_fill_ratio": avg("budget_fill_ratio"),
                "avg_packed_video_count": avg("packed_video_count"),
                "packing_strategy": next(iter(strategy_counts), ""),
                "packing_strategy_counts": strategy_counts,
                "packing_view_counts": merge_count_dicts("packing_view_counts"),
                "dropped_view_counts": merge_count_dicts("dropped_view_counts"),
                "video_filter_source_counts": value_counts("video_filter_source"),
                "strict_video_filter_enabled_count": truthy_count("strict_video_filter_enabled"),
                "dropped_cross_video_count": sum_int("dropped_cross_video_count"),
                "spatial_relation_supplement_used_count": truthy_count("spatial_relation_supplement_used"),
                "spatial_relation_supplement_count": sum_int("spatial_relation_supplement_count"),
                "temporal_event_supplement_used_count": truthy_count("temporal_event_supplement_used"),
                "temporal_event_supplement_count": sum_int("temporal_event_supplement_count"),
                "relation_supplement_used_count": truthy_count("relation_supplement_used"),
                "relation_supplement_count": sum_int("relation_supplement_count"),
                "event_supplement_used_count": truthy_count("event_supplement_used"),
                "event_supplement_count": sum_int("event_supplement_count"),
                "temporal_aware_packing_used_count": truthy_count("temporal_aware_packing_used"),
                "temporal_before_count": sum_int("temporal_before_count"),
                "temporal_focal_count": sum_int("temporal_focal_count"),
                "temporal_after_count": sum_int("temporal_after_count"),
                "temporal_supplement_count": sum_int("temporal_supplement_count"),
                "pedestrian_crosswalk_expansion_used_count": truthy_count("pedestrian_crosswalk_expansion_used"),
                "pedestrian_evidence_count": sum_int("pedestrian_evidence_count"),
                "crosswalk_evidence_count": sum_int("crosswalk_evidence_count"),
                "vehicle_near_pedestrian_evidence_count": sum_int("vehicle_near_pedestrian_evidence_count"),
                "yielding_supported_by_visual_relation_count": truthy_count("yielding_supported_by_visual_relation"),
                "avg_min_packed_items_target": avg("min_packed_items_target"),
                "budget_exhausted": bool(truthy_count("budget_exhausted")),
                "budget_exhausted_count": truthy_count("budget_exhausted"),
            }
        )

    return metrics


def make_baseline_args(args: argparse.Namespace, *, output_dir: Path, workdir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        workdir=str(workdir),
        video_segments_json=None,
        questions=args.questions or "",
        collection=None,
        reference_answers=args.reference_answers,
        visual_field=args.visual_field,
        fine_num_frames=args.fine_num_frames,
        chunk_token_size=args.chunk_token_size,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        model=args.model,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_answer_tokens=args.max_answer_tokens,
        response_type=args.response_type,
        output_dir=str(output_dir),
        limit=None,
        skip_generation=False,
        run_eval=False,
        single_pass_winrate=args.single_pass_winrate,
        eval_runs=args.eval_runs,
        videorag_answers=None,
        naiverag_answers=None,
        textvideorag_answers=None,
        evique_answers=None,
        quant_baseline=args.quant_baseline,
        visual_top_k=args.visual_top_k,
        text_weight=args.text_weight,
        visual_weight=args.visual_weight,
        chunk_context_ratio=args.chunk_context_ratio,
    )


def run_baseline_answers(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path,
    evique_base_metrics: dict[str, Any],
    output_dir: Path,
    model_name: str,
    pipeline_factory,
) -> dict[str, str]:
    baseline_args = make_baseline_args(args, output_dir=output_dir, workdir=workdir)
    generated = run_generation(baseline_args, queries, pipeline_factory=pipeline_factory, model_name=model_name)
    enrich_generation_metrics_file(
        model_name,
        output_root=output_root,
        workdir=workdir,
        evique_workdir=evique_workdir,
        model_answer_dir=output_dir,
        evique_base_mode=args.evique_base_mode,
        evique_base_dir=evique_base_dir,
        evique_base_metrics=evique_base_metrics,
    )
    return generated


def _openai_base_url_from_env() -> str | None:
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def make_lightrag(args: argparse.Namespace, lightrag_workdir: Path):
    try:
        (
            lightrag_cls,
            query_param_cls,
            openai_complete_if_cache,
            openai_embed,
            embedding_func_cls,
            version,
        ) = get_lightrag_api()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"LightRAG dependency is missing: {exc}. Install the local LightRAG requirements before running --models LightRAG."
        ) from exc

    base_url = _openai_base_url_from_env()
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = args.embedding_model
    lightrag_tiktoken_model_name = resolve_lightrag_tiktoken_model_name()
    embedding_dim = infer_embedding_dim(embedding_model, args.videorag_embedding_dim)
    openai_embed_raw = getattr(openai_embed, "func", openai_embed)

    async def lightrag_llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key", api_key)
        return await openai_complete_if_cache(
            args.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )

    async def lightrag_embedding_func(
        texts: list[str],
        context: str = "document",
        **kwargs: Any,
    ):
        kwargs.setdefault("model", embedding_model)
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key", api_key)
        kwargs.setdefault("context", context)
        return await openai_embed_raw(texts, **kwargs)

    embedding_func = embedding_func_cls(
        embedding_dim=embedding_dim,
        func=lightrag_embedding_func,
        max_token_size=8192,
        send_dimensions=False,
        model_name=embedding_model,
        supports_asymmetric=True,
    )
    rag = lightrag_cls(
        working_dir=str(lightrag_workdir),
        llm_model_func=lightrag_llm_model_func,
        llm_model_name=args.model,
        embedding_func=embedding_func,
        embedding_batch_num=args.embedding_batch_size,
        chunk_token_size=args.chunk_token_size,
        tiktoken_model_name=lightrag_tiktoken_model_name,
    )
    return rag, query_param_cls, version, embedding_dim



def _normalised_metric_aliases(metrics: dict[str, Any], *, query_id: str | None = None) -> dict[str, Any]:
    normalised = dict(metrics)
    if query_id is not None:
        normalised.setdefault("query_id", query_id)
    normalised.setdefault("retrieved_item_count", normalised.get("retrieved_count"))
    normalised.setdefault("used_support_item_count", normalised.get("used_count"))
    normalised.setdefault("evidence_context_chars", normalised.get("evidence_chars"))
    return normalised


def _retrieval_metric_record(*, query_id: str | None, query_time_seconds: float | None, retrieved_count: Any, used_count: Any, evidence_chars: Any, llm_input_tokens: Any, metric_source: str, retrieval_counts_available: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    retrieved_value = _float_or_none(retrieved_count)
    used_value = _float_or_none(used_count)
    support_ratio = None
    if retrieved_value is not None and used_value is not None:
        support_ratio = (used_value / retrieved_value) if retrieved_value > 0 else 0.0
    metrics: dict[str, Any] = {
        "query_id": query_id,
        "query_time_seconds": query_time_seconds,
        "retrieved_count": retrieved_value,
        "used_count": used_value,
        "support_ratio": support_ratio,
        "evidence_chars": _float_or_none(evidence_chars),
        "llm_input_tokens_estimate": _float_or_none(llm_input_tokens),
        "retrieved_item_count": retrieved_value,
        "used_support_item_count": used_value,
        "evidence_context_chars": _float_or_none(evidence_chars),
        "metric_source": metric_source,
        "retrieval_counts_available": retrieval_counts_available,
    }
    if extra:
        metrics.update(extra)
    return metrics


def _unavailable_retrieval_metrics(*, query_id: str | None, query_time_seconds: float | None, reason: str | None = None) -> dict[str, Any]:
    extra = {"metric_unavailable_reason": reason} if reason else None
    return _retrieval_metric_record(query_id=query_id, query_time_seconds=query_time_seconds, retrieved_count=None, used_count=None, evidence_chars=None, llm_input_tokens=None, metric_source="unavailable", retrieval_counts_available=False, extra=extra)


def _list_section(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _lightrag_context_text(data_section: dict[str, Any]) -> str:
    parts: list[str] = []
    for chunk in _list_section(data_section.get("chunks")):
        if isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text") or chunk.get("description")
            if content:
                parts.append(str(content))
        elif chunk:
            parts.append(str(chunk))
    for entity in _list_section(data_section.get("entities")):
        if isinstance(entity, dict):
            text = " ".join(str(bit) for bit in (entity.get("entity_name"), entity.get("entity_type"), entity.get("description")) if bit)
            if text:
                parts.append(text)
        elif entity:
            parts.append(str(entity))
    for relation in _list_section(data_section.get("relationships")):
        if isinstance(relation, dict):
            text = " ".join(str(bit) for bit in (relation.get("src_id"), relation.get("tgt_id"), relation.get("description"), relation.get("keywords")) if bit)
            if text:
                parts.append(text)
        elif relation:
            parts.append(str(relation))
    return "\n".join(parts)


def _lightrag_query_metrics_from_data(raw_data: dict[str, Any] | None, *, query_id: str | None, question: str, query_time_seconds: float | None) -> dict[str, Any]:
    if not isinstance(raw_data, dict):
        return _unavailable_retrieval_metrics(query_id=query_id, query_time_seconds=query_time_seconds, reason="LightRAG structured context was not returned.")
    data_section = raw_data.get("data") if isinstance(raw_data.get("data"), dict) else {}
    counts = {
        "entities": len(_list_section(data_section.get("entities"))),
        "relationships": len(_list_section(data_section.get("relationships"))),
        "chunks": len(_list_section(data_section.get("chunks"))),
        "references": len(_list_section(data_section.get("references"))),
    }
    item_count = counts["entities"] + counts["relationships"] + counts["chunks"]
    context_text = _lightrag_context_text(data_section)
    if item_count <= 0:
        status = str(raw_data.get("status") or "").lower()
        if status == "failure" or raw_data.get("message"):
            return _retrieval_metric_record(query_id=query_id, query_time_seconds=query_time_seconds, retrieved_count=0, used_count=0, evidence_chars=0, llm_input_tokens=None, metric_source="lightrag_structured_context_empty", retrieval_counts_available=True, extra={"lightrag_context_counts": counts, "raw_status": raw_data.get("status")})
        return _unavailable_retrieval_metrics(query_id=query_id, query_time_seconds=query_time_seconds, reason="LightRAG structured context had no countable data section.")
    serialized_context = context_text or json.dumps(data_section, ensure_ascii=False)
    return _retrieval_metric_record(query_id=query_id, query_time_seconds=query_time_seconds, retrieved_count=item_count, used_count=item_count, evidence_chars=len(serialized_context), llm_input_tokens=token_count(question) + token_count(serialized_context), metric_source="lightrag_structured_context", retrieval_counts_available=True, extra={"lightrag_context_counts": counts})


def _parse_graphrag_citation_counts(answer: str) -> dict[str, int]:
    counts = {"sources": 0, "entities": 0, "relationships": 0, "reports": 0, "claims": 0, "covariates": 0}
    label_map = {"source": "sources", "sources": "sources", "entity": "entities", "entities": "entities", "relationship": "relationships", "relationships": "relationships", "report": "reports", "reports": "reports", "claim": "claims", "claims": "claims", "covariate": "covariates", "covariates": "covariates"}
    for block in re.findall(r"\[Data:\s*(.*?)\]", answer or "", flags=re.IGNORECASE | re.DOTALL):
        for label, body in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", block):
            key = label_map.get(label.strip().lower())
            if not key:
                continue
            items = [item.strip() for item in re.split(r"[,;]", body) if item.strip()]
            if not items and body.strip():
                items = [body.strip()]
            counts[key] += len(items)
    counts["total"] = sum(counts.values())
    return counts


def _graphrag_query_metrics_from_answer(answer: str, *, query_id: str | None, question: str, query_time_seconds: float | None) -> dict[str, Any]:
    citation_counts = _parse_graphrag_citation_counts(answer)
    total = int(citation_counts.get("total") or 0)
    if total <= 0:
        return _unavailable_retrieval_metrics(query_id=query_id, query_time_seconds=query_time_seconds, reason="GraphRAG answer did not contain parseable [Data: ...] citations.")
    return _retrieval_metric_record(query_id=query_id, query_time_seconds=query_time_seconds, retrieved_count=total, used_count=total, evidence_chars=len(answer or ""), llm_input_tokens=token_count(question) + token_count(answer or ""), metric_source="answer_citation_estimate", retrieval_counts_available=True, extra={"citation_counts": citation_counts})


def run_lightrag_answers(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path,
    output_dir: Path,
    lightrag_workdir: Path,
) -> dict[str, str]:
    return asyncio.run(
        _run_lightrag_answers_async(
            args,
            queries,
            output_root=output_root,
            workdir=workdir,
            evique_workdir=evique_workdir,
            evique_base_dir=evique_base_dir,
            output_dir=output_dir,
            lightrag_workdir=lightrag_workdir,
        )
    )


async def _run_lightrag_answers_async(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path,
    output_dir: Path,
    lightrag_workdir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lightrag_workdir.mkdir(parents=True, exist_ok=True)
    if args.skip_generation:
        print(f"[LightRAG] skip-generation: leaving existing outputs untouched in {output_dir}")
        return {}
    flat = len({q.collection_id for q in queries}) <= 1
    input_bundle = load_lightrag_documents(
        output_root=output_root,
        workdir=workdir,
        evique_base_dir=evique_base_dir,
        required=not args.skip_index,
    )
    rag, query_param_cls, lightrag_version, embedding_dim = make_lightrag(args, lightrag_workdir)
    index_metrics_path = lightrag_workdir / "index_build_metrics.json"
    index_build_seconds = 0.0
    if args.skip_index and index_metrics_path.exists():
        existing_metrics = read_json(index_metrics_path)
        index_build_seconds = float(existing_metrics.get("index_build_time_seconds") or 0.0)

    generated: dict[str, str] = {}
    all_results: dict[str, Any] = {}
    result_metrics: list[dict[str, Any]] = []
    await rag.initialize_storages()
    try:
        if args.skip_index:
            print(f"[LightRAG] skip-index: using existing workdir {lightrag_workdir}")
        else:
            print(
                f"[LightRAG] indexing {input_bundle['document_count']} documents "
                f"from {input_bundle['source_name']} -> {lightrag_workdir}"
            )
            start = progress_stage_start("[LightRAG] insert/build index")
            await rag.ainsert(
                input_bundle["documents"],
                ids=input_bundle["ids"],
                file_paths=input_bundle["file_paths"],
            )
            index_build_seconds = time.perf_counter() - start
            progress_stage_finish("[LightRAG] insert/build index", start)
            write_json(
                {
                    "model": LIGHTRAG_MODEL_NAME,
                    "lightrag_version": lightrag_version,
                    "index_build_time_seconds": index_build_seconds,
                    "workdir": str(lightrag_workdir),
                    "input_source": input_bundle["source_name"],
                    "input_source_path": input_bundle["source_path"],
                    "input_source_kind": input_bundle["source_kind"],
                    "document_count": input_bundle["document_count"],
                    "llm_model": args.model,
                    "embedding_model": args.embedding_model,
                    "embedding_dim": embedding_dim,
                    "query_mode": "hybrid",
                    "tiktoken_model_name": resolve_lightrag_tiktoken_model_name(),
                },
                index_metrics_path,
            )

        if not args.skip_generation:
            for query in progress_iter(args, queries, desc=f"{LIGHTRAG_MODEL_NAME} queries", unit="query"):
                print(f"[{LIGHTRAG_MODEL_NAME}] query {query.uid}: {query.question}")
                param = query_param_cls(
                    mode="hybrid",
                    response_type=args.response_type,
                    top_k=args.top_k,
                    chunk_top_k=args.top_k,
                    max_total_tokens=args.max_context_tokens,
                )
                start = time.perf_counter()
                full_response: dict[str, Any] | None = None
                if hasattr(rag, "aquery_llm"):
                    full_response = await rag.aquery_llm(query.question, param=param)
                    llm_response = full_response.get("llm_response", {}) if isinstance(full_response, dict) else {}
                    response_iterator = llm_response.get("response_iterator")
                    if llm_response.get("is_streaming") and response_iterator is not None:
                        answer = "".join([part async for part in response_iterator])
                    else:
                        answer = str(llm_response.get("content") or "")
                else:
                    raw_response = await rag.aquery(query.question, param=param)
                    answer = "".join([part async for part in raw_response]) if hasattr(raw_response, "__aiter__") else str(raw_response)
                query_seconds = time.perf_counter() - start
                structured_context = {key: value for key, value in (full_response or {}).items() if key != "llm_response"}
                metrics = _lightrag_query_metrics_from_data(
                    structured_context or None,
                    query_id=query.uid,
                    question=query.question,
                    query_time_seconds=query_seconds,
                )
                context_counts = metrics.get("lightrag_context_counts") or {}
                result = {
                    "model": LIGHTRAG_MODEL_NAME,
                    "lightrag_version": lightrag_version,
                    "query": asdict(query),
                    "answer": answer,
                    "retrieval": {
                        "mode": "hybrid",
                        "retrieved_count_available": bool(metrics.get("retrieval_counts_available")),
                        "counts": context_counts,
                        "raw_data": structured_context,
                    },
                    "context": {
                        "available": metrics.get("evidence_context_chars") is not None,
                        "chars": metrics.get("evidence_context_chars"),
                        "metric_source": metrics.get("metric_source"),
                        "raw_data": structured_context,
                    },
                    "metrics": metrics,
                }
                generated[query.uid] = answer
                all_results[query.uid] = result
                result_metrics.append(metrics)
                write_text(answer, answer_path(output_dir, query, flat=flat))
                write_json(result, result_path(output_dir, query, flat=flat))
    finally:
        await rag.finalize_storages()

    if all_results:
        write_json(all_results, output_dir / "all_query_results.json")
    generation_metrics = aggregate_generation_metrics(
        LIGHTRAG_MODEL_NAME,
        result_metrics,
        index_build_seconds=index_build_seconds,
        index_size_mb=_legacy_index_size_mb_for_model(
            LIGHTRAG_MODEL_NAME,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
        index_size_metrics=compute_query_index_size_metrics(
            LIGHTRAG_MODEL_NAME,
            output_root,
            workdir,
            evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
    )
    if _model_uses_evique_base_dependency(
        LIGHTRAG_MODEL_NAME,
        workdir=workdir,
        evique_workdir=evique_workdir,
        evique_base_dir=evique_base_dir,
    ):
        generation_metrics.update(_evique_base_metrics(
            base_mode=args.evique_base_mode,
            base_dir=evique_base_dir if args.evique_base_mode == "standalone" else workdir,
            base_result=None,
        ))
    generation_metrics.update(
        {
            "lightrag_version": lightrag_version,
            "model_version": f"LightRAG-{lightrag_version}",
            "query_mode": "hybrid",
            "lightrag_workdir": str(lightrag_workdir),
            "answers_dir": str(output_dir),
            "input_source": input_bundle["source_name"],
            "input_source_path": input_bundle["source_path"],
            "input_source_kind": input_bundle["source_kind"],
            "document_count": input_bundle["document_count"],
            "retrieved_items_count": generation_metrics.get("avg_retrieved_count"),
            "used_items_count": generation_metrics.get("avg_used_count"),
            "query_metrics": result_metrics,
            "llm_model": args.model,
            "embedding_model": args.embedding_model,
            "embedding_dim": embedding_dim,
            "tiktoken_model_name": resolve_lightrag_tiktoken_model_name(),
        }
    )
    write_json(generation_metrics, output_dir / "generation_metrics.json")
    write_json(
        {
            "model": LIGHTRAG_MODEL_NAME,
            "lightrag_version": lightrag_version,
            "workdir": str(lightrag_workdir),
            "answers_dir": str(output_dir),
            "query_mode": "hybrid",
            "skip_index": args.skip_index,
            "skip_generation": args.skip_generation,
            "input_source": input_bundle["source_name"],
            "input_source_path": input_bundle["source_path"],
            "input_source_kind": input_bundle["source_kind"],
            "document_count": input_bundle["document_count"],
            "index_build_time_seconds": index_build_seconds,
            "llm_model": args.model,
            "embedding_model": args.embedding_model,
            "embedding_dim": embedding_dim,
            "tiktoken_model_name": resolve_lightrag_tiktoken_model_name(),
        },
        output_dir / "run_config.json",
    )
    return generated


def selected_graphrag_models(selected_models: list[str]) -> list[str]:
    return [model for model in selected_models if model in GRAPHRAG_MODEL_NAMES]


def graphrag_method_for_model(model_name: str) -> str:
    if model_name == GRAPHRAG_LOCAL_MODEL_NAME:
        return "local"
    if model_name == GRAPHRAG_GLOBAL_MODEL_NAME:
        return "global"
    raise ValueError(model_name)


def graphrag_output_dir(graphrag_workdir: Path) -> Path:
    return graphrag_workdir / "output"


def _graphrag_record_to_row(
    record: dict[str, Any],
    *,
    source_name: str,
    source_kind: str,
    fallback_video_id: str | None,
    fallback_record_id: str,
    row_index: int,
    used_ids: set[str],
) -> dict[str, str] | None:
    record_id, text = _lightrag_record_to_document(
        record,
        source_kind=source_kind,
        fallback_video_id=fallback_video_id,
        fallback_record_id=fallback_record_id or str(row_index),
    )
    if not text:
        return None
    video_id = _first_lightrag_field(
        record,
        ["video_id", "video_name", "source_vid", "source_video", "video_path", "collection_id"],
    ) or fallback_video_id or ""
    segment_id = _first_lightrag_field(record, ["segment_id", "video_segment_id", "segment", "segment_index"])
    chunk_id = _first_lightrag_field(record, ["chunk_id", "id", "full_doc_id", "chunk_order_index"])
    start_time = _first_lightrag_field(record, ["start_time", "start", "start_sec", "begin_time"])
    end_time = _first_lightrag_field(record, ["end_time", "end", "end_sec", "finish_time"])
    row_id = _make_unique_lightrag_doc_id(source_name, record_id or str(row_index), used_ids)
    title_bits = [bit for bit in [_stringify_lightrag_value(video_id), _stringify_lightrag_value(segment_id or chunk_id)] if bit]
    return {
        "id": row_id,
        "title": " ".join(title_bits) or row_id,
        "text": text,
        "video_id": _stringify_lightrag_value(video_id),
        "segment_id": _stringify_lightrag_value(segment_id or chunk_id or record_id),
        "start_time": _stringify_lightrag_value(start_time),
        "end_time": _stringify_lightrag_value(end_time),
    }


def _convert_graphrag_rows(
    data: Any,
    *,
    source_name: str,
    source_path: Path,
    source_kind: str,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for index, (video_id, fallback_id, record) in enumerate(_iter_lightrag_records(data)):
        row = _graphrag_record_to_row(
            record,
            source_name=source_name,
            source_kind=source_kind,
            fallback_video_id=video_id,
            fallback_record_id=fallback_id or str(index),
            row_index=index,
            used_ids=used_ids,
        )
        if row:
            rows.append(row)
    return {
        "rows": rows,
        "source_name": source_name,
        "source_path": str(source_path),
        "source_kind": source_kind,
        "document_count": len(rows),
    }


def load_graphrag_input_rows(
    *,
    output_root: Path,
    workdir: Path,
    evique_base_dir: Path,
    required: bool,
) -> dict[str, Any]:
    candidates: list[tuple[str, Path, str]] = [
        ("evique_video_segments", evique_base_dir / "evique_video_segments.json", "segments"),
        ("evique_text_chunks", evique_base_dir / "evique_text_chunks.json", "chunks"),
        ("evique_compat_video_segments", evique_base_dir / "kv_store_video_segments.json", "segments"),
        ("evique_compat_text_chunks", evique_base_dir / "kv_store_text_chunks.json", "chunks"),
        ("shared_video_segments", workdir / "kv_store_video_segments.json", "segments"),
        ("shared_text_chunks", workdir / "kv_store_text_chunks.json", "chunks"),
    ]
    generic_names = [
        "graphrag_input.jsonl",
        "graphrag_input.json",
        "lightrag_input.jsonl",
        "lightrag_input.json",
        "shared_text_input.jsonl",
        "shared_text_input.json",
        "textual_input.jsonl",
        "textual_input.json",
        "input.jsonl",
        "input.json",
        "input.txt",
    ]
    candidates.extend((f"generic_{path.name}", path, "chunks") for base in (output_root, workdir) for path in (base / name for name in generic_names))
    checked: list[str] = []
    for source_name, path, source_kind in candidates:
        checked.append(str(path))
        if not path.exists():
            continue
        try:
            data = _load_generic_lightrag_input(path) if source_name.startswith("generic_") else read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[GraphRAG] skipping unreadable input {path}: {exc}")
            continue
        bundle = _convert_graphrag_rows(
            data,
            source_name=source_name,
            source_path=path,
            source_kind=source_kind,
        )
        if bundle["rows"]:
            bundle["checked_paths"] = checked
            return bundle
    if required:
        raise SystemExit(f"Cannot find usable GraphRAG text input. Checked: {checked}")
    return {
        "rows": [],
        "source_name": "existing_graphrag_workdir",
        "source_path": "",
        "source_kind": "existing",
        "document_count": 0,
        "checked_paths": checked,
    }


def write_graphrag_input_csv(input_bundle: dict[str, Any], graphrag_workdir: Path) -> Path:
    input_dir = graphrag_workdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / "video_segments.csv"
    rows = input_bundle.get("rows") or []
    if not rows:
        raise SystemExit("GraphRAG input conversion produced no rows.")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "text", "video_id", "segment_id", "start_time", "end_time"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def configure_graphrag_settings_for_csv(
    settings_path: Path,
    *,
    model_name: str | None = None,
    embedding_model: str | None = None,
    env: dict[str, str] | None = None,
    embedding_type: str | None = GRAPHRAG_OPENAI_SDK_EMBEDDING_TYPE,
) -> None:
    if not settings_path.exists():
        return
    text = settings_path.read_text(encoding="utf-8")
    csv_input_section = """input:\n  type: csv # [csv, text, json, jsonl]\n  file_pattern: \".*[.]csv\"\n  id_column: id\n  title_column: title\n  text_column: text\n"""
    updated = re.sub(
        r"input:\n(?:  .*\n)*?\nchunking:",
        csv_input_section + "\nchunking:",
        text,
        count=1,
    )
    if updated == text:
        updated = text.replace("input:\n", csv_input_section, 1)

    graph_env = env or _graphrag_env(model_name=model_name, embedding_model=embedding_model)
    completion_provider, completion_model = _resolve_graphrag_completion_settings_fields(model_name)
    embedding_provider, embedding_model_for_settings = _resolve_graphrag_embedding_settings_fields(embedding_model, graph_env)
    has_api_base = bool(graph_env.get("GRAPHRAG_API_BASE") or graph_env.get("GRAPHRAG_BASE_URL") or graph_env.get("OPENAI_BASE_URL"))

    updated = _configure_graphrag_model_block(
        updated,
        section_name="completion_models",
        model_id="default_completion_model",
        model_provider=completion_provider,
        model_name=completion_model,
        api_base_placeholder=has_api_base,
        remove_unsupported_fields=True,
    )
    updated = _configure_graphrag_model_block(
        updated,
        section_name="completion_models",
        model_id="default_chat_model",
        model_provider=completion_provider,
        model_name=completion_model,
        api_base_placeholder=has_api_base,
        remove_unsupported_fields=True,
    )
    updated = _configure_graphrag_model_block(
        updated,
        section_name="embedding_models",
        model_id="default_embedding_model",
        model_provider=embedding_provider,
        model_name=embedding_model_for_settings,
        api_base_placeholder=has_api_base,
        remove_unsupported_fields=True,
    )
    if embedding_type:
        lines = updated.splitlines()
        block = _find_yaml_model_block(lines, "embedding_models", "default_embedding_model")
        if block is not None:
            start, end = block
            _set_yaml_model_field(lines, start, end, "type", embedding_type, after_field="model")
            updated = "\n".join(lines) + ("\n" if updated.endswith("\n") else "")

    if updated != text:
        settings_path.write_text(updated, encoding="utf-8")


def run_graphrag_command(command_args: list[str], *, stage: str, cwd: Path, timeout_seconds: int | None = None, model_name: str | None = None, embedding_model: str | None = None, adapter_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = _graphrag_python_command() + command_args
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=_graphrag_env(model_name=model_name, embedding_model=embedding_model, adapter_dir=adapter_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "GraphRAG CLI is unavailable. Install it with: "
            "python -m pip install -e ./RAG_Baselines/GraphRAG/packages/graphrag"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"GraphRAG {stage} command timed out: {' '.join(command)}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or "(no output)"
        raise SystemExit(
            f"GraphRAG {stage} command failed with exit code {proc.returncode}: {' '.join(command)}\n{detail}"
        )
    return proc


def build_graphrag_index(
    args: argparse.Namespace,
    *,
    output_root: Path,
    workdir: Path,
    evique_base_dir: Path,
    graphrag_workdir: Path,
) -> tuple[Path, float, dict[str, Any]]:
    graphrag_workdir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ensure_graphrag_openai_sdk_embedding_adapter(graphrag_workdir)
    output_dir = graphrag_output_dir(graphrag_workdir)
    settings_path = graphrag_workdir / "settings.yaml"
    metrics_path = graphrag_workdir / "index_build_metrics.json"
    model_name = os.getenv("GRAPHRAG_LLM_MODEL") or args.model or os.getenv("OPENAI_MODEL")
    embedding_model = os.getenv("GRAPHRAG_EMBEDDING_MODEL") or args.embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL")
    graph_env = _graphrag_env(model_name=model_name, embedding_model=embedding_model)
    if args.skip_index:
        if settings_path.exists():
            configure_graphrag_settings_for_csv(settings_path, model_name=model_name, embedding_model=embedding_model, env=graph_env)
            validate_graphrag_settings_template(settings_path, graph_env)
            print_graphrag_settings_summary(settings_path)
        if not output_dir.exists():
            raise SystemExit(
                f"GraphRAG --skip-index requested but index output is missing: {output_dir}. "
                "Run once without --skip-index to build the shared GraphRAG index."
            )
        existing = read_json(metrics_path) if metrics_path.exists() else {}
        return output_dir, float(existing.get("index_build_time_seconds", 0.0)), existing

    input_bundle = load_graphrag_input_rows(
        output_root=output_root,
        workdir=workdir,
        evique_base_dir=evique_base_dir,
        required=True,
    )
    input_csv = write_graphrag_input_csv(input_bundle, graphrag_workdir)
    if not settings_path.exists():
        run_graphrag_command(
            ["init", "--root", str(graphrag_workdir), "--force", "--model", model_name, "--embedding", embedding_model],
            stage="init",
            cwd=PROJECT_ROOT,
            model_name=model_name,
            embedding_model=embedding_model,
            adapter_dir=adapter_dir,
        )
    configure_graphrag_settings_for_csv(settings_path, model_name=model_name, embedding_model=embedding_model, env=graph_env)
    validate_graphrag_settings_template(settings_path, graph_env)
    print_graphrag_settings_summary(settings_path)
    start = progress_stage_start("[GraphRAG] indexing")
    run_graphrag_command(
        ["index", "--root", str(graphrag_workdir), "--method", "standard"],
        stage="index",
        cwd=PROJECT_ROOT,
        model_name=model_name,
        embedding_model=embedding_model,
        adapter_dir=adapter_dir,
    )
    elapsed = time.perf_counter() - start
    progress_stage_finish("[GraphRAG] indexing", start)
    if not output_dir.exists():
        raise SystemExit(f"GraphRAG index completed but output directory was not found: {output_dir}")
    metrics = {
        "model": "GraphRAG",
        "graphrag_version": graphrag_version_from_pyproject(),
        "index_build_time_seconds": elapsed,
        "workdir": str(graphrag_workdir),
        "output_dir": str(output_dir),
        "input_csv": str(input_csv),
        "input_source": input_bundle["source_name"],
        "input_source_path": input_bundle["source_path"],
        "input_source_kind": input_bundle["source_kind"],
        "document_count": input_bundle["document_count"],
        "llm_model": model_name,
        "embedding_model": embedding_model,
    }
    write_json(metrics, metrics_path)
    return output_dir, elapsed, metrics


def run_graphrag_answers(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    evique_base_dir: Path,
    graphrag_workdir: Path,
    output_dir: Path,
    model_name: str,
    graph_method: str,
    index_build_seconds: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ensure_graphrag_openai_sdk_embedding_adapter(graphrag_workdir)
    flat = len({q.collection_id for q in queries}) <= 1
    generated: dict[str, str] = {}
    all_results: dict[str, Any] = {}
    result_metrics: list[dict[str, Any]] = []
    index_output_dir = graphrag_output_dir(graphrag_workdir)
    llm_model_name = os.getenv("GRAPHRAG_LLM_MODEL") or args.model or os.getenv("OPENAI_MODEL")
    embedding_model = os.getenv("GRAPHRAG_EMBEDDING_MODEL") or args.embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL")
    graph_env = _graphrag_env(model_name=llm_model_name, embedding_model=embedding_model)
    settings_path = graphrag_workdir / "settings.yaml"
    if settings_path.exists():
        configure_graphrag_settings_for_csv(settings_path, model_name=llm_model_name, embedding_model=embedding_model, env=graph_env)
        validate_graphrag_settings_template(settings_path, graph_env)
        print_graphrag_settings_summary(settings_path)
    for query in progress_iter(args, queries, desc=f"{model_name} {graph_method} queries", unit="query"):
        print(f"[{model_name}] query {query.uid}: {query.question}")
        start = time.perf_counter()
        proc = run_graphrag_command(
            [
                "query",
                query.question,
                "--root",
                str(graphrag_workdir),
                "--data",
                str(index_output_dir),
                "--method",
                graph_method,
                "--response-type",
                args.response_type,
            ],
            stage=f"query {graph_method}",
            cwd=PROJECT_ROOT,
            model_name=llm_model_name,
            embedding_model=embedding_model,
            adapter_dir=adapter_dir,
        )
        query_seconds = time.perf_counter() - start
        answer = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        metrics = _graphrag_query_metrics_from_answer(
            answer,
            query_id=query.uid,
            question=query.question,
            query_time_seconds=query_seconds,
        )
        result = {
            "model": model_name,
            "graphrag_version": graphrag_version_from_pyproject(),
            "graph_rag_method": graph_method,
            "query": asdict(query),
            "answer": answer,
            "retrieval": {
                "method": graph_method,
                "retrieved_count_available": bool(metrics.get("retrieval_counts_available")),
                "citation_counts": metrics.get("citation_counts"),
            },
            "context": {
                "available": metrics.get("evidence_context_chars") is not None,
                "chars": metrics.get("evidence_context_chars"),
                "metric_source": metrics.get("metric_source"),
            },
            "metrics": metrics,
        }
        generated[query.uid] = answer
        all_results[query.uid] = result
        result_metrics.append(metrics)
        write_text(answer, answer_path(output_dir, query, flat=flat))
        write_json(result, result_path(output_dir, query, flat=flat))

    write_json(all_results, output_dir / "all_query_results.json")
    generation_metrics = aggregate_generation_metrics(
        model_name,
        result_metrics,
        index_build_seconds=index_build_seconds,
        index_size_mb=_legacy_index_size_mb_for_model(
            model_name,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
        index_size_metrics=compute_query_index_size_metrics(
            model_name,
            output_root,
            workdir,
            evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=args.evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
    )
    if _model_uses_evique_base_dependency(
        model_name,
        workdir=workdir,
        evique_workdir=evique_workdir,
        evique_base_dir=evique_base_dir,
    ):
        generation_metrics.update(_evique_base_metrics(
            base_mode=args.evique_base_mode,
            base_dir=evique_base_dir if args.evique_base_mode == "standalone" else workdir,
            base_result=None,
        ))
    generation_metrics.update(
        {
            "model_name": model_name,
            "graphrag_version": graphrag_version_from_pyproject(),
            "graph_rag_method": graph_method,
            "graphrag_workdir": str(graphrag_workdir),
            "answers_dir": str(output_dir),
            "retrieved_items_count": generation_metrics.get("avg_retrieved_count"),
            "used_items_count": generation_metrics.get("avg_used_count"),
            "query_metrics": result_metrics,
        }
    )
    write_json(generation_metrics, output_dir / "generation_metrics.json")
    write_json(
        {
            "model": model_name,
            "graphrag_version": graphrag_version_from_pyproject(),
            "graph_rag_method": graph_method,
            "workdir": str(graphrag_workdir),
            "output_dir": str(index_output_dir),
            "answers_dir": str(output_dir),
            "index_build_time_seconds": index_build_seconds,
            "query_command_method": graph_method,
        },
        output_dir / "run_config.json",
    )
    return generated

def run_paper_evaluation(
    args: argparse.Namespace,
    queries: list[QueryRecord],
    *,
    output_root: Path,
    selected_models: list[str],
    evique_dir: Path,
    videorag_dir: Path,
    naiverag_dir: Path,
    textvideorag_dir: Path,
    lightrag_dir: Path,
    graphrag_l_dir: Path,
    graphrag_g_dir: Path,
) -> tuple[Path | None, Path | None, dict[str, float]]:
    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_only = getattr(args, "eval_only", "all")
    print(f"[evaluation] eval_only={eval_only}")
    eval_args = argparse.Namespace(
        evique_answers=str(evique_dir) if EVIQUE_MODEL_NAME in selected_models else None,
        videorag_answers=str(videorag_dir) if VIDEO_RAG_MODEL_NAME in selected_models else None,
        naiverag_answers=str(naiverag_dir) if NAIVE_MODEL_NAME in selected_models else None,
        textvideorag_answers=str(textvideorag_dir) if TEXT_VIDEO_MODEL_NAME in selected_models else None,
    )
    sources = collect_answer_sources(eval_args, queries, {}, current_model_name="none")
    if LIGHTRAG_MODEL_NAME in selected_models:
        lightrag_answers: dict[str, str] = {}
        for query in queries:
            answer = read_answer_from_dir(lightrag_dir, query)
            if answer is not None:
                lightrag_answers[query.uid] = answer
        if lightrag_answers:
            sources[LIGHTRAG_MODEL_NAME] = lightrag_answers
    for graph_model_name, graph_dir in (
        (GRAPHRAG_LOCAL_MODEL_NAME, graphrag_l_dir),
        (GRAPHRAG_GLOBAL_MODEL_NAME, graphrag_g_dir),
    ):
        if graph_model_name not in selected_models:
            continue
        graph_answers: dict[str, str] = {}
        for query in queries:
            answer = read_answer_from_dir(graph_dir, query)
            if answer is not None:
                graph_answers[query.uid] = answer
        if graph_answers:
            sources[graph_model_name] = graph_answers
    if len(sources) < 2:
        raise SystemExit("Need at least two answer sources for evaluation.")
    winrate_reference_model = None
    if eval_only in {"all", "winrate"}:
        winrate_reference_model = resolve_winrate_reference_model(args)
        if winrate_reference_model and winrate_reference_model not in selected_models:
            selected_text = ", ".join(selected_models) if selected_models else "(none)"
            raise SystemExit(
                f"Win-rate reference model {winrate_reference_model!r} is not in selected models: {selected_text}"
            )
    client = make_openai_client()
    winrate_path: Path | None = eval_dir / "winrate_table.csv"
    quantitative_path: Path | None = eval_dir / "quantitative_table.csv"
    win_rows: list[dict[str, Any]] = []
    quant_rows: list[dict[str, Any]] = []

    if eval_only in {"all", "winrate"}:
        winrate_stage = progress_stage_start("[evaluation] win-rate")
        win_rows = run_winrate_eval(
            client,
            llm_model=args.judge_model,
            queries=queries,
            answer_sources=sources,
            output_dir=output_root,
            bidirectional=not args.single_pass_winrate,
            eval_runs=args.eval_runs,
            anchor_winrate_model=winrate_reference_model,
        )
        progress_stage_finish("[evaluation] win-rate", winrate_stage)
        print()
        print("Win-Rate Comparison")
        print(markdown_table(win_rows))
    else:
        print(f"[evaluation] skip winrate evaluation because eval_only={eval_only}")
        if not winrate_path.exists():
            winrate_path = None

    if eval_only in {"all", "quantitative"}:
        quant_stage = progress_stage_start("[evaluation] quantitative")
        quant_rows = run_quantitative_eval(
            client,
            llm_model=args.judge_model,
            queries=queries,
            answer_sources=sources,
            output_dir=output_root,
            baseline_model=args.quant_baseline,
            eval_runs=args.eval_runs,
        )
        progress_stage_finish("[evaluation] quantitative", quant_stage)
        print()
        print("Quantitative Comparison")
        print(markdown_table(quant_rows))
    else:
        print(f"[evaluation] skip quantitative evaluation because eval_only={eval_only}")
        if quantitative_path.exists():
            try:
                quant_rows = list(_load_quantitative_score_rows(quantitative_path).values())
            except (OSError, csv.Error):
                quant_rows = []
        else:
            quantitative_path = None

    quant_score_by_model: dict[str, float] = {}
    for row in quant_rows:
        try:
            quant_score_by_model[str(row["Model"])] = float(row.get("Overall Score", 0.0))
        except (TypeError, ValueError):
            quant_score_by_model[str(row.get("Model", ""))] = 0.0
    return winrate_path, quantitative_path, quant_score_by_model


def _query_text_from_result(result: dict[str, Any]) -> tuple[str | None, str]:
    query_data = result.get("query") if isinstance(result.get("query"), dict) else {}
    query_id = str(query_data.get("uid") or query_data.get("id") or query_data.get("query_id") or "") or None
    question = str(query_data.get("question") or query_data.get("query") or "")
    return query_id, question


def _rederive_result_metrics_from_all_results(model_name: str, all_results: dict[str, Any]) -> list[dict[str, Any]]:
    result_metrics: list[dict[str, Any]] = []
    for fallback_uid, result_value in all_results.items():
        if not isinstance(result_value, dict):
            continue
        old_metrics = result_value.get("metrics") if isinstance(result_value.get("metrics"), dict) else {}
        query_id, question = _query_text_from_result(result_value)
        query_id = query_id or str(fallback_uid)
        query_seconds = _float_or_none(old_metrics.get("query_time_seconds"))
        if model_name == LIGHTRAG_MODEL_NAME:
            raw_data = None
            for section_name in ("context", "retrieval"):
                section = result_value.get(section_name)
                if isinstance(section, dict) and isinstance(section.get("raw_data"), dict):
                    raw_data = section.get("raw_data")
                    break
            has_nonzero_metric = any((_float_or_none(old_metrics.get(key)) or 0.0) > 0 for key in ("retrieved_count", "used_count", "evidence_chars"))
            if raw_data is None and old_metrics.get("metric_source") and has_nonzero_metric:
                result_metrics.append(_normalised_metric_aliases(old_metrics, query_id=query_id))
            else:
                result_metrics.append(_lightrag_query_metrics_from_data(raw_data, query_id=query_id, question=question, query_time_seconds=query_seconds))
        elif model_name in GRAPHRAG_MODEL_NAMES:
            result_metrics.append(_graphrag_query_metrics_from_answer(str(result_value.get("answer") or ""), query_id=query_id, question=question, query_time_seconds=query_seconds))
        else:
            result_metrics.append(_normalised_metric_aliases(old_metrics, query_id=query_id))
    return result_metrics


def _legacy_zero_retrieval_metrics(metrics: dict[str, Any]) -> bool:
    if metrics.get("metric_source_counts"):
        return False
    retrieved = _float_or_none(metrics.get("avg_retrieved_count"))
    used = _float_or_none(metrics.get("avg_used_count"))
    evidence = _float_or_none(metrics.get("avg_evidence_chars"))
    return retrieved == 0.0 and used == 0.0 and evidence == 0.0


def _scrub_legacy_unavailable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(metrics)
    for key in ("avg_retrieved_count", "avg_used_count", "avg_support_ratio", "avg_evidence_chars", "avg_llm_input_tokens_estimate", "retrieved_items_count", "used_items_count"):
        repaired[key] = None
    repaired["metric_source_counts"] = {"unavailable": 1}
    repaired.setdefault("metric_unavailable_reason", "Legacy run recorded unavailable retrieval metrics as zero.")
    return repaired


def repair_generation_metrics_for_summary(metrics: dict[str, Any], output_dir: Path, *, model_name: str) -> dict[str, Any]:
    if model_name not in {LIGHTRAG_MODEL_NAME, *GRAPHRAG_MODEL_NAMES}:
        return metrics
    all_results_path = output_dir / "all_query_results.json"
    if all_results_path.exists():
        try:
            all_results = read_json(all_results_path)
        except (OSError, json.JSONDecodeError):
            all_results = {}
        if isinstance(all_results, dict) and all_results:
            result_metrics = _rederive_result_metrics_from_all_results(model_name, all_results)
            if result_metrics:
                aggregate = aggregate_generation_metrics(model_name, result_metrics, index_build_seconds=_float_or_none(metrics.get("index_build_time_seconds")) or 0.0, index_size_mb=_float_or_none(metrics.get("index_size_mb")) or 0.0)
                repaired = dict(metrics)
                for key in ("avg_query_time_seconds", "avg_evidence_chars", "avg_llm_input_tokens_estimate", "avg_retrieved_count", "avg_used_count", "avg_support_ratio", "metric_source_counts"):
                    repaired[key] = aggregate.get(key)
                repaired["retrieved_items_count"] = repaired.get("avg_retrieved_count")
                repaired["used_items_count"] = repaired.get("avg_used_count")
                repaired["query_metrics"] = result_metrics
                return repaired
    if _legacy_zero_retrieval_metrics(metrics):
        return _scrub_legacy_unavailable_metrics(metrics)
    return metrics


def load_generation_metrics(
    output_dir: Path,
    *,
    model_name: str,
    output_root: Path,
    workdir: Path,
    evique_workdir: Path,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
    evique_base_metrics: dict[str, Any] | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    path = output_dir / "generation_metrics.json"
    if path.exists():
        metrics = enrich_generation_metrics_with_index_sizes(
            read_json(path),
            model_name,
            output_root=output_root,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
            evique_base_metrics=evique_base_metrics,
        )
        metrics = repair_generation_metrics_for_summary(metrics, output_dir, model_name=model_name)
        if persist:
            write_json(metrics, path)
        return metrics
    all_results_path = output_dir / "all_query_results.json"
    if not all_results_path.exists():
        metrics = enrich_generation_metrics_with_index_sizes(
            {"model": model_name, "index_size_mb": 0.0},
            model_name,
            output_root=output_root,
            workdir=workdir,
            evique_workdir=evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
            evique_base_metrics=evique_base_metrics,
        )
        if persist:
            write_json(metrics, path)
        return metrics
    all_results = read_json(all_results_path)
    result_metrics = _rederive_result_metrics_from_all_results(model_name, all_results) if isinstance(all_results, dict) else []
    metrics = aggregate_generation_metrics(
        str(next(iter(all_results.values())).get("model", output_dir.name)) if all_results else output_dir.name,
        result_metrics,
        index_build_seconds=0.0,
        index_size_mb=directory_size_bytes(output_dir) / (1024 * 1024),
        index_size_metrics=compute_query_index_size_metrics(
            model_name,
            output_root,
            workdir,
            evique_workdir,
            model_answer_dir=output_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
        ),
    )
    metrics["legacy_index_size_mb"] = _float_or_zero(metrics.get("index_size_mb"))
    if model_name == EVIQUE_MODEL_NAME:
        for key, value in (evique_base_metrics or {}).items():
            metrics.setdefault(key, value)
    if persist:
        write_json(metrics, path)
    return metrics


def _path_for_summary(path: Path, *, output_root: Path) -> str:
    path = Path(path)
    for base in (output_root, REPO_ROOT):
        try:
            return str(path.resolve().relative_to(Path(base).resolve())).replace("\\", "/")
        except ValueError:
            continue
        except OSError:
            break
    return str(path).replace("\\", "/")


def _paths_for_summary(paths: list[Path], *, output_root: Path) -> str:
    return "; ".join(_path_for_summary(path, output_root=output_root) for path in safe_existing_paths(paths))


def _find_numeric_key(data: Any, keys: list[str]) -> float | None:
    if not isinstance(data, dict):
        return None
    lowered = {str(key).lower(): value for key, value in data.items()}
    for key in keys:
        value = _float_or_none(lowered.get(key.lower()))
        if value is not None:
            return value
    for value in data.values():
        nested = _find_numeric_key(value, keys)
        if nested is not None:
            return nested
    return None


def _shared_base_index_build_time(workdir: Path) -> tuple[float | None, str]:
    metrics_path = Path(workdir) / "index_build_metrics.json"
    if not metrics_path.exists():
        return None, "历史运行未单独记录共享基础索引构建耗时，无法精确拆分。"
    try:
        metrics = read_json(metrics_path)
    except (OSError, json.JSONDecodeError):
        return None, "共享基础索引构建耗时文件存在但无法读取，无法精确拆分。"
    shared_seconds = _find_numeric_key(metrics, SHARED_BASE_INDEX_BUILD_TIME_KEYS)
    if shared_seconds is None:
        return None, "历史运行未单独记录共享基础索引构建耗时，无法精确拆分；未来运行应在 index_build_metrics.json 中写入 shared_base_index_build_time_seconds。"
    return shared_seconds, "共享基础索引构建时间读取自 videorag-workdir/index_build_metrics.json。"


def _index_build_time_summary(
    model_name: str,
    metrics: dict[str, Any],
    *,
    workdir: Path,
    has_shared_dependency: bool,
) -> tuple[float | None, float | None, float | None, str]:
    total_or_method_seconds = _float_or_none(metrics.get("index_build_time_seconds"))
    shared_seconds, shared_note = _shared_base_index_build_time(workdir)
    if not has_shared_dependency:
        note = "该模型未检测到共享基础索引文件；端到端索引构建时间等于方法专属索引构建时间。"
        return total_or_method_seconds, 0.0, total_or_method_seconds, note

    if shared_seconds is None:
        if model_name == VIDEO_RAG_MODEL_NAME:
            method_seconds = None
        else:
            method_seconds = total_or_method_seconds
        note = (
            "方法专属索引构建时间来自该模型 generation_metrics.json；"
            f"{shared_note}端到端索引构建时间未输出，以避免把未记录的共享基础索引耗时伪造成精确值。"
        )
        return method_seconds, None, None, note

    if model_name == VIDEO_RAG_MODEL_NAME:
        method_seconds = (
            max(0.0, total_or_method_seconds - shared_seconds)
            if total_or_method_seconds is not None
            else None
        )
    else:
        method_seconds = total_or_method_seconds
    end_to_end_seconds = (
        method_seconds + shared_seconds
        if method_seconds is not None
        else None
    )
    note = (
        "方法专属索引构建时间统计该模型额外构建的查询索引耗时；"
        "共享基础索引构建时间统计 VideoRAG 基础 video_path/video_segments/text_chunks 依赖耗时；"
        "端到端索引构建时间为二者之和。"
    )
    return method_seconds, shared_seconds, end_to_end_seconds, note


def _index_size_note() -> str:
    return (
        "方法专属索引大小只统计该模型查询阶段需要的专属索引文件，不包含原始视频、模型权重、答案文件、评估结果和 LLM cache；"
        "端到端查询索引大小额外包含共享基础 video/text segment 依赖。"
    )


def _load_quantitative_score_rows(accuracy_file: Path | None) -> dict[str, dict[str, str]]:
    if accuracy_file is None or not accuracy_file.exists():
        return {}
    with accuracy_file.open("r", encoding="utf-8", newline="") as f:
        return {str(row.get("Model", "")): row for row in csv.DictReader(f)}


def _score_value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return f"{_float_or_zero(value):.2f}" if value not in (None, "") else ""


def _workdir_for_summary(
    model_name: str,
    *,
    workdir: Path,
    evique_workdir: Path,
    model_dir: Path,
) -> Path:
    if model_name == EVIQUE_MODEL_NAME:
        return evique_workdir
    if model_name == VIDEO_RAG_MODEL_NAME:
        return workdir
    if model_name == LIGHTRAG_MODEL_NAME:
        return model_dir.parent / "lightrag-workdir"
    if model_name in GRAPHRAG_MODEL_NAMES:
        return model_dir.parent / "graphrag-workdir"
    return model_dir


def summary_columns() -> list[str]:
    return list(SUMMARY_COLUMNS)


def _reliable_evique_base_build_time(
    *,
    evique_base_dir: Path | None,
    evique_base_metrics: dict[str, Any] | None,
    metrics: dict[str, Any] | None = None,
) -> float | None:
    for source in (metrics or {}, evique_base_metrics or {}):
        for key in ("evique_base_build_time_seconds", "base_build_time_seconds", "build_time_seconds"):
            value = _float_or_none(source.get(key)) if isinstance(source, dict) else None
            if value is not None and value > 0:
                return value
    if evique_base_dir is None:
        return None
    manifest_path = Path(evique_base_dir) / "evique_base_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("base_build_time_seconds", "build_time_seconds"):
        value = _float_or_none(manifest.get(key))
        if value is not None and value > 0:
            return value
    return None


def _dataset_base_build_time_for_summary(
    model_name: str,
    metrics: dict[str, Any],
    *,
    workdir: Path,
    evique_base_mode: str,
    base_paths: list[Path],
    evique_base_dir: Path | None,
    evique_base_metrics: dict[str, Any] | None,
) -> float | None:
    reliable_evique_seconds = _reliable_evique_base_build_time(
        evique_base_dir=evique_base_dir,
        evique_base_metrics=evique_base_metrics,
        metrics=metrics,
    )
    if evique_base_mode == "standalone" and base_paths:
        return reliable_evique_seconds
    if model_name == EVIQUE_MODEL_NAME and evique_base_mode == "standalone":
        return reliable_evique_seconds
    if base_paths:
        seconds, _ = _shared_base_index_build_time(workdir)
        return seconds if seconds is not None and seconds > 0 else None
    return None


def _query_temp_index_size_mb(metrics: dict[str, Any]) -> float:
    return _float_or_zero(
        metrics.get("avg_query_temp_index_size_mb")
        or metrics.get("avg_query_time_index_size_mb")
        or metrics.get("query_time_index_size_mb")
        or 0.0
    )


def _query_temp_index_seconds(metrics: dict[str, Any]) -> float:
    return _float_or_zero(
        metrics.get("avg_query_temp_index_seconds")
        or metrics.get("avg_query_time_index_seconds")
        or metrics.get("query_time_index_seconds")
        or 0.0
    )


def _accuracy_score_for_summary(
    model_name: str,
    quant_row: dict[str, str],
    accuracy_scores: dict[str, float],
) -> str:
    overall_score = _score_value(quant_row, "Overall Score")
    if overall_score:
        return overall_score
    if model_name in accuracy_scores:
        return f"{float(accuracy_scores[model_name]):.2f}"
    return "N/A"


def _accuracy_file_for_summary(accuracy_file: Path | None) -> str:
    return str(accuracy_file) if accuracy_file is not None and accuracy_file.exists() else "N/A"


def write_summary(
    *,
    output_root: Path,
    dataset_name: str,
    video_meta: dict[str, float],
    model_dirs: dict[str, Path],
    workdir: Path,
    evique_workdir: Path,
    evique_base_mode: str = "shared",
    evique_base_dir: Path | None = None,
    evique_base_metrics: dict[str, Any] | None = None,
    accuracy_scores: dict[str, float],
    accuracy_file: Path | None,
) -> Path:
    rows: list[dict[str, Any]] = []
    base_build_time_values: list[float | None] = []
    quantitative_rows = _load_quantitative_score_rows(accuracy_file)
    for model_name, model_dir in model_dirs.items():
        metrics = load_generation_metrics(
            model_dir,
            model_name=model_name,
            output_root=output_root,
            workdir=workdir,
            evique_workdir=evique_workdir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
            evique_base_metrics=evique_base_metrics,
            persist=True,
        )
        retrieved = _float_or_none(metrics.get("avg_retrieved_count"))
        used = _float_or_none(metrics.get("avg_used_count"))
        ratio = _float_or_none(metrics.get("avg_support_ratio"))
        if ratio is None and retrieved is not None and used is not None:
            ratio = (used / retrieved) if retrieved > 0 else 0.0
        method_paths, shared_paths, _ = _query_index_paths_for_model(
            model_name,
            output_root,
            workdir,
            evique_workdir,
            model_answer_dir=model_dir,
            evique_base_mode=evique_base_mode,
            evique_base_dir=evique_base_dir,
        )
        method_size_mb = file_size_mb(method_paths)
        shared_size_mb = file_size_mb(shared_paths)
        base_build_seconds = _dataset_base_build_time_for_summary(
            model_name,
            metrics,
            workdir=workdir,
            evique_base_mode=evique_base_mode,
            base_paths=shared_paths,
            evique_base_dir=evique_base_dir,
            evique_base_metrics=evique_base_metrics,
        )
        base_build_time_values.append(base_build_seconds)
        quant_row = quantitative_rows.get(model_name, {})
        row = {
            "数据集": dataset_name,
            "视频时长 (min)": f"{float(video_meta.get('duration_min', 0.0)):.2f}",
            "数据集大小 (MB)": f"{float(video_meta.get('size_mb', 0.0)):.2f}",
            "模型名称": model_name,
            "数据集基础索引构建时间 (秒)": _format_float_or_na(base_build_seconds),
            "基础索引大小 (MB)": f"{shared_size_mb:.2f}",
            "方法增量索引大小 (MB)": f"{method_size_mb:.2f}",
            "端到端索引大小 (MB)": f"{method_size_mb + shared_size_mb:.2f}",
            "平均 query 临时索引大小 (MB)": f"{_query_temp_index_size_mb(metrics):.2f}",
            "平均 query 临时索引时间 (秒)": f"{_query_temp_index_seconds(metrics):.2f}",
            "平均查询时间 (秒)": _format_float_or_na(_float_or_none(metrics.get("avg_query_time_seconds"))),
            "平均准确率得分": _accuracy_score_for_summary(model_name, quant_row, accuracy_scores),
            "检索到的片段或项目数": _format_float_or_na(retrieved),
            "使用的支持片段或项目数": _format_float_or_na(used),
            "使用 / 检索": _format_float_or_na(ratio, decimals=4),
            "最终证据包平均大小 (字符)": _format_float_or_na(_float_or_none(metrics.get("avg_evidence_chars"))),
            "LLM 输入 tokens 平均估计值": _format_float_or_na(_float_or_none(metrics.get("avg_llm_input_tokens_estimate"))),
            "答案文件": str(model_dir),
            "准确率文件": _accuracy_file_for_summary(accuracy_file),
        }
        rows.append(row)
    summary_columns = list(SUMMARY_COLUMNS)
    if not any(value is not None for value in base_build_time_values):
        summary_columns = [column for column in summary_columns if column != SUMMARY_COLUMNS[4]]
    rows = [{column: row.get(column, "") for column in summary_columns} for row in rows]
    summary_csv = output_root / "comparison_summary.csv"
    save_csv(rows, summary_csv)
    write_text(markdown_table(rows), output_root / "comparison_summary.md")
    return summary_csv


def main() -> None:
    args = parse_args()
    args._progress_enabled = resolve_progress_enabled(args)
    if args.check_layout:
        print_layout_check(args)
        return
    if args.check_lightrag:
        print_lightrag_check()
        return
    if args.check_graphrag:
        print_graphrag_check()
        return
    if args.check_graphrag_embedding:
        print_graphrag_embedding_smoke(args)
        return

    selected_models = parse_model_selection(args.models)
    unified_evique_base = use_unified_evique_base(args)
    if unified_evique_base:
        args.evique_base_mode = "standalone"
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    video_paths = [Path(value).expanduser() for value in (args.video or [DEFAULT_VIDEO])]
    missing = [str(path) for path in video_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Video file(s) not found: {missing}")

    dataset_name = args.dataset_name or safe_slug("+".join(path.stem for path in video_paths))
    output_root = Path(args.output_root or f"comparison_runs/{dataset_name}_{now_stamp()}").resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else output_root / "videorag-workdir"
    evique_workdir = Path(args.evique_workdir).resolve() if args.evique_workdir else output_root / "evique-workdir"
    lightrag_workdir = output_root / "lightrag-workdir"
    graphrag_workdir = output_root / "graphrag-workdir"
    evique_base_dir = (
        Path(args.evique_base_dir).resolve()
        if args.evique_base_dir
        else evique_workdir / "base"
    )
    graphrag_selected_models = selected_graphrag_models(selected_models)
    video_text_base_models_selected = any(
        model in selected_models
        for model in (VIDEO_RAG_MODEL_NAME, NAIVE_MODEL_NAME, TEXT_VIDEO_MODEL_NAME)
    )
    lightrag_can_use_standalone_base = (
        LIGHTRAG_MODEL_NAME in selected_models
        and args.evique_base_mode == "standalone"
        and (
            unified_evique_base
            or EVIQUE_MODEL_NAME in selected_models
            or _evique_standalone_base_has_lightrag_input(evique_base_dir)
        )
    )
    graphrag_can_use_standalone_base = (
        bool(graphrag_selected_models)
        and args.evique_base_mode == "standalone"
        and (
            unified_evique_base
            or EVIQUE_MODEL_NAME in selected_models
            or _evique_standalone_base_has_lightrag_input(evique_base_dir)
        )
    )
    shared_base_required = (not unified_evique_base) and (
        VIDEO_RAG_MODEL_NAME in selected_models
        or NAIVE_MODEL_NAME in selected_models
        or TEXT_VIDEO_MODEL_NAME in selected_models
        or (EVIQUE_MODEL_NAME in selected_models and args.evique_base_mode == "shared")
        or (LIGHTRAG_MODEL_NAME in selected_models and not lightrag_can_use_standalone_base)
        or (bool(graphrag_selected_models) and not graphrag_can_use_standalone_base)
    )
    videorag_compatible_base_required = unified_evique_base and video_text_base_models_selected
    output_root.mkdir(parents=True, exist_ok=True)
    if shared_base_required or videorag_compatible_base_required:
        workdir.mkdir(parents=True, exist_ok=True)
    evique_workdir.mkdir(parents=True, exist_ok=True)
    if LIGHTRAG_MODEL_NAME in selected_models:
        lightrag_workdir.mkdir(parents=True, exist_ok=True)
    if graphrag_selected_models:
        graphrag_workdir.mkdir(parents=True, exist_ok=True)

    video_meta = load_video_metadata(video_paths)
    evique_base_for_config = evique_base_dir if args.evique_base_mode == "standalone" else workdir
    write_json(
        {
            "dataset_name": dataset_name,
            "evique_version": EVIQUE_VERSION,
            "evique_model_version": EVIQUE_VERSION_LABEL,
            "video_paths": [str(path) for path in video_paths],
            "video_metadata": video_meta,
            "selected_models": selected_models,
            "evique_base_mode": args.evique_base_mode,
            "evique_base_dir": str(evique_base_for_config),
            "unified_base_source": args.unified_base_source or "legacy",
            "videorag_compatible_base_dir": str(workdir),
            "videorag_compatible_base_derived_from_evique": unified_evique_base,
            "visual_compactor": config_to_dict(get_visual_compactor_config()),
            "paper_protocol": {
                "win_rate": "Pairwise LLM judge, bidirectional answer order unless --single-pass-winrate, repeated --eval-runs times.",
                "anchor_winrate_model": resolve_winrate_reference_model(args),
                "winrate_reference_model": resolve_winrate_reference_model(args),
                "quantitative": "1-5 LLM judge scoring against NaiveRAG unless reference answers are supplied, repeated --eval-runs times.",
                "metrics": ["Comprehensiveness", "Empowerment", "Trustworthiness", "Depth", "Density", "Overall Winner/Score"],
                "shared_input": "VideoRAG, NaiveRAG, TextVideoRAG and LightRAG read grounded visual-description + ASR text from shared or standalone segment/chunk stores when those models are selected.",
                "lightrag_input": "LightRAG is adapted in this runner from EVIQUE standalone base when available, otherwise VideoRAG shared base, and queries use QueryParam(mode='hybrid').",
                "graphrag_input": "GraphRAG-l/g are adapted in this runner from the same grounded text base into graphrag-workdir/input/video_segments.csv and share one graphrag-workdir index.",
                "evique_input": (
                    "EVIQUE standalone builds its own base files from raw video under evique-workdir/base before materializing Scope/Target/Track/Event plus Adaptive Event views."
                    if args.evique_base_mode == "standalone"
                    else "EVIQUE reads the same VideoRAG visual-description + ASR segment store and materializes Scope/Target/Track/Event plus Adaptive Event views."
                ),
                "evique_model_version": EVIQUE_VERSION_LABEL,
                "event_segmentation": get_event_segmentation_config(),
                "cost_based_view_planner": get_cost_planner_config(),
                "evidence_packer": get_evidence_packer_config(),
                "shared_chunking": "VideoRAG chunking_by_video_segments with 1200 tokens by default.",
                "frame_count": args.fine_num_frames,
            },
        },
        output_root / "comparison_config.json",
    )

    rag = None
    index_seconds = 0.0
    shared_video_segments: dict[str, dict[str, dict[str, Any]]] | None = None
    model_progress = ModelProgress(args, selected_models)

    if shared_base_required:
        rag, index_seconds = build_videorag_index(args, workdir, video_paths, evique_base_dir=evique_base_dir)
        shared_video_segments = read_video_segments(workdir)
        if not args.allow_non_15_frame_segments:
            validate_frame_counts(shared_video_segments, args.fine_num_frames)
    else:
        print("[shared-base] not required for selected models/base mode; skipping shared workdir build.")

    evique_base_result: dict[str, Any] | None = None
    evique_video_segments: dict[str, dict[str, dict[str, Any]]] | None = None
    evique_video_path_map: dict[str, str] | None = None
    if (EVIQUE_MODEL_NAME in selected_models or unified_evique_base) and args.evique_base_mode == "standalone":
        evique_base_stage = progress_stage_start("[EVIQUE] base build")
        evique_base_result = build_evique_standalone_base(
            video_paths=video_paths,
            output_base_dir=evique_base_dir,
            dataset_name=dataset_name,
            chunk_token_size=args.chunk_token_size,
            fine_num_frames=args.fine_num_frames,
            rough_num_frames=args.rough_num_frames,
            segment_length=args.segment_length,
            rebuild=args.evique_rebuild_base,
            model_name=args.model,
            embedding_model=args.embedding_model,
            embedding_dim=args.videorag_embedding_dim,
        )
        progress_stage_finish("[EVIQUE] base build", evique_base_stage)
        evique_video_segments = evique_base_result["video_segments"]
        evique_video_path_map = evique_base_result["video_path_map"]
        if not args.allow_non_15_frame_segments:
            validate_frame_counts(evique_video_segments, args.fine_num_frames)
        if videorag_compatible_base_required:
            prepare_videorag_compatible_base_from_evique(evique_base_dir, workdir, skip_index=args.skip_index)
            shared_video_segments = read_video_segments(workdir)
            if not args.allow_non_15_frame_segments:
                validate_frame_counts(shared_video_segments, args.fine_num_frames)
        if VIDEO_RAG_MODEL_NAME in selected_models:
            rag, index_seconds = build_videorag_index(args, workdir, video_paths, evique_base_dir=evique_base_dir)
    elif EVIQUE_MODEL_NAME in selected_models:
        evique_video_segments = shared_video_segments
        evique_video_path_map = read_video_path_map(workdir)

    active_query_base_dir = evique_base_dir if unified_evique_base or shared_video_segments is None else workdir
    if args.skip_generation and args.skip_eval and not args.questions and args.auto_generate_questions <= 0:
        queries = []
        print("[questions] skip-generation + skip-eval: no questions required for index-only run.")
    else:
        queries = prepare_queries(args, active_query_base_dir, output_root)
    evique_base_metrics = _evique_base_metrics(
        base_mode=args.evique_base_mode,
        base_dir=evique_base_dir if args.evique_base_mode == "standalone" else workdir,
        base_result=evique_base_result,
    )
    evique_index_seconds = 0.0
    if EVIQUE_MODEL_NAME in selected_models:
        if evique_video_segments is None:
            raise SystemExit("EVIQUE base segments are unavailable; check --evique-base-mode and --workdir.")
        evique_workdir, evique_index_seconds = build_evique_index(
            args,
            evique_workdir=evique_workdir,
            video_segments=evique_video_segments,
            workdir=workdir if args.evique_base_mode == "shared" else None,
            video_path_map=evique_video_path_map,
            video_paths=video_paths,
            queries=queries,
            evique_base_metrics=evique_base_metrics,
        )
        update_comparison_config_with_visual_compact(output_root, evique_workdir)

    evique_dir = output_root / "answers-evique"
    videorag_dir = output_root / "answers-videorag"
    naiverag_dir = output_root / "answers-naiverag"
    textvideorag_dir = output_root / "answers-textvideorag"
    lightrag_dir = output_root / "answers-lightrag"
    graphrag_l_dir = output_root / "answers-graphrag-l"
    graphrag_g_dir = output_root / "answers-graphrag-g"
    if LIGHTRAG_MODEL_NAME in selected_models:
        run_lightrag_answers(
            args,
            queries,
            output_root=output_root,
            workdir=workdir,
            evique_workdir=evique_workdir,
            evique_base_dir=evique_base_dir,
            output_dir=lightrag_dir,
            lightrag_workdir=lightrag_workdir,
        )
        model_progress.mark(LIGHTRAG_MODEL_NAME, stage="answers ready")
    graphrag_index_seconds = 0.0
    if graphrag_selected_models:
        _, graphrag_index_seconds, _ = build_graphrag_index(
            args,
            output_root=output_root,
            workdir=workdir,
            evique_base_dir=evique_base_dir,
            graphrag_workdir=graphrag_workdir,
        )
        if not args.skip_generation:
            if GRAPHRAG_LOCAL_MODEL_NAME in graphrag_selected_models:
                run_graphrag_answers(
                    args,
                    queries,
                    output_root=output_root,
                    workdir=workdir,
                    evique_workdir=evique_workdir,
                    evique_base_dir=evique_base_dir,
                    graphrag_workdir=graphrag_workdir,
                    output_dir=graphrag_l_dir,
                    model_name=GRAPHRAG_LOCAL_MODEL_NAME,
                    graph_method="local",
                    index_build_seconds=graphrag_index_seconds,
                )
                model_progress.mark(GRAPHRAG_LOCAL_MODEL_NAME, stage="answers ready")
            if GRAPHRAG_GLOBAL_MODEL_NAME in graphrag_selected_models:
                run_graphrag_answers(
                    args,
                    queries,
                    output_root=output_root,
                    workdir=workdir,
                    evique_workdir=evique_workdir,
                    evique_base_dir=evique_base_dir,
                    graphrag_workdir=graphrag_workdir,
                    output_dir=graphrag_g_dir,
                    model_name=GRAPHRAG_GLOBAL_MODEL_NAME,
                    graph_method="global",
                    index_build_seconds=graphrag_index_seconds,
                )
                model_progress.mark(GRAPHRAG_GLOBAL_MODEL_NAME, stage="answers ready")
    if not args.skip_generation:
        if EVIQUE_MODEL_NAME in selected_models:
            run_evique_answers(
                args,
                queries,
                output_dir=evique_dir,
                workdir=workdir,
                evique_workdir=evique_workdir,
                index_build_seconds=evique_index_seconds,
                evique_base_dir=evique_base_dir,
                evique_base_metrics=evique_base_metrics,
            )
            model_progress.mark(EVIQUE_MODEL_NAME, stage="answers ready")
        if VIDEO_RAG_MODEL_NAME in selected_models:
            if rag is None:
                raise SystemExit("VideoRAG was selected but the shared VideoRAG base was not built.")
            run_videorag_answers(
                args,
                rag,
                queries,
                output_dir=videorag_dir,
                index_build_seconds=index_seconds,
                workdir=workdir,
                evique_workdir=evique_workdir,
                evique_base_dir=evique_base_dir,
                evique_base_metrics=evique_base_metrics,
            )
            model_progress.mark(VIDEO_RAG_MODEL_NAME, stage="answers ready")
        if NAIVE_MODEL_NAME in selected_models:
            run_baseline_answers(
                args,
                queries,
                output_root=output_root,
                workdir=workdir,
                evique_workdir=evique_workdir,
                evique_base_dir=evique_base_dir,
                evique_base_metrics=evique_base_metrics,
                output_dir=naiverag_dir,
                model_name=NAIVE_MODEL_NAME,
                pipeline_factory=NaiveRAGPipeline,
            )
            model_progress.mark(NAIVE_MODEL_NAME, stage="answers ready")
        if TEXT_VIDEO_MODEL_NAME in selected_models:
            run_baseline_answers(
                args,
                queries,
                output_root=output_root,
                workdir=workdir,
                evique_workdir=evique_workdir,
                evique_base_dir=evique_base_dir,
                evique_base_metrics=evique_base_metrics,
                output_dir=textvideorag_dir,
                model_name=TEXT_VIDEO_MODEL_NAME,
                pipeline_factory=TextVideoRAGPipeline,
            )
            model_progress.mark(TEXT_VIDEO_MODEL_NAME, stage="answers ready")

    model_progress.mark_remaining(stage="generation skipped" if args.skip_generation else "ready")
    model_progress.close()

    accuracy_file: Path | None = None
    accuracy_scores: dict[str, float] = {}
    if not args.skip_eval:
        if len(selected_models) < 2:
            print("[eval] Only one model selected; skipping LLM-judge evaluation.")
        else:
            _, accuracy_file, accuracy_scores = run_paper_evaluation(
                args,
                queries,
                output_root=output_root,
                selected_models=selected_models,
                evique_dir=evique_dir,
                videorag_dir=videorag_dir,
                naiverag_dir=naiverag_dir,
                textvideorag_dir=textvideorag_dir,
                lightrag_dir=lightrag_dir,
                graphrag_l_dir=graphrag_l_dir,
                graphrag_g_dir=graphrag_g_dir,
            )

    model_dirs: dict[str, Path] = {}
    if EVIQUE_MODEL_NAME in selected_models:
        model_dirs[EVIQUE_MODEL_NAME] = evique_dir
    if VIDEO_RAG_MODEL_NAME in selected_models:
        model_dirs[VIDEO_RAG_MODEL_NAME] = videorag_dir
    if NAIVE_MODEL_NAME in selected_models:
        model_dirs[NAIVE_MODEL_NAME] = naiverag_dir
    if TEXT_VIDEO_MODEL_NAME in selected_models:
        model_dirs[TEXT_VIDEO_MODEL_NAME] = textvideorag_dir
    if LIGHTRAG_MODEL_NAME in selected_models:
        model_dirs[LIGHTRAG_MODEL_NAME] = lightrag_dir
    if GRAPHRAG_LOCAL_MODEL_NAME in selected_models:
        model_dirs[GRAPHRAG_LOCAL_MODEL_NAME] = graphrag_l_dir
    if GRAPHRAG_GLOBAL_MODEL_NAME in selected_models:
        model_dirs[GRAPHRAG_GLOBAL_MODEL_NAME] = graphrag_g_dir

    summary_csv = write_summary(
        output_root=output_root,
        dataset_name=dataset_name,
        video_meta=video_meta,
        model_dirs=model_dirs,
        workdir=workdir,
        evique_workdir=evique_workdir,
        evique_base_mode=args.evique_base_mode,
        evique_base_dir=evique_base_dir,
        evique_base_metrics=evique_base_metrics,
        accuracy_scores=accuracy_scores,
        accuracy_file=accuracy_file,
    )
    print(f"[done] summary: {summary_csv}")
    print(f"[done] output root: {output_root}")


if __name__ == "__main__":
    main()

