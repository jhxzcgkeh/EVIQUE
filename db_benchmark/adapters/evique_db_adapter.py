from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.schema import DBQuery
from db_benchmark.utils import directory_size_mb, read_jsonl, safe_float


REQUIRED_INDEX_FILES = [
    "index_manifest.json",
    "scope_view.jsonl",
    "target_view.jsonl",
    "track_view.jsonl",
    "event_view.jsonl",
    "adaptive_event_view.jsonl",
    "evidence_nodes.jsonl",
    "evidence_relations.jsonl",
    "graph_stats.json",
    "view_stats.json",
]

RETRIEVABLE_INDEX_FILES = [
    "scope_view.jsonl",
    "target_view.jsonl",
    "track_view.jsonl",
    "event_view.jsonl",
    "adaptive_event_view.jsonl",
    "visual_object_view.jsonl",
    "visual_track_view.jsonl",
    "visual_event_view.jsonl",
    "visual_relations.jsonl",
    "visual_relation_view.jsonl",
    "kv_store_video_segments.json",
    "graph_chunk_entity_relation.graphml",
]

BASE_SEGMENT_FILES = [
    "base/kv_store_video_segments.json",
    "base/evique_video_segments.json",
    "kv_store_video_segments.json",
    "evique_video_segments.json",
]

BUILD_PROGRESS_FILES = [
    "index_manifest.json",
    "visual_object_view.jsonl",
    "visual_track_view.jsonl",
    "visual_event_view.jsonl",
    "graph_stats.json",
    "view_stats.json",
]

MOVABLE_LABELS = ["person", "car", "truck", "bus", "van", "bicycle", "motorcycle", "boat"]
VEHICLE_LABELS = ["car", "truck", "bus", "van", "bicycle", "motorcycle", "boat"]
TRAFFIC_OBJECT_LABELS = VEHICLE_LABELS
LABEL_TERMS = {
    "bus": {"bus", "buses"},
    "truck": {"truck", "trucks"},
    "car": {"car", "cars", "automobile", "automobiles"},
    "van": {"van", "vans"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes", "cyclist", "cyclists"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes"},
    "boat": {"boat", "boats", "motorboat", "motorboats", "sailboat", "sailboats"},
    "person": {"person", "people", "pedestrian", "pedestrians", "human", "humans"},
}
LABEL_INTENT_CANDIDATES = {
    "boat": ["boat"],
    "bicycle": ["bicycle"],
    "van": ["car", "truck", "van", "vehicle"],
    "vehicle": VEHICLE_LABELS,
    "traffic_object": TRAFFIC_OBJECT_LABELS,
}
VAN_SYNONYM_METADATA_CANDIDATES = ["car", "truck", "van", "vehicle"]
LABEL_LIKE_FIELDS = ("label", "detector_label", "class", "object_label", "caption", "captions", "text", "object_caption")
VEHICLE_TERMS = {"vehicle", "vehicles"}
OBJECT_TERMS = {"object", "objects"}
TRAFFIC_OBJECT_PHRASES = {"traffic object", "traffic objects"}
TRAFFIC_LIGHT_PHRASES = {"traffic light", "traffic lights"}
TRACKED_OBJECT_PHRASES = {"tracked object", "tracked objects", "moving object", "moving objects"}
ATTRIBUTE_TERMS = {"red", "green", "white", "black", "gray", "grey", "blue", "yellow", "roof", "mast", "masts", "sail", "sails"}
MOTION_TERMS = {"move", "moves", "moving", "across", "trajectory", "travel", "travels", "traveling", "travelling"}
COUNTING_TERMS = {
    "at least",
    "several",
    "multiple",
    "many",
    "dense",
    "density",
    "crowded",
    "congested",
    "count",
    "traffic becomes dense",
}


class EviqueDBAdapter(BaseDBAdapter):
    implementation_fidelity = "native"
    adapter_status = "integrated"

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        self.index_dir = self.context.evique_workdir
        self._retriever = None
        self._build_failed_reason = ""
        self._base_segment_time_index: dict[str, tuple[float, float]] | None = None
        self._build_start_time = ""

    def build_index(self) -> dict:
        start = time.perf_counter()
        self._build_start_time = _utc_timestamp()
        workdir = self.index_dir
        before_snapshot = _snapshot_files(workdir)
        before_inspection = self._inspect_workdir(workdir)
        self.logger.info("build_index action=pending")
        self.logger.info("workdir path=%s", workdir or "")
        self.logger.info("workdir existed before build=%s", bool(workdir and workdir.exists()))
        self.logger.info("workdir file count before build=%s", before_inspection["file_count"])
        self.logger.info("index size before build=%s", before_inspection["index_size_mb"])
        self.logger.info("required index files found before=%s", before_inspection["required_found"])
        self.logger.info("required index files missing before=%s", before_inspection["required_missing"])
        self.logger.info("retrievable index files found before=%s", before_inspection["retrievable_found"])
        self._log_build_snapshot("start", start, workdir)

        if not workdir:
            reason = "--evique-workdir is required for EVIQUE-DB build/reuse"
            self._build_failed_reason = reason
            self.logger.info("build_index action=failed reason=%s", reason)
            return self._index_metadata(
                status="not_available",
                reason=reason,
                start=start,
                before=before_inspection,
                after=before_inspection,
                created_files=[],
            )

        if self.context.reuse_index and before_inspection["reusable"]:
            self.logger.info("build_index action=reuse")
            return self._index_metadata(
                status="reuse_existing",
                reason="",
                start=start,
                before=before_inspection,
                after=before_inspection,
                created_files=[],
            )

        if not self.context.build_index_requested:
            if self.context.reuse_index:
                reason = "requested --reuse-evique-index, but workdir is not a valid EVIQUE index"
            elif before_inspection["reusable"]:
                reason = "valid EVIQUE index exists, but --reuse-evique-index was not provided"
            else:
                reason = "EVIQUE index is missing or incomplete; pass --build-index to build it"
            self._build_failed_reason = reason
            self.logger.info("build_index action=failed reason=%s", reason)
            return self._index_metadata(
                status="not_available",
                reason=reason,
                start=start,
                before=before_inspection,
                after=before_inspection,
                created_files=[],
            )

        if self.context.dry_run:
            reason = "dry-run does not build EVIQUE indexes"
            self._build_failed_reason = reason
            self.logger.info("build_index action=failed reason=%s", reason)
            return self._index_metadata(
                status="not_available",
                reason=reason,
                start=start,
                before=before_inspection,
                after=before_inspection,
                created_files=[],
            )

        video_path = self.context.video_path
        if not video_path or not video_path.exists():
            reason = f"--video is required and must exist to build EVIQUE-DB index: {video_path or ''}"
            self._build_failed_reason = reason
            self.logger.info("build_index action=failed reason=%s", reason)
            return self._index_metadata(
                status="not_available",
                reason=reason,
                start=start,
                before=before_inspection,
                after=before_inspection,
                created_files=[],
            )

        workdir.mkdir(parents=True, exist_ok=True)
        heartbeat_stop = threading.Event()
        heartbeat_thread = self._start_build_heartbeat(start, workdir, heartbeat_stop)
        try:
            self.logger.info("build_index action=build")
            self._build_real_evique_index(workdir=workdir, video_path=video_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._build_failed_reason = reason
            after_inspection = self._inspect_workdir(workdir)
            created_files = _created_files(before_snapshot, _snapshot_files(workdir))
            self.logger.exception("build_index action=failed reason=%s", reason)
            self.logger.info("workdir file count after build=%s", after_inspection["file_count"])
            self.logger.info("index size after build=%s", after_inspection["index_size_mb"])
            self.logger.info("required index files found after=%s", after_inspection["required_found"])
            self.logger.info("required index files missing after=%s", after_inspection["required_missing"])
            self.logger.info("retrievable index files found after=%s", after_inspection["retrievable_found"])
            return self._index_metadata(
                status="failed",
                reason=reason,
                start=start,
                before=before_inspection,
                after=after_inspection,
                created_files=created_files,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

        after_inspection = self._inspect_workdir(workdir)
        created_files = _created_files(before_snapshot, _snapshot_files(workdir))
        self.logger.info("workdir file count after build=%s", after_inspection["file_count"])
        self.logger.info("index size after build=%s", after_inspection["index_size_mb"])
        self.logger.info("required index files found after=%s", after_inspection["required_found"])
        self.logger.info("required index files missing after=%s", after_inspection["required_missing"])
        self.logger.info("retrievable index files found after=%s", after_inspection["retrievable_found"])
        self.logger.info("created files list=%s", created_files)

        if not after_inspection["reusable"]:
            reason = "EVIQUE build completed, but workdir is empty or missing required retrievable index files"
            self._build_failed_reason = reason
            self.logger.info("build_index action=failed reason=%s", reason)
            return self._index_metadata(
                status="failed",
                reason=reason,
                start=start,
                before=before_inspection,
                after=after_inspection,
                created_files=created_files,
            )

        self._build_failed_reason = ""
        return self._index_metadata(
            status="built",
            reason="",
            start=start,
            before=before_inspection,
            after=after_inspection,
            created_files=created_files,
        )

    def run_query(self, query: DBQuery) -> list[dict]:
        inspection = self._inspect_workdir(self.index_dir)
        if self._build_failed_reason:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason=self._build_failed_reason,
                    metadata=self._query_metadata(extra={"index_inspection": inspection}),
                )
            ]
        if not inspection["reusable"]:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="EVIQUE workdir is not a valid reusable index; build it with --build-index or pass --reuse-evique-index for a valid workdir",
                    metadata=self._query_metadata(extra={"index_inspection": inspection}),
                )
            ]
        if not self.context.reuse_index and not self.context.build_index_requested:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="valid EVIQUE index exists, but --reuse-evique-index was not provided and --build-index was not requested",
                    metadata=self._query_metadata(extra={"index_inspection": inspection}),
                )
            ]

        self.logger.info("query text=%s", query.query)
        t0 = time.perf_counter()
        try:
            retriever = self._get_retriever()
            package = retriever.retrieve(
                query.query,
                {
                    "dataset": query.dataset,
                    "video_id": query.video_id,
                    "source_vid": query.video_id,
                    "video_name": query.video_id,
                },
            )
        except Exception as exc:
            return [self.exception_record(query, exc)]
        query_time = time.perf_counter() - t0
        timing = {
            "query_time_sec": round(query_time, 6),
            "rerank_time_sec": 0.0,
            "total_time_sec": round(query_time, 6),
        }

        evidence = list(package.get("evidence") or [])
        self.logger.info("raw retrieved evidence count=%s", len(evidence))
        self.logger.info("sample evidence keys=%s", [sorted(item.keys())[:40] for item in evidence[:3] if isinstance(item, dict)])
        deterministic_rows = self._deterministic_operator_results(query, timing=timing, package=package)
        if deterministic_rows:
            self.logger.info("deterministic operator recovered window count=%s", len(deterministic_rows))
        if len(deterministic_rows) >= self.context.top_k:
            return deterministic_rows[: self.context.top_k]
        if not evidence:
            fallback_rows = self._visual_index_fallback(query, timing=timing, package=package, retriever=retriever)
            if fallback_rows:
                self.logger.info("visual fallback used=true recovered window count=%s", len(fallback_rows))
                return self._merge_ranked_results(deterministic_rows, fallback_rows)
            if deterministic_rows:
                return self._merge_ranked_results(deterministic_rows, [])
            self.logger.info("visual fallback used=false recovered window count=0")
            return [
                self.status_record(
                    query,
                    status="no_evidence",
                    reason="EvidenceRetriever returned no evidence and deterministic visual-index fallback found no matching visual rows",
                    timing=timing,
                    metadata=self._query_metadata(
                        package=package,
                        extra={
                            "evidence_count": 0,
                            "view_order": package.get("view_order") or [],
                            "visual_used": bool(package.get("visual_used")),
                            "fallback_used": False,
                            "fallback_reason": "empty_retriever_evidence",
                        },
                    ),
                )
            ]

        results, skipped = self._native_postprocess_evidence(
            query,
            evidence=evidence,
            timing=timing,
            package=package,
            retriever=retriever,
        )
        self.logger.info("native postprocess recovered window count=%s", len(results))
        if results:
            if skipped:
                self.logger.info("query_id=%s skipped %s evidence rows without recoverable windows", query.query_id, skipped)
            return self._merge_ranked_results(deterministic_rows, results)
        if deterministic_rows:
            return self._merge_ranked_results(deterministic_rows, [])

        return [
            self.status_record(
                query,
                status="no_window",
                reason="EVIQUE returned evidence, but no timestamp/start/end/segment window could be recovered",
                timing=timing,
                metadata=self._query_metadata(
                    package=package,
                    extra={
                        "evidence_count": len(evidence),
                        "warnings": list(dict.fromkeys(warnings)),
                        "view_order": package.get("view_order") or [],
                        "visual_used": bool(package.get("visual_used")),
                    },
                ),
            )
        ]

    def _native_postprocess_evidence(
        self,
        query: DBQuery,
        *,
        evidence: list[dict[str, Any]],
        timing: dict[str, Any],
        package: dict[str, Any],
        retriever: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        intent = _extract_visual_intent(query.query)
        label_intents = set(intent["labels"])
        records: list[dict[str, Any]] = []
        skipped = 0
        for original_rank, item in enumerate(evidence, start=1):
            window_candidates = self._db_window_candidates(item, retriever, include_record=True, max_windows=2)
            if not window_candidates:
                skipped += 1
                continue
            label_match = _first_matching_label(item, label_intents)
            source_view = _native_source_view(item)
            evidence_type = self._evidence_type(item)
            native_label_matched = bool(label_match)
            if label_intents and not native_label_matched:
                continue
            motion_verified = _record_motion_verified(item)
            stationary_motion = _record_stationary_motion(item)
            if intent.get("motion") and stationary_motion:
                continue
            warnings: list[str] = []
            if intent["attributes"] and (not native_label_matched or not _record_matches_attribute(item, intent["attributes"])):
                warnings.append("attribute_not_verified_by_visual_index")
            if intent.get("motion") and not motion_verified:
                warnings.append("motion_not_verified_by_visual_index")
            synonym_metadata = _synonym_fallback_metadata(intent, label_match)
            priority = _native_sort_priority(
                source_view=source_view,
                evidence_type=evidence_type,
                label_matched=native_label_matched,
                window_clipped=any(bool(window.get("window_clipped")) for window in window_candidates),
                motion_required=bool(intent.get("motion")),
                motion_verified=motion_verified,
            )
            for window_index, window_candidate in enumerate(window_candidates, start=1):
                start_time, end_time = window_candidate["window"]
                record = self.ok_record(
                    query,
                    rank=0,
                    start_time=start_time,
                    end_time=end_time,
                    score=safe_float(item.get("score"), 0.0),
                    bbox=item.get("bbox"),
                    track_id=_as_optional_str(item.get("track_id")),
                    evidence_type=evidence_type,
                    evidence_text=str(item.get("text") or item.get("short_text") or ""),
                    timing=timing,
                    metadata=self._query_metadata(
                        package=package,
                        extra={
                            "record_id": item.get("id") or item.get("node_id"),
                            "source_view": source_view,
                            "source_views": _sequence_values(item.get("source_views")),
                            "segment_id": item.get("segment_id"),
                            "segment_ids": _sequence_values(item.get("segment_ids")),
                            "warnings": list(dict.fromkeys(warnings)),
                            "recovered_window_source": "evique_evidence",
                            "native_postprocess_used": True,
                            "native_label_matched": native_label_matched,
                            "label_match": label_match,
                            "motion_required": bool(intent.get("motion")),
                            "motion_verified": motion_verified,
                            "stationary_motion": stationary_motion,
                            "window_clipped": bool(window_candidate.get("window_clipped")),
                            "original_start_time": window_candidate.get("original_start_time"),
                            "original_end_time": window_candidate.get("original_end_time"),
                            "original_duration": window_candidate.get("original_duration"),
                            "native_window_mode": window_candidate.get("mode"),
                            "native_original_rank": original_rank,
                            "native_window_index": window_index,
                            "db_sort_priority": priority,
                            **synonym_metadata,
                        },
                    ),
                )
                records.append(record)

        label_matched_records = [row for row in records if row.get("metadata", {}).get("native_label_matched")]
        if label_intents and len(label_matched_records) < self.context.top_k:
            fallback_rows = self._visual_index_fallback(
                query,
                timing=timing,
                package=package,
                retriever=retriever,
                fallback_reason="insufficient_label_matched_native_evidence",
                supplement=True,
            )
            for row in fallback_rows:
                metadata = row.setdefault("metadata", {})
                metadata["supplement_fallback_used"] = True
                metadata["fallback_reason"] = "insufficient_label_matched_native_evidence"
                metadata["db_sort_priority"] = _result_sort_priority(row)
            records.extend(fallback_rows)

        records.sort(key=_result_sort_key)
        results: list[dict[str, Any]] = []
        for rank, row in enumerate(records[: self.context.top_k], start=1):
            row["rank"] = rank
            results.append(row)
        return results, skipped

    def _deterministic_operator_results(
        self,
        query: DBQuery,
        *,
        timing: dict[str, Any],
        package: dict[str, Any],
    ) -> list[dict[str, Any]]:
        intent = _extract_visual_intent(query.query)
        labels = list(intent.get("labels") or [])
        operator = _deterministic_operator_type(query, intent)
        if not operator or not labels:
            return []
        self.logger.info("deterministic operator=%s labels=%s query_type=%s", operator, labels, query.type)
        if operator == "counting":
            candidates = self._deterministic_counting_candidates(intent)
            evidence_type = "deterministic_counting"
        elif operator == "attribute_size":
            candidates = self._deterministic_large_object_candidates(intent)
            evidence_type = "deterministic_large_object"
        elif operator == "spatial":
            candidates = self._deterministic_spatial_center_candidates(intent)
            evidence_type = "deterministic_spatial_center"
        elif operator == "motion":
            candidates = self._deterministic_motion_candidates(intent)
            evidence_type = "deterministic_motion"
        else:
            candidates = self._deterministic_existence_candidates(intent)
            evidence_type = "deterministic_existence"
            operator = "existence"
        return self._records_from_operator_candidates(
            query,
            candidates,
            timing=timing,
            package=package,
            operator=operator,
            evidence_type=evidence_type,
            label_candidates=labels,
            intent=intent,
        )

    def _deterministic_existence_candidates(self, intent: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for source_view, filename in [
            ("visual_object", "visual_object_view.jsonl"),
            ("visual_track", "visual_track_view.jsonl"),
            ("visual_event", "visual_event_view.jsonl"),
            ("adaptive_event", "adaptive_event_view.jsonl"),
        ]:
            for row in self._read_visual_rows(filename):
                label = _first_matching_label(row, set(intent["labels"]))
                if not label:
                    continue
                window = self._operator_window_for_row(row)
                if not window:
                    continue
                confidence = _record_confidence(row)
                area = _record_bbox_area(row) or 0.0
                track_support = _record_track_support(row)
                exact_score = _label_match_exactness(intent, label)
                source_score = _operator_source_score(source_view)
                score = exact_score * 1_000_000.0 + source_score * 100_000.0 + confidence * 1_000.0 + min(area, 1_000_000.0) / 1_000.0 + track_support
                candidates.append(
                    self._operator_candidate(
                        row=row,
                        source_view=source_view,
                        label=label,
                        window=window,
                        score=score,
                        components={
                            "label_exactness": exact_score,
                            "source_score": source_score,
                            "confidence": confidence,
                            "bbox_area": area,
                            "track_support": track_support,
                        },
                    )
                )
        return candidates

    def _deterministic_counting_candidates(self, intent: dict[str, Any]) -> list[dict[str, Any]]:
        labels = set(intent["labels"])
        rows: list[dict[str, Any]] = []
        for source_view, filename in [
            ("visual_object", "visual_object_view.jsonl"),
            ("visual_track", "visual_track_view.jsonl"),
        ]:
            for row in self._read_visual_rows(filename):
                label = _first_matching_label(row, labels)
                if not label:
                    continue
                window = self._operator_window_for_row(row)
                if not window:
                    continue
                rows.append({"row": row, "source_view": source_view, "label": label, "window": window})
        if not rows:
            return []
        stride = max(0.001, float(self.context.stride or self.context.window_size or 1.0))
        window_size = max(0.001, float(self.context.window_size or 1.0))
        starts = sorted({max(0.0, float(int(item["window"][0] // stride) * stride)) for item in rows})
        candidates: list[dict[str, Any]] = []
        for start in starts:
            end = start + window_size
            window_rows = [item for item in rows if _windows_overlap(item["window"], (start, end))]
            if not window_rows:
                continue
            track_ids = {_operator_track_id(item["row"]) or f"row:{idx}" for idx, item in enumerate(window_rows)}
            label_counts: dict[str, int] = {}
            for item in window_rows:
                label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
            track_count = len(track_ids)
            object_count = len(window_rows)
            if max(track_count, object_count) < 2:
                continue
            sample = max(window_rows, key=lambda item: _record_confidence(item["row"]))
            score = float(track_count) * 10_000.0 + float(object_count) * 100.0 + _record_confidence(sample["row"])
            candidates.append(
                self._operator_candidate(
                    row=sample["row"],
                    source_view=sample["source_view"],
                    label=sample["label"],
                    window=(start, end),
                    score=score,
                    components={
                        "track_count": track_count,
                        "object_count": object_count,
                        "label_counts": label_counts,
                    },
                    extra={"track_count": track_count, "object_count": object_count, "label_counts": label_counts},
                )
            )
        return candidates

    def _deterministic_large_object_candidates(self, intent: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for source_view, filename in [
            ("visual_object", "visual_object_view.jsonl"),
            ("visual_track", "visual_track_view.jsonl"),
        ]:
            for row in self._read_visual_rows(filename):
                label = _first_matching_label(row, set(intent["labels"]))
                if not label:
                    continue
                area = _record_bbox_area(row)
                if area is None:
                    area = _record_track_area(row)
                if area is None:
                    continue
                window = self._operator_window_for_row(row)
                if not window:
                    continue
                exact_score = _label_match_exactness(intent, label)
                source_score = _operator_source_score(source_view)
                score = exact_score * 1_000_000.0 + source_score * 100_000.0 + float(area)
                candidates.append(
                    self._operator_candidate(
                        row=row,
                        source_view=source_view,
                        label=label,
                        window=window,
                        score=score,
                        components={
                            "label_exactness": exact_score,
                            "source_score": source_score,
                            "bbox_area": area,
                        },
                    )
                )
        return candidates

    def _deterministic_spatial_center_candidates(self, intent: dict[str, Any]) -> list[dict[str, Any]]:
        frame_size = self._infer_visual_frame_size()
        candidates: list[dict[str, Any]] = []
        for row in self._read_visual_rows("visual_object_view.jsonl"):
            label = _first_matching_label(row, set(intent["labels"]))
            if not label:
                continue
            center_score = _record_center_priority_score(row, frame_size=frame_size)
            if center_score is None:
                continue
            window = self._operator_window_for_row(row)
            if not window:
                continue
            exact_score = _label_match_exactness(intent, label)
            score = exact_score * 1_000_000.0 + float(center_score) * 100_000.0 + _record_confidence(row)
            candidates.append(
                self._operator_candidate(
                    row=row,
                    source_view="visual_object",
                    label=label,
                    window=window,
                    score=score,
                    components={
                        "label_exactness": exact_score,
                        "center_score": center_score,
                        "confidence": _record_confidence(row),
                    },
                )
            )
        return candidates

    def _deterministic_motion_candidates(self, intent: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for source_view, filename in [
            ("visual_track", "visual_track_view.jsonl"),
            ("visual_event", "visual_event_view.jsonl"),
        ]:
            for row in self._read_visual_rows(filename):
                label = _first_matching_label(row, set(intent["labels"]))
                if not label or not _record_motion_verified(row):
                    continue
                window = self._operator_window_for_row(row)
                if not window:
                    continue
                displacement = _record_track_displacement(row) or 0.0
                exact_score = _label_match_exactness(intent, label)
                source_score = _operator_source_score(source_view)
                score = exact_score * 1_000_000.0 + source_score * 100_000.0 + displacement * 1_000.0 + _record_confidence(row)
                candidates.append(
                    self._operator_candidate(
                        row=row,
                        source_view=source_view,
                        label=label,
                        window=window,
                        score=score,
                        components={
                            "label_exactness": exact_score,
                            "source_score": source_score,
                            "track_displacement": displacement,
                            "motion_verified": True,
                        },
                        extra={"motion_verified": True},
                    )
                )
        return candidates

    def _operator_candidate(
        self,
        *,
        row: dict[str, Any],
        source_view: str,
        label: str,
        window: tuple[float, float],
        score: float,
        components: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "row": row,
            "source_view": source_view,
            "label": label,
            "window": tuple(sorted((float(window[0]), float(window[1])))),
            "score": round(float(score), 6),
            "components": components,
            **(extra or {}),
        }

    def _operator_window_for_row(self, row: dict[str, Any]) -> tuple[float, float] | None:
        timestamp = self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance"))
        if timestamp is not None:
            return self._aligned_window_from_timestamp(timestamp)
        points_window = _points_window(row.get("compact_points")) or _points_window(row.get("bbox_sequence"))
        if points_window is not None:
            midpoint = (float(points_window[0]) + float(points_window[1])) / 2.0
            return self._aligned_window_from_timestamp(midpoint)
        direct = self._start_end_from_sources(row, row.get("metadata"), row.get("provenance"))
        if direct is not None:
            midpoint = (float(direct[0]) + float(direct[1])) / 2.0
            return self._aligned_window_from_timestamp(midpoint)
        return None

    def _aligned_window_from_timestamp(self, timestamp: float) -> tuple[float, float]:
        stride = max(0.001, float(self.context.stride or self.context.window_size or 1.0))
        window_size = max(0.001, float(self.context.window_size or 1.0))
        start = max(0.0, float(int(float(timestamp) // stride) * stride))
        return start, start + window_size

    def _records_from_operator_candidates(
        self,
        query: DBQuery,
        candidates: list[dict[str, Any]],
        *,
        timing: dict[str, Any],
        package: dict[str, Any],
        operator: str,
        evidence_type: str,
        label_candidates: list[str],
        intent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for item in sorted(candidates, key=lambda row: (-float(row["score"]), float(row["window"][0]), str(_record_id(row["row"])))):
            if any(_windows_overlap(item["window"], existing["window"]) for existing in selected):
                continue
            selected.append(item)
            if len(selected) >= self.context.top_k:
                break
        rows: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            row = item["row"]
            start_time, end_time = item["window"]
            label = item["label"]
            metadata = self._query_metadata(
                package=package,
                extra={
                    "deterministic_operator_used": True,
                    "deterministic_operator": operator,
                    "label_candidates": list(label_candidates),
                    "source_view": item["source_view"],
                    "label_match": label,
                    "operator_score": item["score"],
                    "score_components": item["components"],
                    "raw_record_id": _record_id(row),
                    "db_sort_priority": 0,
                    "native_postprocess_used": False,
                    "native_label_matched": True,
                    "window_clipped": False,
                    "native_window_mode": "deterministic_window",
                    **_synonym_fallback_metadata(intent, label),
                },
            )
            for key in ("object_count", "track_count", "label_counts", "motion_verified"):
                if key in item:
                    metadata[key] = item[key]
            rows.append(
                self.ok_record(
                    query,
                    rank=rank,
                    start_time=start_time,
                    end_time=end_time,
                    score=float(item["score"]),
                    bbox=_record_bbox(row),
                    track_id=_as_optional_str(_operator_track_id(row)),
                    evidence_type=evidence_type,
                    evidence_text=_deterministic_evidence_text(operator, label, item["source_view"], start_time, end_time, item),
                    timing=timing,
                    metadata=metadata,
                )
            )
        return rows

    def _merge_ranked_results(self, primary: list[dict[str, Any]], supplemental: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_windows: set[tuple[Any, ...]] = set()
        for row in sorted(primary + supplemental, key=_result_sort_key):
            key = _result_window_identity(row)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            merged.append(row)
            if len(merged) >= self.context.top_k:
                break
        for rank, row in enumerate(merged, start=1):
            row["rank"] = rank
        return merged

    def _visual_index_fallback(
        self,
        query: DBQuery,
        *,
        timing: dict[str, Any],
        package: dict[str, Any],
        retriever: Any,
        fallback_reason: str = "empty_retriever_evidence",
        supplement: bool = False,
    ) -> list[dict[str, Any]]:
        intent = _extract_visual_intent(query.query)
        labels = intent["labels"]
        self.logger.info(
            "visual fallback intent labels=%s attributes=%s counting=%s large_object=%s center_object=%s motion=%s traffic_object=%s specific=%s",
            labels,
            intent["attributes"],
            intent["counting"],
            intent["large_object"],
            intent["center_object"],
            intent.get("motion"),
            intent.get("traffic_object"),
            intent.get("specific_label_intents"),
        )
        if not labels:
            return []
        if intent["counting"]:
            rows = self._counting_visual_object_fallback(
                query,
                labels=labels,
                timing=timing,
                package=package,
                retriever=retriever,
                fallback_reason=fallback_reason,
                supplement=supplement,
            )
            if rows:
                return rows

        candidates: list[dict[str, Any]] = []
        matched_row_labels: set[str] = set()
        inferred_frame_size = self._infer_visual_frame_size() if intent["center_object"] else None
        for source_view, filename in [
            ("visual_event", "visual_event_view.jsonl"),
            ("visual_track", "visual_track_view.jsonl"),
            ("visual_object", "visual_object_view.jsonl"),
        ]:
            for row in self._read_visual_rows(filename):
                row_labels = [label for label in _record_label_candidates(row) if label in labels]
                if not row_labels:
                    continue
                if intent.get("motion") and not _record_motion_verified(row):
                    continue
                label = row_labels[0]
                window_candidates = self._fallback_record_windows(row, source_view, retriever)
                if not window_candidates:
                    continue
                matched_row_labels.update(row_labels)
                warnings = []
                if intent["attributes"] and not _record_matches_attribute(row, intent["attributes"]):
                    warnings.append("attribute_not_verified_by_visual_index")
                base_score = self._fallback_sort_score(
                    row,
                    source_view,
                    query.query,
                    intent=intent,
                    frame_size=inferred_frame_size,
                )
                for window_candidate in window_candidates:
                    candidates.append(
                        {
                            "row": row,
                            "source_view": source_view,
                            "label": label,
                            "window": window_candidate["window"],
                            "window_mode": window_candidate["mode"],
                            "representative_timestamp": window_candidate.get("representative_timestamp"),
                            "original_track_start_time": window_candidate.get("original_track_start_time"),
                            "original_track_end_time": window_candidate.get("original_track_end_time"),
                            "original_track_duration": window_candidate.get("original_track_duration"),
                            "window_clipped": bool(window_candidate.get("window_clipped")),
                            "native_window_mode": window_candidate.get("mode"),
                            "motion_verified": _record_motion_verified(row),
                            "warnings": warnings,
                            "score": base_score,
                        }
                    )

        synonym_fallback_metadata: dict[str, Any] = {}
        if "van" in intent.get("original_label_intents", []):
            if "van" in matched_row_labels:
                candidates = [item for item in candidates if item["label"] == "van"]
            elif candidates:
                synonym_fallback_metadata = {
                    "synonym_fallback_used": True,
                    "original_label_intent": "van",
                    "mapped_label_candidates": list(VAN_SYNONYM_METADATA_CANDIDATES),
                }

        candidates.sort(key=lambda item: (-float(item["score"]), item["window"][0], str(_record_id(item["row"]))))
        results = []
        for rank, item in enumerate(candidates[: self.context.top_k], start=1):
            row = item["row"]
            source_view = item["source_view"]
            start_time, end_time = item["window"]
            kind = source_view.replace("visual_", "")
            label = item["label"]
            evidence_text = _fallback_evidence_text(label, kind, row, start_time, end_time, item)
            results.append(
                self.ok_record(
                    query,
                    rank=rank,
                    start_time=start_time,
                    end_time=end_time,
                    score=float(item["score"]),
                    bbox=_record_bbox(row),
                    track_id=_as_optional_str(row.get("track_id") or row.get("actor_track_id")),
                    evidence_type=f"{source_view}_fallback",
                    evidence_text=evidence_text,
                    timing=timing,
                    metadata=self._query_metadata(
                        package=package,
                        extra={
                            "fallback_used": True,
                            "fallback_reason": fallback_reason,
                            **({"supplement_fallback_used": True} if supplement else {}),
                            "source_view": source_view,
                            "label_match": label,
                            "raw_record_id": _record_id(row),
                            "fallback_window_mode": item["window_mode"],
                            "original_track_start_time": item.get("original_track_start_time"),
                            "original_track_end_time": item.get("original_track_end_time"),
                            "original_track_duration": item.get("original_track_duration"),
                            "window_clipped": bool(item.get("window_clipped")),
                            "motion_required": bool(intent.get("motion")),
                            "motion_verified": bool(item.get("motion_verified")),
                            "original_start_time": item.get("original_track_start_time"),
                            "original_end_time": item.get("original_track_end_time"),
                            "original_duration": item.get("original_track_duration"),
                            "native_window_mode": item.get("window_mode"),
                            "db_sort_priority": 1,
                            "warnings": item["warnings"],
                            **synonym_fallback_metadata,
                        },
                    ),
                )
            )
        return results

    def _counting_visual_object_fallback(
        self,
        query: DBQuery,
        *,
        labels: list[str],
        timing: dict[str, Any],
        package: dict[str, Any],
        retriever: Any,
        fallback_reason: str = "empty_retriever_evidence",
        supplement: bool = False,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self._read_visual_rows("visual_object_view.jsonl")
            if any(label in labels for label in _record_label_candidates(row))
            and self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance")) is not None
        ]
        if not rows:
            return []
        stride = max(0.001, float(self.context.stride or self.context.window_size or 1.0))
        window_size = max(0.001, float(self.context.window_size or 1.0))
        timestamps = sorted({float(self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance")) or 0.0) for row in rows})
        starts = sorted({max(0.0, float(int(ts // stride) * stride)) for ts in timestamps})
        windows = []
        for start in starts:
            end = start + window_size
            window_rows = [
                row
                for row in rows
                if start <= float(self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance")) or -1.0) < end
            ]
            if not window_rows:
                continue
            label_counts: dict[str, int] = {}
            for row in window_rows:
                for label in _record_label_candidates(row):
                    if label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "count": len(window_rows),
                    "label_counts": label_counts,
                    "sample": max(window_rows, key=lambda row: safe_float(row.get("confidence"), 0.0) or 0.0),
                }
            )
        windows.sort(key=lambda item: (-int(item["count"]), float(item["start"])))
        results = []
        for rank, item in enumerate(windows[: self.context.top_k], start=1):
            sample = item["sample"]
            results.append(
                self.ok_record(
                    query,
                    rank=rank,
                    start_time=float(item["start"]),
                    end_time=float(item["end"]),
                    score=float(item["count"]),
                    bbox=_record_bbox(sample),
                    track_id=_as_optional_str(sample.get("track_id")),
                    evidence_type="visual_object_fallback",
                    evidence_text=(
                        f"Fallback match: dense object window with {item['count']} movable object detections "
                        f"from {float(item['start']):.2f}s to {float(item['end']):.2f}s."
                    ),
                    timing=timing,
                    metadata=self._query_metadata(
                        package=package,
                        extra={
                            "fallback_used": True,
                            "fallback_reason": fallback_reason,
                            **({"supplement_fallback_used": True} if supplement else {}),
                            "source_view": "visual_object",
                            "label_match": ",".join(labels),
                            "raw_record_id": _record_id(sample),
                            "fallback_window_mode": "count_window",
                            "original_track_start_time": None,
                            "original_track_end_time": None,
                            "original_track_duration": None,
                            "object_count": item["count"],
                            "vehicle_count": sum(item["label_counts"].get(label, 0) for label in VEHICLE_LABELS),
                            "label_counts": item["label_counts"],
                            "db_sort_priority": 3,
                            "warnings": [],
                        },
                    ),
                )
            )
        return results

    def _start_build_heartbeat(self, start: float, workdir: Path, stop_event: threading.Event) -> threading.Thread:
        def run() -> None:
            while not stop_event.wait(30.0):
                self._log_build_snapshot("heartbeat", start, workdir)

        thread = threading.Thread(target=run, name="evique-db-build-heartbeat", daemon=True)
        thread.start()
        return thread

    def _log_build_snapshot(self, status: str, start: float, workdir: Path | None) -> None:
        inspection = self._inspect_workdir(workdir)
        key_files = _key_file_status(workdir)
        elapsed = time.perf_counter() - start
        if status == "heartbeat":
            message = (
                f"[heartbeat] stage=build_index elapsed={elapsed:.2f}s "
                f"workdir_size={inspection['index_size_mb']} file_count={inspection['file_count']} "
                f"key_files={key_files}"
            )
        else:
            message = (
                f"[progress] stage=build_index status={status} elapsed={elapsed:.2f}s "
                f"build_start_time={self._build_start_time} "
                f"workdir_size={inspection['index_size_mb']} file_count={inspection['file_count']} "
                f"key_files={key_files}"
            )
        self._progress_log(message)

    def _progress_log(self, message: str) -> None:
        self.logger.info(message)
        if getattr(self.context, "progress", False):
            print(message, flush=True)

    def _read_visual_rows(self, filename: str) -> list[dict[str, Any]]:
        if not self.index_dir:
            return []
        path = self.index_dir / filename
        if not path.exists():
            return []
        try:
            return read_jsonl(path)
        except Exception as exc:
            self.logger.info("visual fallback failed to read %s: %s", path, exc)
            return []

    def _fallback_record_windows(self, row: dict[str, Any], source_view: str, retriever: Any) -> list[dict[str, Any]]:
        return self._db_window_candidates(row, retriever, include_record=False, max_windows=2)

    def _db_window_candidates(
        self,
        row: dict[str, Any],
        retriever: Any,
        *,
        include_record: bool,
        max_windows: int = 2,
    ) -> list[dict[str, Any]]:
        sources: list[Any] = [row, row.get("metadata"), row.get("provenance")]
        if include_record:
            sources.append(row.get("record"))
        direct = self._start_end_from_sources(*sources)
        timestamp = self._timestamp_from_sources(*sources)
        timestamp_values = _record_timestamp_values(row, include_record=include_record)
        timestamp_window = (min(timestamp_values), max(timestamp_values)) if timestamp_values else None
        original = direct or timestamp_window
        if original is None:
            span = safe_float(row.get("span_seconds"), None)
            if span is not None and timestamp is not None:
                original = (float(timestamp), float(timestamp) + max(0.0, float(span)))
        if original is None:
            original = self._segment_fallback_window(row, retriever)
        if original is None and timestamp is not None:
            original = (float(timestamp), float(timestamp))
        if original is None:
            return []

        original_start, original_end = sorted((float(original[0]), float(original[1])))
        original_duration = max(0.0, original_end - original_start)
        max_full_duration = max(2.0 * float(self.context.window_size), 30.0)
        if original_duration > max_full_duration:
            representatives = [
                float(value)
                for value in timestamp_values
                if original_start <= float(value) <= original_end
            ]
            if not representatives and timestamp is not None and original_start <= float(timestamp) <= original_end:
                representatives = [float(timestamp)]
            mode = "timestamp_window" if representatives else "representative_window"
            if representatives:
                representatives = _select_representative_timestamps(
                    representatives,
                    original_start,
                    original_end,
                    limit=max_windows,
                )
            else:
                representatives = _representative_track_timestamps(row, original_start, original_end, limit=max_windows)
            return [
                {
                    "window": self._window_from_timestamp(value),
                    "mode": mode,
                    "representative_timestamp": float(value),
                    "original_track_start_time": original_start,
                    "original_track_end_time": original_end,
                    "original_track_duration": original_duration,
                    "original_start_time": original_start,
                    "original_end_time": original_end,
                    "original_duration": original_duration,
                    "window_clipped": True,
                }
                for value in representatives
            ]

        if timestamp is not None and original_start == original_end:
            window = self._window_from_timestamp(timestamp)
            mode = "timestamp_window"
            representative = float(timestamp)
        elif original_start == original_end:
            window = self._window_from_timestamp(original_start)
            mode = "timestamp_window"
            representative = original_start
        else:
            window = (original_start, original_end)
            mode = "representative_window"
            representative = (original_start + original_end) / 2.0
        return [
            {
                "window": window,
                "mode": mode,
                "representative_timestamp": representative,
                "original_track_start_time": original_start,
                "original_track_end_time": original_end,
                "original_track_duration": original_duration,
                "original_start_time": original_start,
                "original_end_time": original_end,
                "original_duration": original_duration,
                "window_clipped": False,
            }
        ]

    def _fallback_track_windows(self, row: dict[str, Any], retriever: Any) -> list[dict[str, Any]]:
        direct = self._start_end_from_sources(row, row.get("metadata"), row.get("provenance"))
        timestamp_window = _timestamp_list_window(row.get("timestamps"))
        point_window = _points_window(row.get("compact_points")) or _points_window(row.get("bbox_sequence"))
        original = direct or timestamp_window or point_window
        if original is None:
            span = safe_float(row.get("span_seconds"), None)
            timestamp = self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance"))
            if span is not None and timestamp is not None:
                original = (float(timestamp), float(timestamp) + max(0.0, float(span)))
        if original is None:
            segment_window = self._segment_fallback_window(row, retriever)
            original = segment_window
        if original is None:
            timestamp = self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance"))
            if timestamp is not None:
                original = (float(timestamp), float(timestamp))
        if original is None:
            return []

        original_start, original_end = sorted((float(original[0]), float(original[1])))
        original_duration = max(0.0, original_end - original_start)
        max_full_track_duration = max(2.0 * float(self.context.window_size), 30.0)
        if original_duration <= max_full_track_duration:
            if original_start == original_end:
                window = self._window_from_timestamp(original_start)
                mode = "timestamp_window"
                representative = original_start
            else:
                window = (original_start, original_end)
                mode = "track_representative_window"
                representative = (original_start + original_end) / 2.0
            return [
                {
                    "window": window,
                    "mode": mode,
                    "representative_timestamp": representative,
                    "original_track_start_time": original_start,
                    "original_track_end_time": original_end,
                    "original_track_duration": original_duration,
                }
            ]

        representatives = _representative_track_timestamps(row, original_start, original_end, limit=2)
        return [
            {
                "window": self._window_from_timestamp(timestamp),
                "mode": "track_representative_window",
                "representative_timestamp": float(timestamp),
                "original_track_start_time": original_start,
                "original_track_end_time": original_end,
                "original_track_duration": original_duration,
            }
            for timestamp in representatives[:2]
        ]

    def _segment_fallback_window(self, row: dict[str, Any], retriever: Any) -> tuple[float, float] | None:
        timestamp = self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance"))
        if timestamp is not None:
            return self._window_from_timestamp(timestamp)
        segment_ids = self._segment_ids_from_sources(row, row.get("metadata"), row.get("provenance"))
        starts: list[float] = []
        ends: list[float] = []
        for segment_id in segment_ids:
            scope = getattr(retriever, "scopes_by_segment", {}).get(segment_id)
            window = self._start_end_from_sources(
                scope,
                scope.get("metadata") if isinstance(scope, dict) else None,
                scope.get("provenance") if isinstance(scope, dict) else None,
            )
            if window is None:
                window = self._base_segment_time_index_lookup(segment_id)
            if window:
                starts.append(window[0])
                ends.append(window[1])
        if starts and ends:
            return min(starts), max(ends)
        return None

    def _infer_visual_frame_size(self) -> tuple[float, float] | None:
        max_right = 0.0
        max_bottom = 0.0
        for filename in ("visual_object_view.jsonl", "visual_track_view.jsonl", "visual_event_view.jsonl"):
            for row in self._read_visual_rows(filename):
                explicit = _record_frame_size(row)
                if explicit is not None:
                    return explicit
                bbox = _record_bbox_xyxy(row)
                if bbox is None:
                    continue
                max_right = max(max_right, bbox[0], bbox[2])
                max_bottom = max(max_bottom, bbox[1], bbox[3])
        if max_right <= 0.0 or max_bottom <= 0.0:
            return None
        if max_right <= 1.5 and max_bottom <= 1.5:
            return 1.0, 1.0
        return max_right, max_bottom

    def _fallback_sort_score(
        self,
        row: dict[str, Any],
        source_view: str,
        query_text: str,
        *,
        intent: dict[str, Any] | None = None,
        frame_size: tuple[float, float] | None = None,
    ) -> float:
        intent = intent or {}
        score = safe_float(row.get("score"), None)
        if score is None:
            score = safe_float(row.get("confidence"), None)
        if score is None:
            score = safe_float(row.get("detector_score"), None)
        if source_view == "visual_track":
            start_end = self._start_end_from_sources(row)
            if start_end:
                score = max(float(score or 0.0), max(0.0, start_end[1] - start_end[0]) / 10.0)
            elif isinstance(row.get("timestamps"), list):
                score = max(float(score or 0.0), len(row.get("timestamps") or []) / 10.0)
        if source_view == "visual_event" and any(term in query_text.lower() for term in ("move", "moving", "moves")):
            score = float(score or 0.0) + 0.25
        if score is None:
            timestamp = self._timestamp_from_sources(row, row.get("metadata"), row.get("provenance"))
            score = 1.0 / (1.0 + float(timestamp or 0.0))
        area_score = _record_bbox_area(row)
        center_score = _record_center_priority_score(row, frame_size=frame_size)
        if intent.get("center_object"):
            if center_score is None:
                return -1.0
            if intent.get("large_object") and area_score is not None:
                return round(float(center_score) * 1_000_000.0 + float(area_score), 6)
            return round(float(center_score), 6)
        if intent.get("large_object"):
            if area_score is None:
                return -1.0
            return round(float(area_score), 6)
        return round(float(score), 6)

    def _build_real_evique_index(self, *, workdir: Path, video_path: Path) -> None:
        from evique.builder import build_evique_from_segments
        from evique.standalone_base_builder import build_evique_standalone_base

        dataset_name = video_path.stem
        base_dir = workdir / "base"
        with _temporary_env({"EVIQUE_VISUAL_MODE": "hybrid", "EVIQUE_VIDEO_PATH": str(video_path)}):
            base_result = build_evique_standalone_base(
                video_paths=[video_path],
                output_base_dir=base_dir,
                dataset_name=dataset_name,
                chunk_token_size=1200,
                fine_num_frames=15,
                rough_num_frames=15,
                segment_length=30,
                rebuild=not self.context.reuse_index,
                model_name=os.getenv("OPENAI_MODEL"),
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
                embedding_dim=None,
            )
            manifest = build_evique_from_segments(
                base_result["video_segments"],
                video_paths=[video_path],
                video_path_map=base_result.get("video_path_map"),
                question_records=[],
                output_dir=workdir,
                event_window_seconds=120,
                track_gap_seconds=120,
                visual_field=None,
            )
        self.logger.info("build_index evique_base_dir=%s", base_dir)
        self.logger.info("build_index evique_base_generated=%s", base_result.get("generated"))
        self.logger.info("build_index evique_manifest_keys=%s", sorted(manifest.keys()))

    def _get_retriever(self):
        if self._retriever is None:
            os.environ.setdefault("EVIQUE_DISABLE_VISUAL_RELATIONS", "0")
            from evique.retriever import EvidenceRetriever

            self._retriever = EvidenceRetriever(
                Path(self.index_dir),
                max_evidence=max(self.context.top_k, 18),
            )
        return self._retriever

    def _recover_window(self, item: dict[str, Any], retriever) -> tuple[tuple[float, float] | None, str | None]:
        direct = self._start_end_from_sources(item, item.get("metadata"), item.get("provenance"), item.get("record"))
        if direct:
            return direct, None

        timestamp = self._timestamp_from_sources(item, item.get("metadata"), item.get("provenance"), item.get("record"))
        if timestamp is not None:
            return self._window_from_timestamp(timestamp), "window_recovered_from_point_timestamp_using_window_size"

        segment_ids = self._segment_ids_from_sources(item, item.get("metadata"), item.get("provenance"), item.get("record"))
        if segment_ids:
            starts: list[float] = []
            ends: list[float] = []
            for segment_id in segment_ids:
                scope = getattr(retriever, "scopes_by_segment", {}).get(segment_id)
                window = self._start_end_from_sources(scope, scope.get("metadata") if isinstance(scope, dict) else None, scope.get("provenance") if isinstance(scope, dict) else None)
                if window is None:
                    window = self._base_segment_time_index_lookup(segment_id)
                if window:
                    starts.append(window[0])
                    ends.append(window[1])
            if starts and ends:
                return (min(starts), max(ends)), "window_recovered_from_segment_id"

        return None, "missing_recoverable_time_window"

    def _start_end_from_sources(self, *sources: Any) -> tuple[float, float] | None:
        for source in _flatten_sources(sources):
            start = _first_number(
                source,
                ["start_time", "start", "segment_start", "source_start", "begin_time", "timestamp_start"],
            )
            end = _first_number(
                source,
                ["end_time", "end", "segment_end", "source_end", "stop_time", "timestamp_end"],
            )
            if start is None or end is None:
                parsed = _parse_time_range(source.get("time"))
                if parsed:
                    start = parsed[0] if start is None else start
                    end = parsed[1] if end is None else end
            if start is None and end is None:
                continue
            if start is None:
                start = max(0.0, float(end) - float(self.context.window_size))
            if end is None:
                end = float(start) + float(self.context.window_size)
            start = float(start)
            end = float(end)
            if end < start:
                start, end = end, start
            return start, end
        return None

    def _timestamp_from_sources(self, *sources: Any) -> float | None:
        for source in _flatten_sources(sources):
            timestamp = _first_number(source, ["timestamp", "timestamp_sec", "time", "frame_time"])
            if timestamp is not None:
                return float(timestamp)
        return None

    def _segment_ids_from_sources(self, *sources: Any) -> list[str]:
        values: list[str] = []

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
                return
            text = str(value)
            if text and text not in values:
                values.append(text)

        for source in _flatten_sources(sources):
            add(source.get("segment_id"))
            add(source.get("segment_ids"))
            add(source.get("related_segment_ids"))
            add(source.get("anchor_segment"))
        return values

    def _window_from_timestamp(self, timestamp: float) -> tuple[float, float]:
        half = max(0.0, float(self.context.window_size) / 2.0)
        return max(0.0, float(timestamp) - half), float(timestamp) + half

    def _base_segment_time_index_lookup(self, segment_id: str) -> tuple[float, float] | None:
        if self._base_segment_time_index is None:
            self._base_segment_time_index = self._load_base_segment_time_index()
        return self._base_segment_time_index.get(str(segment_id))

    def _load_base_segment_time_index(self) -> dict[str, tuple[float, float]]:
        index: dict[str, tuple[float, float]] = {}
        if not self.index_dir:
            return index
        for rel_path in BASE_SEGMENT_FILES:
            path = self.index_dir / rel_path
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.info("failed to load segment time index from %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                continue
            for video_name, segments in payload.items():
                if not isinstance(segments, dict):
                    continue
                for key, record in segments.items():
                    if not isinstance(record, dict):
                        continue
                    window = self._start_end_from_sources(record, record.get("metadata"))
                    if not window:
                        continue
                    aliases = {
                        str(key),
                        str(record.get("segment_id") or ""),
                        f"{video_name}_{key}",
                        f"{video_name}_{record.get('segment_id') or ''}",
                    }
                    for alias in aliases:
                        if alias:
                            index[alias] = window
            self.logger.info("loaded segment time index rows=%s from=%s", len(index), path)
            if index:
                break
        return index

    def _inspect_workdir(self, workdir: Path | None) -> dict[str, Any]:
        if not workdir:
            return {
                "workdir": "",
                "exists": False,
                "file_count": 0,
                "index_size_mb": 0.0,
                "required_found": [],
                "required_missing": list(REQUIRED_INDEX_FILES),
                "retrievable_found": [],
                "retrievable_missing": list(RETRIEVABLE_INDEX_FILES),
                "reusable": False,
            }
        workdir = Path(workdir)
        required_found = [name for name in REQUIRED_INDEX_FILES if (workdir / name).is_file()]
        required_missing = [name for name in REQUIRED_INDEX_FILES if name not in required_found]
        retrievable_found = [
            name
            for name in RETRIEVABLE_INDEX_FILES
            if (workdir / name).is_file() and (workdir / name).stat().st_size > 0
        ]
        retrievable_missing = [name for name in RETRIEVABLE_INDEX_FILES if name not in retrievable_found]
        file_count = _file_count(workdir)
        return {
            "workdir": str(workdir),
            "exists": workdir.exists(),
            "file_count": file_count,
            "index_size_mb": directory_size_mb(workdir),
            "required_found": required_found,
            "required_missing": required_missing,
            "retrievable_found": retrievable_found,
            "retrievable_missing": retrievable_missing,
            "reusable": bool(workdir.exists() and file_count > 0 and not required_missing and retrievable_found),
        }

    def _index_metadata(
        self,
        *,
        status: str,
        reason: str,
        start: float,
        before: dict[str, Any],
        after: dict[str, Any],
        created_files: list[str],
    ) -> dict[str, Any]:
        build_elapsed = time.perf_counter() - start
        self._log_build_snapshot(status, start, self.index_dir)
        return {
            "method": self.method,
            "index_dir": str(self.index_dir) if self.index_dir else "",
            "benchmark_index_dir": str(self.context.index_dir),
            "index_build_time_sec": round(build_elapsed, 6),
            "build_start_time": self._build_start_time,
            "build_elapsed_seconds": round(build_elapsed, 6),
            "current_workdir_file_count": after.get("file_count", 0),
            "current_workdir_size_mb": after.get("index_size_mb", 0.0),
            "key_files_generated": _key_file_status(self.index_dir),
            "index_size_mb": after.get("index_size_mb", 0.0),
            "status": status,
            "reason": reason,
            "reuse_evique_index": self.context.reuse_index,
            "build_index_requested": self.context.build_index_requested,
            "workdir_existed_before_build": before.get("exists", False),
            "workdir_file_count_before_build": before.get("file_count", 0),
            "workdir_file_count_after_build": after.get("file_count", 0),
            "index_size_mb_before_build": before.get("index_size_mb", 0.0),
            "index_size_mb_after_build": after.get("index_size_mb", 0.0),
            "required_index_files_found": after.get("required_found", []),
            "required_index_files_missing": after.get("required_missing", []),
            "retrievable_index_files_found": after.get("retrievable_found", []),
            "retrievable_index_files_missing": after.get("retrievable_missing", []),
            "created_files": created_files,
            "build_entrypoint": "evique.standalone_base_builder.build_evique_standalone_base + evique.builder.build_evique_from_segments",
            "visual_mode": "hybrid",
            "detector_model_env": os.getenv("EVIQUE_DETECTOR_MODEL", ""),
        }

    def _query_metadata(self, package: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = {
            "source": "evique",
            "workdir": str(self.index_dir) if self.index_dir else "",
            "dry_run": self.context.dry_run,
        }
        if package:
            metadata.update(
                {
                    "package_used_count": package.get("used_count"),
                    "view_order": package.get("view_order") or [],
                    "visual_used": bool(package.get("visual_used")),
                }
            )
        if extra:
            metadata.update(extra)
        return metadata

    def _evidence_type(self, item: dict[str, Any]) -> str:
        view = str(item.get("view") or "")
        source_views = set(str(value) for value in _sequence_values(item.get("source_views")))
        if view == "segment" and source_views & {"visual_object", "visual_track", "visual_relation", "visual_event"}:
            return "hybrid"
        if view in {"scope", "segment"}:
            return "caption"
        if view in {"target", "object"}:
            return "object"
        if view:
            return view
        return "hybrid"


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _flatten_sources(sources: Iterable[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        flattened.append(source)
        for key in ("metadata", "meta", "provenance", "record"):
            nested = source.get(key)
            if isinstance(nested, dict):
                flattened.append(nested)
    return flattened


def _sequence_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _extract_visual_intent(query_text: str) -> dict[str, Any]:
    lowered = query_text.lower()
    normalized = re.sub(r"\s+", " ", lowered.replace("_", " ").replace("-", " "))
    tokens = set(re.findall(r"[a-z][a-z0-9_-]*", lowered))
    labels: list[str] = []
    original_label_intents: list[str] = []
    specific_label_intents: list[str] = []
    explicit_traffic_light = any(phrase in normalized for phrase in TRAFFIC_LIGHT_PHRASES)
    traffic_object_intent = any(phrase in normalized for phrase in TRAFFIC_OBJECT_PHRASES)
    for label, terms in LABEL_TERMS.items():
        if tokens & terms:
            specific_label_intents.append(label)
            original_label_intents.append(label)
            labels.extend(_mapped_label_candidates(label))
    if explicit_traffic_light:
        specific_label_intents.append("traffic light")
        original_label_intents.append("traffic light")
        labels.append("traffic light")
    object_intent = bool(tokens & OBJECT_TERMS) or any(phrase in normalized for phrase in TRACKED_OBJECT_PHRASES)
    large_object = (
        any(phrase in normalized for phrase in ("large object", "large objects", "big object", "big objects"))
        or (bool(tokens & {"large", "big"}) and bool(tokens & OBJECT_TERMS))
    )
    center_object = bool(tokens & {"center", "centre", "middle"})
    if specific_label_intents:
        pass
    elif traffic_object_intent:
        original_label_intents.append("traffic_object")
        labels.extend(_mapped_label_candidates("traffic_object"))
    elif tokens & VEHICLE_TERMS:
        original_label_intents.append("vehicle")
        labels.extend(_mapped_label_candidates("vehicle"))
    elif object_intent or large_object or center_object:
        labels.extend(MOVABLE_LABELS)
    if _is_counting_query(lowered) and not labels:
        labels.extend(MOVABLE_LABELS)
    labels = [label for label in dict.fromkeys(labels) if label]
    original_label_intents = [label for label in dict.fromkeys(original_label_intents) if label]
    specific_label_intents = [label for label in dict.fromkeys(specific_label_intents) if label]
    attributes = sorted(tokens & ATTRIBUTE_TERMS)
    if "grey" in attributes and "gray" not in attributes:
        attributes.append("gray")
    return {
        "labels": labels,
        "original_label_intents": original_label_intents,
        "specific_label_intents": specific_label_intents,
        "specific_label_query": bool(specific_label_intents),
        "traffic_object": traffic_object_intent and not bool(specific_label_intents),
        "attributes": attributes,
        "counting": _is_counting_query(lowered),
        "large_object": large_object,
        "center_object": center_object,
        "motion": _is_motion_query(lowered),
    }


def _mapped_label_candidates(label: str) -> list[str]:
    return list(LABEL_INTENT_CANDIDATES.get(label, [label]))


def _is_counting_query(lowered: str) -> bool:
    normalized = re.sub(r"\s+", " ", lowered)
    return any(term in normalized for term in COUNTING_TERMS)


def _is_motion_query(lowered: str) -> bool:
    tokens = set(re.findall(r"[a-z][a-z0-9_-]*", lowered))
    return bool(tokens & MOTION_TERMS)


def _deterministic_operator_type(query: DBQuery, intent: dict[str, Any]) -> str:
    query_type = re.sub(r"[^a-z0-9]+", "_", str(query.type or "").strip().lower()).strip("_")
    if query_type in {"existence", "counting", "attribute_size", "spatial", "motion"}:
        return query_type
    if query_type in {"size", "large", "attribute"}:
        return "attribute_size"
    if query_type in {"center", "spatial_center", "spatial_relation"}:
        return "spatial"
    if query_type in {"trajectory", "motion_trajectory"}:
        return "motion"
    text = str(query.query or "").lower()
    normalized = re.sub(r"\s+", " ", text)
    if intent.get("motion"):
        return "motion"
    if intent.get("counting") or re.search(r"\b(multiple|several|many|at least)\b", normalized):
        return "counting"
    if intent.get("large_object") or re.search(r"\b(large|largest|big|biggest)\b", normalized):
        return "attribute_size"
    if intent.get("center_object") or re.search(r"\b(center|centre|middle|closest)\b", normalized):
        return "spatial"
    if re.search(r"\b(visible|appears|appear|shown|present)\b", normalized):
        return "existence"
    return "existence"


def _operator_source_score(source_view: str) -> float:
    return {
        "visual_object": 4.0,
        "visual_track": 4.0,
        "visual_event": 3.0,
        "adaptive_event": 1.0,
        "event": 0.5,
        "scope": 0.25,
    }.get(str(source_view), 0.0)


def _label_match_exactness(intent: dict[str, Any], label: str) -> float:
    original = set(intent.get("original_label_intents") or [])
    specific = set(intent.get("specific_label_intents") or [])
    label = str(label or "")
    if "van" in original and label != "van":
        return 0.25
    if specific:
        return 1.0 if label in specific else 0.0
    return 1.0


def _record_confidence(row: dict[str, Any]) -> float:
    for key in ("score", "confidence", "detector_score", "object_confidence"):
        value = safe_float(row.get(key), None)
        if value is not None:
            return float(value)
    return 0.0


def _record_track_support(row: dict[str, Any]) -> int:
    for key in ("object_ids", "frame_ids", "timestamps", "compact_points", "bbox_sequence"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    return 1 if _operator_track_id(row) else 0


def _record_track_area(row: dict[str, Any]) -> float | None:
    areas: list[float] = []
    for key in ("bbox_sequence", "compact_points"):
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    bbox = _bbox_xyxy_from_value(item.get("bbox") or item.get("box") or item.get("bbox_xyxy"))
                    if bbox is not None:
                        left, top, right, bottom = bbox
                        area = abs(float(right) - float(left)) * abs(float(bottom) - float(top))
                        if area > 0.0:
                            areas.append(area)
    if not areas:
        return None
    return sum(areas) / len(areas)


def _operator_track_id(row: dict[str, Any]) -> str:
    return str(row.get("track_id") or row.get("actor_track_id") or row.get("target_track_id") or row.get("id") or "")


def _windows_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(float(left[0]), float(right[0])) < min(float(left[1]), float(right[1]))


def _result_window_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(row.get("start_time") or 0.0), 3),
        round(float(row.get("end_time") or 0.0), 3),
    )


def _deterministic_evidence_text(
    operator: str,
    label: str,
    source_view: str,
    start_time: float,
    end_time: float,
    item: dict[str, Any],
) -> str:
    if operator == "counting":
        count = item.get("track_count") or item.get("object_count") or 0
        return f"Deterministic counting match: {count} {label} objects in {float(start_time):.2f}s-{float(end_time):.2f}s."
    if operator == "attribute_size":
        return f"Deterministic large-object match: large {label} from {source_view} in {float(start_time):.2f}s-{float(end_time):.2f}s."
    if operator == "spatial":
        return f"Deterministic spatial-center match: {label} near center from {source_view} in {float(start_time):.2f}s-{float(end_time):.2f}s."
    if operator == "motion":
        return f"Deterministic motion match: {label} moves across from {source_view} in {float(start_time):.2f}s-{float(end_time):.2f}s."
    return f"Deterministic existence match: {label} visible from {source_view} in {float(start_time):.2f}s-{float(end_time):.2f}s."


def _record_label(row: dict[str, Any]) -> str:
    labels = _record_label_candidates(row)
    return labels[0] if labels else ""


def _record_label_candidates(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for value in _iter_label_like_values(row):
        label = _canonical_visual_label(str(value or ""))
        if label:
            labels.append(label)
    return [label for label in dict.fromkeys(labels) if label]


def _first_matching_label(row: dict[str, Any], label_intents: set[str]) -> str | None:
    if not label_intents:
        return None
    for label in _record_label_candidates(row):
        if label in label_intents:
            return label
    return None


def _synonym_fallback_metadata(intent: dict[str, Any], label_match: str | None) -> dict[str, Any]:
    if "van" in intent.get("original_label_intents", []) and label_match and label_match != "van":
        return {
            "synonym_fallback_used": True,
            "original_label_intent": "van",
            "mapped_label_candidates": list(VAN_SYNONYM_METADATA_CANDIDATES),
        }
    return {}


def _native_source_view(item: dict[str, Any]) -> str:
    views = [str(item.get("view") or "")]
    views.extend(str(value) for value in _sequence_values(item.get("source_views")))
    for preferred in ("visual_object", "visual_track", "visual_event", "visual_relation"):
        if preferred in views:
            return preferred
    for preferred in ("adaptive_event", "event", "scope", "segment", "target", "object"):
        if preferred in views:
            return preferred
    return next((view for view in views if view), "")


def _native_sort_priority(
    *,
    source_view: str,
    evidence_type: str,
    label_matched: bool,
    window_clipped: bool,
    motion_required: bool = False,
    motion_verified: bool = False,
) -> int:
    if motion_required and label_matched and motion_verified and source_view in {"visual_event", "visual_track"}:
        return 1
    if label_matched and source_view in {"visual_event", "visual_track", "visual_object"}:
        return 2 if motion_required and not motion_verified else 1
    if label_matched:
        return 5 if motion_required and not motion_verified else 2
    if source_view in {"counting", "co_occurrence"} or evidence_type in {"counting", "co_occurrence"}:
        return 3
    if source_view in {"adaptive_event", "event", "scope", "segment"}:
        return 4
    if source_view == "visual_track" and window_clipped:
        return 5
    return 4


def _result_sort_priority(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("db_sort_priority") is not None:
        try:
            return int(metadata.get("db_sort_priority"))
        except (TypeError, ValueError):
            pass
    source_view = str(metadata.get("source_view") or "")
    label_matched = bool(metadata.get("label_match"))
    evidence_type = str(row.get("evidence_type") or "")
    if label_matched and source_view in {"visual_event", "visual_track", "visual_object"}:
        return 1
    if label_matched:
        return 2
    if source_view in {"counting", "co_occurrence"} or evidence_type in {"counting", "co_occurrence", "visual_object_fallback"}:
        return 3
    if source_view in {"adaptive_event", "event", "scope", "segment"}:
        return 4
    if source_view == "visual_track" and metadata.get("window_clipped"):
        return 5
    return 4


def _result_sort_key(row: dict[str, Any]) -> tuple:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    priority = _result_sort_priority(row)
    native_rank = safe_float(metadata.get("native_original_rank"), None)
    fallback_rank = safe_float(row.get("rank"), None)
    sequence = native_rank if native_rank is not None else fallback_rank if fallback_rank is not None else 1_000_000.0
    start = safe_float(row.get("start_time"), 0.0) or 0.0
    score = safe_float(row.get("score"), 0.0) or 0.0
    return (priority, sequence, -float(score), float(start))


def _iter_label_like_values(row: dict[str, Any]) -> Iterable[Any]:
    sources: list[dict[str, Any]] = [row]
    for key in ("metadata", "provenance"):
        value = row.get(key)
        if isinstance(value, dict):
            sources.append(value)
    for source in sources:
        for field in LABEL_LIKE_FIELDS:
            yield from _expand_label_like_value(source.get(field))


def _expand_label_like_value(value: Any) -> Iterable[Any]:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _expand_label_like_value(item)
        return
    yield value


def _canonical_visual_label(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower().replace("_", " ").replace("-", " "))
    if not normalized:
        return ""
    if any(phrase in normalized for phrase in TRAFFIC_LIGHT_PHRASES):
        return "traffic light"
    tokens = set(re.findall(r"[a-z][a-z0-9_-]*", normalized))
    for label, terms in LABEL_TERMS.items():
        if tokens & terms or normalized in terms:
            return label
    if tokens & VEHICLE_TERMS:
        return "vehicle"
    return normalized


def _record_id(row: dict[str, Any]) -> str:
    return str(
        row.get("id")
        or row.get("event_id")
        or row.get("track_id")
        or row.get("object_id")
        or row.get("node_id")
        or row.get("frame_id")
        or ""
    )


def _record_matches_attribute(row: dict[str, Any], attributes: list[str]) -> bool:
    if not attributes:
        return True
    normalized_attrs = {"gray" if value == "grey" else value for value in attributes}
    row_values: set[str] = set()
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance")]):
        for key in ("color", "colour", "color_majority", "dominant_color", "dominant_colour"):
            _add_attribute_value(row_values, source.get(key))
        for value in source.get("attributes") or []:
            _add_attribute_value(row_values, value)
    return normalized_attrs.issubset(row_values)


def _record_motion_verified(row: dict[str, Any]) -> bool:
    if _record_stationary_motion(row):
        return False
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance"), row.get("record")]):
        direction = str(source.get("direction_text") or source.get("direction") or "").strip().lower()
        if direction and direction not in {"unknown", "stationary", "none", "n/a"}:
            return True
        event_type = str(source.get("event_type") or source.get("type") or "").strip().lower()
        if event_type.startswith("move_") or event_type in {"move", "moving", "movement"}:
            return True
        motion_summary = str(source.get("motion_summary") or source.get("motion") or "").strip().lower()
        if motion_summary and "stationary" not in motion_summary and "unknown" not in motion_summary:
            if re.search(r"\b(move|moves|moving|travel|travels|cross|across)\b", motion_summary):
                return True
        speed = safe_float(source.get("speed_proxy") or source.get("speed"), None)
        if speed is not None and float(speed) > 0.0:
            return True
    displacement = _record_track_displacement(row)
    return displacement is not None and displacement >= 12.0


def _record_stationary_motion(row: dict[str, Any]) -> bool:
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance"), row.get("record")]):
        direction = str(source.get("direction_text") or source.get("direction") or "").strip().lower()
        if direction == "stationary":
            return True
        motion_summary = str(source.get("motion_summary") or source.get("motion") or "").strip().lower()
        if "stationary" in motion_summary:
            return True
    return False


def _record_track_displacement(row: dict[str, Any]) -> float | None:
    centers: list[tuple[float, float]] = []
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance"), row.get("record")]):
        for key in ("start_center", "end_center"):
            center = _point_xy_from_value(source.get(key))
            if center is not None:
                centers.append(center)
        for key in ("compact_points", "bbox_sequence"):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    center = _center_from_track_point(item)
                    if center is not None:
                        centers.append(center)
    if len(centers) < 2:
        center = _record_bbox_center(row)
        if center is not None:
            centers.append(center)
    if len(centers) < 2:
        return None
    first = centers[0]
    last = centers[-1]
    return ((float(last[0]) - float(first[0])) ** 2 + (float(last[1]) - float(first[1])) ** 2) ** 0.5


def _center_from_track_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        for key in ("center", "bbox_center", "bbox_center_xy"):
            center = _point_xy_from_value(value.get(key))
            if center is not None:
                return center
        bbox = _bbox_xyxy_from_value(value.get("bbox") or value.get("box") or value.get("bbox_xyxy"))
        if bbox is not None:
            left, top, right, bottom = bbox
            return (float(left) + float(right)) / 2.0, (float(top) + float(bottom)) / 2.0
    return _point_xy_from_value(value)


def _add_attribute_value(row_values: set[str], value: Any) -> None:
    if value is None:
        return
    normalized = "gray" if str(value).strip().lower() == "grey" else str(value).strip().lower()
    if not normalized:
        return
    row_values.add(normalized)
    for token in re.findall(r"[a-z][a-z0-9_-]*", normalized):
        row_values.add("gray" if token == "grey" else token)


def _record_bbox(row: dict[str, Any]) -> Any:
    for key in ("bbox", "bbox_xyxy", "box"):
        value = row.get(key)
        if value:
            return value
    if all(row.get(key) is not None for key in ("left", "top", "right", "bottom")):
        return [row.get("left"), row.get("top"), row.get("right"), row.get("bottom")]
    sequence = row.get("bbox_sequence") or row.get("compact_points") or []
    if isinstance(sequence, list):
        for item in sequence:
            if isinstance(item, dict) and item.get("bbox"):
                return item.get("bbox")
    return None


def _record_bbox_xyxy(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _bbox_xyxy_from_value(_record_bbox(row))


def _bbox_xyxy_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        nested = value.get("bbox") or value.get("box") or value.get("bbox_xyxy")
        if nested is not None and nested is not value:
            parsed = _bbox_xyxy_from_value(nested)
            if parsed is not None:
                return parsed
        key_sets = [
            ("left", "top", "right", "bottom"),
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
        ]
        for keys in key_sets:
            numbers = [safe_float(value.get(key), None) for key in keys]
            if all(number is not None for number in numbers):
                return tuple(float(number) for number in numbers)  # type: ignore[return-value]
        x = safe_float(value.get("x"), None)
        y = safe_float(value.get("y"), None)
        width = safe_float(value["width"] if "width" in value else value.get("w"), None)
        height = safe_float(value["height"] if "height" in value else value.get("h"), None)
        if x is not None and y is not None and width is not None and height is not None:
            return float(x), float(y), float(x) + float(width), float(y) + float(height)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        numbers = [safe_float(item, None) for item in value[:4]]
        if all(number is not None for number in numbers):
            return tuple(float(number) for number in numbers)  # type: ignore[return-value]
    return None


def _record_bbox_area(row: dict[str, Any]) -> float | None:
    bbox = _record_bbox_xyxy(row)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width = abs(float(right) - float(left))
    height = abs(float(bottom) - float(top))
    if width <= 0.0 or height <= 0.0:
        return None
    return width * height


def _record_center_priority_score(
    row: dict[str, Any],
    *,
    frame_size: tuple[float, float] | None = None,
) -> float | None:
    center = _record_bbox_center(row)
    if center is None:
        return None
    frame = frame_size or _record_frame_size(row)
    if frame is None:
        bbox = _record_bbox_xyxy(row)
        if bbox is None:
            return None
        max_coord = max(abs(bbox[0]), abs(bbox[1]), abs(bbox[2]), abs(bbox[3]))
        if max_coord <= 1.5:
            frame = (1.0, 1.0)
        else:
            return None
    width, height = frame
    if width <= 0.0 or height <= 0.0:
        return None
    frame_center_x = width / 2.0
    frame_center_y = height / 2.0
    max_distance_sq = frame_center_x * frame_center_x + frame_center_y * frame_center_y
    if max_distance_sq <= 0.0:
        return None
    distance_sq = (float(center[0]) - frame_center_x) ** 2 + (float(center[1]) - frame_center_y) ** 2
    return max(0.0, 1.0 - (distance_sq / max_distance_sq))


def _record_bbox_center(row: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("bbox_center", "center", "bbox_center_xy"):
        value = row.get(key)
        parsed = _point_xy_from_value(value)
        if parsed is not None:
            return parsed
    bbox = _record_bbox_xyxy(row)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (float(left) + float(right)) / 2.0, (float(top) + float(bottom)) / 2.0


def _point_xy_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        x_value = value["x"] if "x" in value else value["cx"] if "cx" in value else value.get("center_x")
        y_value = value["y"] if "y" in value else value["cy"] if "cy" in value else value.get("center_y")
        x = safe_float(x_value, None)
        y = safe_float(y_value, None)
        if x is not None and y is not None:
            return float(x), float(y)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = safe_float(value[0], None)
        y = safe_float(value[1], None)
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def _record_frame_size(row: dict[str, Any]) -> tuple[float, float] | None:
    explicit_pairs = [
        ("frame_width", "frame_height"),
        ("image_width", "image_height"),
        ("video_width", "video_height"),
        ("source_width", "source_height"),
    ]
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance")]):
        for width_key, height_key in explicit_pairs:
            width = safe_float(source.get(width_key), None)
            height = safe_float(source.get(height_key), None)
            if width is not None and height is not None and width > 0.0 and height > 0.0:
                return float(width), float(height)
        size = source.get("frame_size") or source.get("image_size") or source.get("video_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width = safe_float(size[0], None)
            height = safe_float(size[1], None)
            if width is not None and height is not None and width > 0.0 and height > 0.0:
                return float(width), float(height)
        generic_width = safe_float(source.get("width"), None)
        generic_height = safe_float(source.get("height"), None)
        if generic_width is not None and generic_height is not None and generic_width > 0.0 and generic_height > 0.0:
            bbox = _record_bbox_xyxy(row)
            if bbox is None or (generic_width >= max(bbox[0], bbox[2]) and generic_height >= max(bbox[1], bbox[3])):
                return float(generic_width), float(generic_height)
    return None


def _fallback_evidence_text(
    label: str,
    kind: str,
    row: dict[str, Any],
    start_time: float,
    end_time: float,
    item: dict[str, Any],
) -> str:
    representative = item.get("representative_timestamp")
    if kind == "track" and item.get("window_mode") == "track_representative_window":
        if representative is not None:
            return f"Fallback match: {label} track representative window around {float(representative):.1f}s."
        midpoint = (float(start_time) + float(end_time)) / 2.0
        return f"Fallback match: {label} track representative window around {midpoint:.1f}s."
    timestamp_text = _fallback_time_text(row, start_time, end_time)
    return f"Fallback match: {label} {kind} at {timestamp_text}."


def _fallback_time_text(row: dict[str, Any], start_time: float, end_time: float) -> str:
    timestamp = None
    for source in _flatten_sources([row, row.get("metadata"), row.get("provenance")]):
        timestamp = _first_number(source, ["timestamp", "timestamp_sec", "time", "frame_time"])
        if timestamp is not None:
            break
    if timestamp is not None:
        return f"timestamp {float(timestamp):.2f}s"
    return f"{float(start_time):.2f}s-{float(end_time):.2f}s"


def _representative_track_timestamps(row: dict[str, Any], start_time: float, end_time: float, *, limit: int = 2) -> list[float]:
    candidates: list[float] = []
    candidates.extend(_timestamp_list_values(row.get("compact_points")))
    candidates.extend(_timestamp_list_values(row.get("timestamps")))
    candidates.extend(_timestamp_list_values(row.get("bbox_sequence")))
    if not candidates:
        midpoint = (float(start_time) + float(end_time)) / 2.0
        candidates = [float(start_time), midpoint, float(end_time)]
    valid = sorted({float(value) for value in candidates if float(start_time) <= float(value) <= float(end_time)})
    if not valid:
        valid = [float(start_time)]
    if len(valid) <= limit:
        return valid
    midpoint = (float(start_time) + float(end_time)) / 2.0
    ranked = sorted(valid, key=lambda value: (abs(value - midpoint), value))
    selected = sorted(ranked[:limit])
    return selected


def _select_representative_timestamps(values: list[float], start_time: float, end_time: float, *, limit: int = 2) -> list[float]:
    valid = sorted({float(value) for value in values if float(start_time) <= float(value) <= float(end_time)})
    if not valid:
        return []
    if len(valid) <= limit:
        return valid
    midpoint = (float(start_time) + float(end_time)) / 2.0
    ranked = sorted(valid, key=lambda value: (abs(value - midpoint), value))
    return sorted(ranked[: max(1, int(limit))])


def _timestamp_list_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    timestamps: list[float] = []
    for value in values:
        if isinstance(value, dict):
            timestamp = safe_float(value.get("timestamp") or value.get("time") or value.get("frame_time"), None)
        else:
            timestamp = safe_float(value, None)
        if timestamp is not None:
            timestamps.append(float(timestamp))
    return timestamps


def _record_timestamp_values(row: dict[str, Any], *, include_record: bool = False) -> list[float]:
    sources: list[Any] = [row, row.get("metadata"), row.get("provenance")]
    if include_record:
        sources.append(row.get("record"))
    values: list[float] = []
    for source in _flatten_sources(sources):
        point = _first_number(source, ["timestamp", "timestamp_sec", "time", "frame_time"])
        if point is not None:
            values.append(float(point))
        values.extend(_timestamp_list_values(source.get("compact_points")))
        values.extend(_timestamp_list_values(source.get("timestamps")))
        values.extend(_timestamp_list_values(source.get("bbox_sequence")))
    return sorted({float(value) for value in values})


def _timestamp_list_window(values: Any) -> tuple[float, float] | None:
    timestamps = _timestamp_list_values(values)
    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def _points_window(values: Any) -> tuple[float, float] | None:
    timestamps = _timestamp_list_values(values)
    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def _first_number(source: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = safe_float(source.get(key), None)
        if value is not None:
            return value
    return None


def _parse_time_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = safe_float(value[0], None)
        end = safe_float(value[1], None)
        if start is not None and end is not None:
            return float(start), float(end)
    text = str(value).strip()
    for sep in ("-", ",", "to"):
        if sep not in text:
            continue
        left, right = text.split(sep, 1)
        start = safe_float(left.strip(), None)
        end = safe_float(right.strip(), None)
        if start is not None and end is not None:
            return float(start), float(end)
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _file_count(path: Path) -> int:
    if not path or not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _snapshot_files(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.is_file():
        return {path.name}
    return {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}


def _created_files(before: set[str], after: set[str]) -> list[str]:
    return sorted(after - before)


def _key_file_status(workdir: Path | None) -> dict[str, bool]:
    if not workdir:
        return {name: False for name in BUILD_PROGRESS_FILES}
    workdir = Path(workdir)
    return {
        name: (workdir / name).is_file() and (workdir / name).stat().st_size > 0
        for name in BUILD_PROGRESS_FILES
    }


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
