from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from db_benchmark.schema import DBQuery, empty_timing, make_result_record
from db_benchmark.utils import directory_size_mb, ensure_dir, setup_method_logger


@dataclass
class AdapterContext:
    root: Path
    output_base: Path
    index_dir: Path
    result_path: Path
    log_path: Path
    video_path: Path | None
    evique_workdir: Path | None
    top_k: int
    window_size: float
    stride: float
    dry_run: bool = False
    reuse_index: bool = False
    build_index_requested: bool = False
    progress: bool = False


class BaseDBAdapter:
    method = "BASE"
    result_stem = "base"
    implementation_fidelity = "unknown"
    adapter_status = "unknown"
    reason = ""

    def __init__(self, context: AdapterContext, spec: dict[str, Any] | None = None):
        self.context = context
        self.spec = spec or {}
        self.method = str(self.spec.get("canonical_name") or self.method)
        self.result_stem = str(self.spec.get("result_stem") or self.result_stem)
        self.implementation_fidelity = str(
            self.spec.get("implementation_fidelity") or self.implementation_fidelity
        )
        self.adapter_status = str(self.spec.get("adapter_status") or self.adapter_status)
        self.reason = str(self.spec.get("reason") or self.reason)
        self.logger = setup_method_logger(self.result_stem, context.log_path)

    def build_index(self) -> dict[str, Any]:
        ensure_dir(self.context.index_dir)
        start = time.perf_counter()
        elapsed = time.perf_counter() - start
        return {
            "method": self.method,
            "index_dir": str(self.context.index_dir),
            "index_build_time_sec": round(elapsed, 6),
            "index_size_mb": directory_size_mb(self.context.index_dir),
            "status": "noop",
            "reason": "adapter does not require a benchmark-managed index",
        }

    def run_query(self, query: DBQuery) -> list[dict[str, Any]]:
        return [
            self.status_record(
                query,
                status="unsupported",
                reason=self.reason or "adapter not implemented",
            )
        ]

    def status_record(
        self,
        query: DBQuery,
        *,
        status: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
        implementation_fidelity: str | None = None,
        adapter_status: str | None = None,
    ) -> dict[str, Any]:
        return make_result_record(
            query,
            method=self.method,
            rank=None,
            status=status,
            reason=reason,
            implementation_fidelity=implementation_fidelity or self.implementation_fidelity,
            adapter_status=adapter_status or self.adapter_status,
            timing=timing or empty_timing(),
            metadata=metadata or {},
        )

    def ok_record(
        self,
        query: DBQuery,
        *,
        rank: int,
        start_time: float,
        end_time: float,
        score: float | None,
        evidence_type: str,
        evidence_text: str,
        bbox: Any = None,
        track_id: str | None = None,
        timing: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        implementation_fidelity: str | None = None,
        adapter_status: str | None = None,
    ) -> dict[str, Any]:
        return make_result_record(
            query,
            method=self.method,
            rank=rank,
            status="ok",
            start_time=float(start_time),
            end_time=float(end_time),
            score=score,
            bbox=bbox,
            track_id=track_id,
            evidence_type=evidence_type,
            evidence_text=evidence_text,
            implementation_fidelity=implementation_fidelity or self.implementation_fidelity,
            adapter_status=adapter_status or self.adapter_status,
            timing=timing or empty_timing(),
            metadata=metadata or {},
        )

    def exception_record(self, query: DBQuery, exc: BaseException) -> dict[str, Any]:
        self.logger.exception("adapter failed for query_id=%s", query.query_id)
        return self.status_record(
            query,
            status="adapter_error",
            reason=f"{type(exc).__name__}: {exc}",
            metadata={"exception_type": type(exc).__name__},
        )
