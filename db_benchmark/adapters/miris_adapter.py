from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from db_benchmark.adapters.miris_format_converter import build_miris_query_input
from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter
from db_benchmark.schema import DBQuery


class MIRISProxyAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "miris"
    declared_method = "MIRIS"
    declared_source_kind = "official_proxy"
    declared_source_paths = (
        "third_party/external/official/MIRIS/miris-master",
        "db_benchmark/adapters/miris_format_converter.py",
        "third_party/proxy/PROXY_POLICY.md",
    )


class MIRISOfficialAdapter(BaseDBAdapter):
    implementation_fidelity = "official_adapted"
    adapter_status = "official_adapted_smoke_runnable"

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        self.source_dir = (
            self.context.root
            / "DB_Baselines"
            / "third_party_baselines"
            / "official"
            / "MIRIS"
            / "miris-master"
        )
        self.converter_path = self.context.root / "DB_Baselines" / "adapters" / "miris_format_converter.py"
        self.raw_base = self.context.output_base / "baseline_raw" / "miris"
        self.raw_input_dir = self.raw_base / "input"
        self.raw_output_dir = self.raw_base / "output"
        self.converted_output_path = self.context.result_path

    def build_index(self) -> dict:
        metadata = super().build_index()
        metadata.update(self._base_metadata())
        metadata["status"] = "present" if metadata["official_code_present"] else "not_available"
        metadata["reason"] = "" if metadata["official_code_present"] else "MIRIS official source directory was not found"
        return metadata

    def run_query(self, query: DBQuery) -> list[dict]:
        start = time.perf_counter()
        mapping_payload = self._write_original_input(query)
        query_mapping = mapping_payload["query_mapping"]
        metadata = self._base_metadata(query)
        metadata.update(
            {
                "original_input_path": mapping_payload["original_input_path"],
                "query_mapping": query_mapping,
                "supported_query_type": query_mapping.get("supported_query_type", ""),
                "unsupported_reason": query_mapping.get("unsupported_reason", ""),
            }
        )

        if not metadata["official_code_present"]:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="MIRIS official source directory was not found",
                    adapter_status="not_available",
                    metadata=metadata,
                    timing=self._timing(start),
                    implementation_fidelity="official_adapted",
                )
            ]

        if not query_mapping.get("supported"):
            return [
                self.status_record(
                    query,
                    status="unsupported",
                    reason=query_mapping.get("unsupported_reason") or "Query type is not supported by official MIRIS predicates",
                    adapter_status="unsupported_query_type",
                    metadata=metadata,
                    timing=self._timing(start),
                    implementation_fidelity="official_adapted",
                )
            ]

        predicate = str(query_mapping["miris_predicate"])
        plan_path = self._find_plan(predicate)
        plan = self._read_plan(plan_path)
        missing_artifacts = self._missing_artifacts(predicate, plan)
        go_path = shutil.which("go")
        if not go_path or not plan_path or missing_artifacts:
            metadata.update(self._training_required_metadata(predicate, plan_path, missing_artifacts, go_path))
            return [
                self.status_record(
                    query,
                    status="requires_training",
                    reason=metadata["blocking_reason"],
                    adapter_status="official_present_requires_training",
                    metadata=metadata,
                    timing=self._timing(start),
                    implementation_fidelity="official_adapted",
                )
            ]

        run_metadata = self._run_official_exec(query, predicate, plan_path, go_path)
        metadata.update(run_metadata)
        if run_metadata["return_code"] != 0:
            return [
                self.status_record(
                    query,
                    status="adapter_error",
                    reason="Official MIRIS runner failed",
                    adapter_status="official_runner_failed",
                    metadata=metadata,
                    timing=self._timing(start),
                    implementation_fidelity="official_adapted",
                )
            ]

        records = self._convert_final_to_records(query, Path(run_metadata["original_output_path"]), metadata, start)
        if not records:
            return [
                self.status_record(
                    query,
                    status="no_evidence",
                    reason="Official MIRIS final.json produced no tracks",
                    adapter_status="official_adapted_smoke_runnable",
                    metadata=metadata,
                    timing=self._timing(start),
                    implementation_fidelity="official_adapted",
                )
            ]
        return records

    def _write_original_input(self, query: DBQuery) -> dict[str, Any]:
        query_payload = {
            "query_id": query.query_id,
            "query": query.query,
            "type": query.type,
            "dataset": query.dataset,
            "video_id": query.video_id,
            "metadata": query.metadata,
        }
        return build_miris_query_input(
            query=query_payload,
            dataset=query.dataset,
            video_id=query.video_id,
            video_path=str(self.context.video_path) if self.context.video_path else "",
            output_dir=self.raw_input_dir,
        )

    def _run_official_exec(self, query: DBQuery, predicate: str, plan_path: Path, go_path: str) -> dict[str, Any]:
        plan = self._read_plan(plan_path)
        official_output_path = plan_path.parent / "final.json"
        query_output_dir = self.raw_output_dir / query.query_id
        query_output_dir.mkdir(parents=True, exist_ok=True)
        copied_output_path = query_output_dir / "final.json"
        cmd = [go_path, "run", "exec.go", predicate, str(plan_path)]
        completed = subprocess.run(
            cmd,
            cwd=self.source_dir,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode == 0 and official_output_path.exists():
            shutil.copyfile(official_output_path, copied_output_path)
            original_output_path = copied_output_path
        else:
            original_output_path = official_output_path
        return {
            "runner_command": cmd,
            "return_code": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "plan_path": str(plan_path),
            "plan": plan,
            "original_output_path": str(original_output_path),
            "converted_output_path": str(self.converted_output_path),
            "official_repo_output_path": str(official_output_path),
        }

    def _convert_final_to_records(
        self,
        query: DBQuery,
        final_path: Path,
        metadata: dict[str, Any],
        start: float,
    ) -> list[dict]:
        if not final_path.exists():
            return []
        detections_by_frame = json.loads(final_path.read_text(encoding="utf-8"))
        tracks: dict[str, list[dict[str, Any]]] = {}
        if isinstance(detections_by_frame, list):
            for frame_items in detections_by_frame:
                if not isinstance(frame_items, list):
                    continue
                for det in frame_items:
                    if not isinstance(det, dict):
                        continue
                    track_id = str(det.get("track_id", ""))
                    if not track_id or track_id == "-1":
                        continue
                    tracks.setdefault(track_id, []).append(det)
        fps = _query_fps(query)
        ranked = sorted(
            tracks.items(),
            key=lambda item: (-len(item[1]), min(int(det.get("frame_idx", 0)) for det in item[1]), item[0]),
        )
        records = []
        for rank, (track_id, detections) in enumerate(ranked[: self.context.top_k], start=1):
            frame_indexes = [int(det.get("frame_idx", 0)) for det in detections]
            start_time = min(frame_indexes) / fps
            end_time = (max(frame_indexes) + 1) / fps
            per_record_metadata = dict(metadata)
            per_record_metadata.update(
                {
                    "miris_track_detection_count": len(detections),
                    "miris_frame_start": min(frame_indexes),
                    "miris_frame_end": max(frame_indexes),
                    "fps_for_time_conversion": fps,
                    "training_required": False,
                    "unsupported_reason": "",
                }
            )
            records.append(
                self.ok_record(
                    query,
                    rank=rank,
                    start_time=start_time,
                    end_time=end_time,
                    score=float(len(detections)),
                    bbox=_track_bbox(detections),
                    track_id=track_id,
                    evidence_type="official_miris_track",
                    evidence_text=f"MIRIS official track {track_id} for predicate {metadata['query_mapping'].get('miris_predicate')}.",
                    adapter_status="official_adapted_smoke_runnable",
                    implementation_fidelity="official_adapted",
                    timing=self._timing(start),
                    metadata=per_record_metadata,
                )
            )
        return records

    def _find_plan(self, predicate: str) -> Path | None:
        logs_dir = self.source_dir / "logs" / predicate
        if not logs_dir.exists():
            return None
        plans = sorted(logs_dir.glob("*/*/plan.json"))
        return plans[0] if plans else None

    def _read_plan(self, plan_path: Path | None) -> dict[str, Any]:
        if not plan_path or not plan_path.exists():
            return {}
        try:
            return json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _missing_artifacts(self, predicate: str, plan: dict[str, Any]) -> list[str]:
        if not plan:
            return ["plan.json"]
        freq = int(plan.get("Freq") or 0)
        dataset_name = _predicate_dataset(predicate)
        artifacts: list[Path] = [self.source_dir / "logs" / dataset_name / "gnn" / "model"]
        filter_plan = plan.get("Filter") if isinstance(plan.get("Filter"), dict) else {}
        refine_plan = plan.get("Refine") if isinstance(plan.get("Refine"), dict) else {}
        if filter_plan.get("Name") == "rnn":
            artifacts.append(self.source_dir / "logs" / predicate / str(freq) / "filter-rnn" / "model")
        if refine_plan.get("PSMethod") == "rnn":
            artifacts.append(self.source_dir / "logs" / predicate / str(freq) / "refine-rnn" / "model")
        return [str(path) for path in artifacts if not _artifact_exists(path)]

    def _training_required_metadata(
        self,
        predicate: str,
        plan_path: Path | None,
        missing_artifacts: list[str],
        go_path: str | None,
    ) -> dict[str, Any]:
        required = list(missing_artifacts)
        if not go_path:
            required.append("go executable")
        if not plan_path:
            required.append("logs/<predicate>/<freq>/<bound>/plan.json")
        blocking = (
            "Official MIRIS cannot run this supported predicate until the Go runtime, MIRIS plan, "
            "and trained GNN/RNN artifacts are available."
        )
        return {
            "training_required": True,
            "training_entrypoint": {
                "prepare_rnn": str(self.source_dir / "prepare_rnn.go"),
                "gnn_train": str(self.source_dir / "models" / "gnn" / "train.py"),
                "rnn_train": str(self.source_dir / "models" / "rnn" / "train.py"),
                "planner": str(self.source_dir / "plan.go"),
                "executor": str(self.source_dir / "exec.go"),
            },
            "required_artifacts": required,
            "blocking_reason": blocking,
            "go_executable": go_path or "",
            "miris_predicate": predicate,
            "plan_path": str(plan_path) if plan_path else "",
            "runner_command": [],
            "return_code": None,
            "stderr_tail": "",
        }

    def _base_metadata(self, query: DBQuery | None = None) -> dict[str, Any]:
        return {
            "source_repo": "official_miris",
            "official_code_path": str(self.source_dir),
            "source_dir": str(self.source_dir),
            "official_code_present": self.source_dir.exists(),
            "readme_path": str(self.source_dir / "README.md"),
            "readme_exists": (self.source_dir / "README.md").exists(),
            "requirements_path": str(self.source_dir / "models" / "gnn" / "requirements.txt"),
            "requirements_exists": (self.source_dir / "models" / "gnn" / "requirements.txt").exists(),
            "eval_entry": str(self.source_dir / "eval.go"),
            "plan_entry": str(self.source_dir / "plan.go"),
            "exec_entry": str(self.source_dir / "exec.go"),
            "converter_path": str(self.converter_path),
            "converter_exists": self.converter_path.exists(),
            "original_input_path": "",
            "original_output_path": "",
            "converted_output_path": str(self.converted_output_path),
            "supported_query_type": "",
            "unsupported_reason": "",
            "runner_command": [],
            "return_code": None,
            "stderr_tail": "",
            "query_type": query.type if query else "",
            "query_text": query.query if query else "",
        }

    def _timing(self, start: float) -> dict[str, float]:
        elapsed = round(time.perf_counter() - start, 6)
        return {
            "query_time_sec": elapsed,
            "rerank_time_sec": 0.0,
            "total_time_sec": elapsed,
        }


def _predicate_dataset(predicate: str) -> str:
    if predicate.startswith("shibuya"):
        return "shibuya"
    if predicate.startswith("warsaw"):
        return "warsaw"
    if predicate == "beach-runner":
        return "beach"
    if predicate == "uav":
        return "uav"
    return predicate


def _artifact_exists(path: Path) -> bool:
    if path.exists():
        return True
    parent = path.parent
    if not parent.exists():
        return False
    return any(parent.glob(path.name + "*"))


def _tail(value: str, max_chars: int = 2000) -> str:
    value = value or ""
    return value[-max_chars:]


def _query_fps(query: DBQuery) -> float:
    for key in ("fps", "video_fps", "frame_rate"):
        value = query.metadata.get(key) if isinstance(query.metadata, dict) else None
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return 30.0


def _track_bbox(detections: list[dict[str, Any]]) -> list[float] | None:
    if not detections:
        return None
    left = min(float(det.get("left", 0.0)) for det in detections)
    top = min(float(det.get("top", 0.0)) for det in detections)
    right = max(float(det.get("right", 0.0)) for det in detections)
    bottom = max(float(det.get("bottom", 0.0)) for det in detections)
    return [left, top, right, bottom]
