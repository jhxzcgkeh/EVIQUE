from __future__ import annotations

import os
from pathlib import Path

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.schema import DBQuery


class GroundingDINODirectAdapter(BaseDBAdapter):
    implementation_fidelity = "official_model_adapted"
    adapter_status = "official_present_missing_weights"

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        self.source_dir = (
            self.context.root
            / "DB_Baselines"
            / "third_party_baselines"
            / "official"
            / "GroundingDINO"
            / "GroundingDINO-main"
        )
        self.config_path = self.source_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        self.checkpoint_path = self._find_checkpoint()

    def build_index(self) -> dict:
        metadata = super().build_index()
        metadata.update(self._metadata())
        metadata["status"] = "missing_checkpoint" if not self.checkpoint_path else "present"
        metadata["reason"] = "" if self.checkpoint_path else "GroundingDINO checkpoint was not found"
        return metadata

    def run_query(self, query: DBQuery) -> list[dict]:
        metadata = self._metadata()
        if not self.source_dir.exists():
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="GroundingDINO official source directory was not found",
                    adapter_status="not_available",
                    metadata=metadata,
                )
            ]
        if not self.checkpoint_path:
            return [
                self.status_record(
                    query,
                    status="missing_checkpoint",
                    reason="GroundingDINO checkpoint was not found; set GROUNDINGDINO_CHECKPOINT or place a .pth/.pt checkpoint under the official directory",
                    adapter_status="missing_checkpoint",
                    metadata=metadata,
                )
            ]
        return [
            self.status_record(
                query,
                status="unsupported",
                reason="GroundingDINO checkpoint is present, but frame-sampling inference is not wired in this lightweight benchmark runner yet",
                adapter_status="checkpoint_present_needs_detection_runner",
                implementation_fidelity="official_model_adapted",
                metadata=metadata,
            )
        ]

    def _find_checkpoint(self) -> Path | None:
        env_path = os.getenv("GROUNDINGDINO_CHECKPOINT")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        if not self.source_dir.exists():
            return None
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            for path in self.source_dir.rglob(pattern):
                if path.is_file():
                    return path
        return None

    def _metadata(self) -> dict:
        return {
            "source": "GroundingDINO-direct",
            "source_repo": "official_groundingdino",
            "source_dir": str(self.source_dir),
            "config_path": str(self.config_path),
            "source_dir_exists": self.source_dir.exists(),
            "config_exists": self.config_path.exists(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
            "frame_sampling": {
                "window_size": self.context.window_size,
                "stride": self.context.stride,
                "video": str(self.context.video_path) if self.context.video_path else "",
            },
            "notes": "Open-vocabulary detector baseline; no proxy rows are emitted without a checkpoint.",
        }
