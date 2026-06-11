from __future__ import annotations

import os
from pathlib import Path

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter
from db_benchmark.schema import DBQuery


class UMTProxyAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "umt"
    declared_method = "UMT"
    declared_source_kind = "official_model_proxy"
    declared_source_paths = (
        "third_party/external/official/UMT/UMT-main",
        "third_party/proxy/PROXY_POLICY.md",
    )


class UMTOfficialAdapter(BaseDBAdapter):
    implementation_fidelity = "official_adapted"
    adapter_status = "official_present_missing_weights"

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        self.source_dir = (
            self.context.root
            / "DB_Baselines"
            / "third_party_baselines"
            / "official"
            / "UMT"
            / "UMT-main"
        )
        self.launch_path = self.source_dir / "tools" / "launch.py"
        self.default_config_path = self.source_dir / "configs" / "qvhighlights" / "umt_base_200e_qvhighlights.py"
        self.checkpoint_path = self._find_checkpoint()
        self.original_output_path = self.context.output_base / "external_outputs" / "umt" / "predictions.json"
        self.converted_output_path = self.context.result_path

    def build_index(self) -> dict:
        metadata = super().build_index()
        metadata.update(self._metadata())
        if not self.source_dir.exists():
            metadata["status"] = "not_available"
            metadata["reason"] = "UMT official source directory was not found"
        elif not self.checkpoint_path:
            metadata["status"] = "missing_checkpoint"
            metadata["reason"] = "UMT checkpoint was not found"
        else:
            metadata["status"] = "present"
            metadata["reason"] = ""
        return metadata

    def run_query(self, query: DBQuery) -> list[dict]:
        metadata = self._metadata(query)
        if not self.source_dir.exists():
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="UMT official source directory was not found",
                    adapter_status="not_available",
                    metadata=metadata,
                    implementation_fidelity="official_adapted",
                )
            ]
        if not self.checkpoint_path:
            return [
                self.status_record(
                    query,
                    status="missing_checkpoint",
                    reason="UMT checkpoint was not found; set UMT_CHECKPOINT or place a .pth/.pt/.ckpt file under the official UMT directory",
                    adapter_status="official_present_missing_weights",
                    metadata=metadata,
                    implementation_fidelity="official_adapted",
                )
            ]
        return [
            self.status_record(
                query,
                status="unsupported",
                reason=metadata["unsupported_reason"],
                adapter_status="official_present_needs_adapter",
                metadata=metadata,
                implementation_fidelity="official_adapted",
            )
        ]

    def _find_checkpoint(self) -> Path | None:
        env_path = os.getenv("UMT_CHECKPOINT")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        if not self.source_dir.exists():
            return None
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            for path in self.source_dir.rglob(pattern):
                if path.is_file():
                    return path
        return None

    def _metadata(self, query: DBQuery | None = None) -> dict:
        unsupported_reason = (
            "UMT is an end-to-end moment retrieval/highlight model. A faithful LAVA adapter must "
            "prepare UMT-compatible video features/query records, run official tools/launch.py with "
            "a pretrained checkpoint, and convert predicted moments to DBEvidence rows."
        )
        return {
            "source_repo": "official_umt",
            "category": "end_to_end_moment_retrieval",
            "not_object_level_db_system": True,
            "source_dir": str(self.source_dir),
            "official_code_present": self.source_dir.exists(),
            "readme_path": str(self.source_dir / "README.md"),
            "readme_exists": (self.source_dir / "README.md").exists(),
            "requirements_path": str(self.source_dir / "requirements.txt"),
            "requirements_exists": (self.source_dir / "requirements.txt").exists(),
            "launch_path": str(self.launch_path),
            "launch_exists": self.launch_path.exists(),
            "default_config_path": str(self.default_config_path),
            "default_config_exists": self.default_config_path.exists(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
            "original_output_path": str(self.original_output_path),
            "converted_output_path": str(self.converted_output_path),
            "supported_query_type": "moment_retrieval",
            "unsupported_reason": unsupported_reason,
            "query_type": query.type if query else "",
            "query_text": query.query if query else "",
        }
