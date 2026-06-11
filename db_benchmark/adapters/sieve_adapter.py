from __future__ import annotations

from pathlib import Path

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.schema import DBQuery
from db_benchmark.utils import first_existing


class SIEVEAdapter(BaseDBAdapter):
    implementation_fidelity = "integrated"
    adapter_status = "integrated"

    def __init__(self, context, spec=None):
        super().__init__(context, spec)
        root = self.context.root
        self.local_path = first_existing(
            [
                root / "src" / "session_method",
                root / "DB_Baselines" / "third_party_baselines" / "src" / "session_method",
                root / "DB_Baselines" / "third_party_baselines" / "session_method",
            ]
        )

    def build_index(self) -> dict:
        metadata = super().build_index()
        metadata["local_path"] = str(self.local_path) if self.local_path else ""
        metadata["status"] = "present" if self.local_path else "not_available"
        metadata["reason"] = "" if self.local_path else "src/session_method was not found in this checkout"
        return metadata

    def run_query(self, query: DBQuery) -> list[dict]:
        if not self.local_path:
            return [
                self.status_record(
                    query,
                    status="not_available",
                    reason="src/session_method was not found in this checkout",
                    implementation_fidelity="not_available",
                    adapter_status="not_available",
                    metadata={"searched_paths": self._searched_paths()},
                )
            ]
        return [
            self.status_record(
                query,
                status="not_available",
                reason="SIEVE local path exists, but no DB-window adapter entry point is integrated yet",
                adapter_status="local_path_present_needs_db_adapter",
                metadata={"local_path": str(Path(self.local_path))},
            )
        ]

    def _searched_paths(self) -> list[str]:
        root = self.context.root
        return [
            str(root / "src" / "session_method"),
            str(root / "DB_Baselines" / "third_party_baselines" / "src" / "session_method"),
            str(root / "DB_Baselines" / "third_party_baselines" / "session_method"),
        ]

