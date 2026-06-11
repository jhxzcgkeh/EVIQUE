from __future__ import annotations

from db_benchmark.adapters.base import BaseDBAdapter
from db_benchmark.schema import DBQuery


class UnsupportedDBAdapter(BaseDBAdapter):
    implementation_fidelity = "official_present_not_integrated"
    adapter_status = "official_not_integrated"
    reason = "official adapter not integrated yet"

    def run_query(self, query: DBQuery) -> list[dict]:
        return [
            self.status_record(
                query,
                status="unsupported",
                reason=self.reason,
                metadata={
                    "source": "unsupported_adapter",
                    "registry_method": self.spec.get("registry_method", self.method),
                    "notes": "No proxy result is emitted for the main DB retrieval benchmark.",
                },
            )
        ]

