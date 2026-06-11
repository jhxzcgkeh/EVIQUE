from __future__ import annotations

from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter


class FiGOAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "local_reimplementation"
    adapter_status = "local_reimplementation_proxy_runnable"
    proxy_profile = "figo"
    declared_method = "FiGO"
    declared_source_kind = "local_reimplementation"
    declared_source_paths = (
        "third_party/external/reimpl/FiGO/run_local.py",
        "third_party/proxy/PROXY_POLICY.md",
    )
