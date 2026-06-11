from __future__ import annotations

from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter


class LOVOAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "local_reproduction"
    adapter_status = "proxy_runnable"
    proxy_profile = "lovo"
    declared_method = "LOVO"
    declared_source_kind = "local_reproduction"
    declared_source_paths = (
        "src/lovo_baseline",
        "third_party/method_registry.json",
    )
