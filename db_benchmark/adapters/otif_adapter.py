from __future__ import annotations

from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter


class OTIFProxyAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "otif"
    declared_method = "OTIF"
    declared_source_kind = "official_proxy"
    declared_source_paths = (
        "third_party/external/official/OTIF/otif-master",
        "third_party/proxy/PROXY_POLICY.md",
    )
