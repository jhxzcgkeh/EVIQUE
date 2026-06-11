from __future__ import annotations

from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter


class VOCALProxyAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "vocal"
    declared_method = "VOCAL"
    declared_source_kind = "official_proxy"
    declared_source_paths = (
        "third_party/external/official/VOCAL_EquiVOCAL/EQUI-VOCAL-main",
        "third_party/proxy/PROXY_POLICY.md",
    )
