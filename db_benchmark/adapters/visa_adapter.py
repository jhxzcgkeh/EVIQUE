from __future__ import annotations

from db_benchmark.adapters.third_party_proxy_adapter import ThirdPartyVisualProxyAdapter


class VISAProxyAdapter(ThirdPartyVisualProxyAdapter):
    implementation_fidelity = "third_party_proxy"
    adapter_status = "proxy_runnable"
    proxy_profile = "visa"
    declared_method = "VISA"
    declared_source_kind = "official_model_proxy"
    declared_source_paths = (
        "third_party/external/official/VISA_VideoLISA/VideoLISA-main",
        "third_party/proxy/PROXY_POLICY.md",
    )
