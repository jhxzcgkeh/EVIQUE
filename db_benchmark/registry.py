from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from db_benchmark.adapters.base import AdapterContext, BaseDBAdapter
from db_benchmark.adapters.evique_db_adapter import EviqueDBAdapter
from db_benchmark.adapters.figo_adapter import FiGOAdapter
from db_benchmark.adapters.groundingdino_adapter import GroundingDINODirectAdapter
from db_benchmark.adapters.lovo_adapter import LOVOAdapter
from db_benchmark.adapters.miris_adapter import MIRISOfficialAdapter, MIRISProxyAdapter
from db_benchmark.adapters.otif_adapter import OTIFProxyAdapter
from db_benchmark.adapters.sieve_adapter import SIEVEAdapter
from db_benchmark.adapters.umt_adapter import UMTOfficialAdapter, UMTProxyAdapter
from db_benchmark.adapters.unsupported_adapter import UnsupportedDBAdapter
from db_benchmark.adapters.visa_adapter import VISAProxyAdapter
from db_benchmark.adapters.vocal_adapter import VOCALProxyAdapter
from db_benchmark.adapters.zelda_adapter import ZELDAAdapter
from db_benchmark.utils import read_json, slugify, write_json


DEFAULT_METHODS = [
    "EVIQUE-DB",
    "LOVO",
    "VOCAL",
    "MIRIS",
    "OTIF",
    "UMT",
    "VISA",
    "FiGO",
    "ZELDA",
]

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "EVIQUE-DB": {
        "canonical_name": "EVIQUE-DB",
        "result_stem": "evique_db",
        "adapter": EviqueDBAdapter,
        "implementation_fidelity": "native",
        "adapter_status": "integrated",
        "reason": "",
    },
    "LOVO": {
        "canonical_name": "LOVO",
        "result_stem": "lovo",
        "adapter": LOVOAdapter,
        "implementation_fidelity": "local_reproduction",
        "adapter_status": "proxy_runnable",
        "reason": "",
    },
    "SIEVE": {
        "canonical_name": "SIEVE",
        "result_stem": "sieve",
        "adapter": SIEVEAdapter,
        "implementation_fidelity": "integrated",
        "adapter_status": "integrated",
        "reason": "",
    },
    "FiGO": {
        "canonical_name": "FiGO",
        "result_stem": "figo",
        "adapter": FiGOAdapter,
        "implementation_fidelity": "local_reimplementation",
        "adapter_status": "local_reimplementation_proxy_runnable",
        "reason": "",
    },
    "ZELDA": {
        "canonical_name": "ZELDA",
        "result_stem": "zelda",
        "adapter": ZELDAAdapter,
        "implementation_fidelity": "local_reimplementation",
        "adapter_status": "local_reimplementation_proxy_runnable",
        "reason": "",
    },
    "GroundingDINO-direct": {
        "canonical_name": "GroundingDINO-direct",
        "result_stem": "groundingdino",
        "adapter": GroundingDINODirectAdapter,
        "implementation_fidelity": "official_model_adapted",
        "adapter_status": "official_present_missing_weights",
        "reason": "",
    },
    "VOCAL": {
        "canonical_name": "VOCAL",
        "registry_method": "VOCAL",
        "result_stem": "vocal_equivocal",
        "adapter": VOCALProxyAdapter,
        "implementation_fidelity": "third_party_proxy",
        "adapter_status": "proxy_runnable",
        "reason": "official EQUI-VOCAL status is retained in feasibility; main DB run uses the declared paper proxy over visual views",
    },
    "MIRIS": {
        "canonical_name": "MIRIS",
        "registry_method": "MIRIS",
        "result_stem": "miris",
        "adapter": MIRISProxyAdapter,
        "official_status_adapter": "MIRISOfficialAdapter",
        "implementation_fidelity": "third_party_proxy",
        "adapter_status": "proxy_runnable",
        "reason": "official MIRIS status adapter is retained; main DB run uses the declared MIRIS proxy over visual views",
    },
    "OTIF": {
        "canonical_name": "OTIF",
        "registry_method": "OTIF",
        "result_stem": "otif",
        "adapter": OTIFProxyAdapter,
        "implementation_fidelity": "third_party_proxy",
        "adapter_status": "proxy_runnable",
        "reason": "official OTIF status is retained in feasibility; main DB run uses the declared tracking-preprocess proxy over visual views",
    },
    "UMT": {
        "canonical_name": "UMT",
        "registry_method": "UMT",
        "result_stem": "umt",
        "adapter": UMTProxyAdapter,
        "official_status_adapter": "UMTOfficialAdapter",
        "implementation_fidelity": "third_party_proxy",
        "adapter_status": "proxy_runnable",
        "reason": "official UMT missing-checkpoint status is retained; main DB run uses the declared moment-retrieval proxy over visual views",
    },
    "VISA": {
        "canonical_name": "VISA",
        "registry_method": "VISA",
        "result_stem": "visa_videolisa",
        "adapter": VISAProxyAdapter,
        "implementation_fidelity": "third_party_proxy",
        "adapter_status": "proxy_runnable",
        "reason": "official VideoLISA missing-checkpoint status is retained; main DB run uses the declared VISA proxy over visual views",
    },
}

ALIASES = {
    "evique": "EVIQUE-DB",
    "eviquedb": "EVIQUE-DB",
    "evique_db": "EVIQUE-DB",
    "evique-db": "EVIQUE-DB",
    "lovo": "LOVO",
    "sieve": "SIEVE",
    "figo": "FiGO",
    "zelda": "ZELDA",
    "groundingdino": "GroundingDINO-direct",
    "groundingdino-direct": "GroundingDINO-direct",
    "groundingdino_direct": "GroundingDINO-direct",
    "vocal": "VOCAL",
    "equivocal": "VOCAL",
    "equi-vocal": "VOCAL",
    "vocal_equivocal": "VOCAL",
    "miris": "MIRIS",
    "otif": "OTIF",
    "umt": "UMT",
    "visa": "VISA",
    "videolisa": "VISA",
    "visa_videolisa": "VISA",
}


def parse_methods(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_METHODS)
    methods: list[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        methods.append(canonicalize_method(raw))
    deduped: list[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def canonicalize_method(value: str) -> str:
    normalized = slugify(value).replace("_", "-")
    compact = slugify(value)
    for key in {normalized, compact, value.strip().lower()}:
        if key in ALIASES:
            return ALIASES[key]
    if value in METHOD_SPECS:
        return value
    raise ValueError(f"unknown DB benchmark method: {value}")


def result_filename(method: str) -> str:
    return f"{METHOD_SPECS[method]['result_stem']}.jsonl"


def create_adapter(method: str, context: AdapterContext, spec: dict[str, Any] | None = None) -> BaseDBAdapter:
    merged = dict(METHOD_SPECS[method])
    if spec:
        merged.update(spec)
    adapter_cls = merged["adapter"]
    return adapter_cls(context, merged)


def load_third_party_registry(root: Path) -> dict[str, Any]:
    base = Path(root) / "third_party"
    registry_path = base / "method_registry.json"
    audit_path = base / "baseline_fidelity_audit.csv"
    registry = read_json(registry_path) if registry_path.exists() else {}
    audit_rows = []
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8-sig", newline="") as f:
            audit_rows = [_normalize_csv_row(row) for row in csv.DictReader(f)]
    return {
        "registry_path": str(registry_path),
        "audit_path": str(audit_path),
        "registry": registry,
        "audit_rows": audit_rows,
    }


def build_effective_registry(root: Path, methods: list[str], index_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    third_party = load_third_party_registry(root)
    audit_by_method = {row.get("method"): row for row in third_party.get("audit_rows", [])}
    third_party_by_method: dict[str, dict[str, Any]] = {}
    for section in ("main_methods", "third_party_methods", "support_methods"):
        for row in third_party.get("registry", {}).get(section, []) or []:
            method = row.get("method")
            if method:
                third_party_by_method[str(method)] = row

    effective_methods = []
    for method in methods:
        spec = dict(METHOD_SPECS[method])
        spec.pop("adapter", None)
        registry_method = spec.get("registry_method") or method.replace("-direct", "")
        third_party_row = third_party_by_method.get(registry_method, {})
        audit_row = audit_by_method.get(registry_method, {})
        if third_party_row.get("adapter_status"):
            spec["third_party_declared_adapter_status"] = third_party_row["adapter_status"]
        spec["third_party_registry"] = third_party_row
        spec["fidelity_audit"] = audit_row
        spec["index_metadata"] = (index_metadata or {}).get(method, {})
        effective_methods.append(spec)

    return {
        "schema_version": "db_benchmark_registry_v1",
        "methods": effective_methods,
        "third_party_sources": {
            "method_registry": third_party.get("registry_path", ""),
            "baseline_fidelity_audit": third_party.get("audit_path", ""),
        },
    }


def write_effective_registry(path: Path, root: Path, methods: list[str], index_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = build_effective_registry(root, methods, index_metadata=index_metadata)
    write_json(registry, path)
    return registry


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lstrip("\ufeff"): value for key, value in row.items()}
