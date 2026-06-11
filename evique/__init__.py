"""EVIQUE evidence indexing and retrieval."""

from .builder import build_evique_from_segments
from .evidence_packer import EvidencePacker, get_evidence_packer_config
from .retriever import EvidenceRetriever
from .video_identity import EVIQUE_VERSION, EVIQUE_VERSION_LABEL

MODEL_NAME = "EVIQUE"

__all__ = [
    "MODEL_NAME",
    "EVIQUE_VERSION",
    "EVIQUE_VERSION_LABEL",
    "EvidencePacker",
    "EvidenceRetriever",
    "build_evique_from_segments",
    "get_evidence_packer_config",
]
