from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Provenance:
    video_name: str
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None
    segment_id: str | None = None
    segment_index: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame_times: list[float] = field(default_factory=list)
    source: str = "segment_store"


@dataclass
class ScopeViewRecord:
    id: str
    node_id: str
    video_name: str
    segment_id: str
    segment_index: str
    start_time: float | None
    end_time: float | None
    scene_tags: list[str]
    caption: str
    transcript: str
    text: str
    provenance: Provenance
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class TargetViewRecord:
    id: str
    node_id: str
    video_name: str
    segment_id: str
    label: str
    aliases: list[str]
    attributes: list[str]
    confidence: float
    bbox: Any
    text: str
    provenance: Provenance
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class TrackViewRecord:
    id: str
    node_id: str
    video_name: str
    label: str
    object_ids: list[str]
    segment_ids: list[str]
    start_time: float | None
    end_time: float | None
    motion_summary: str
    action_tags: list[str]
    evidence_text: str = ""
    action_keywords: list[str] = field(default_factory=list)
    direction_keywords: list[str] = field(default_factory=list)
    state_keywords: list[str] = field(default_factory=list)
    related_event_ids: list[str] = field(default_factory=list)
    neighbor_track_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class EventViewRecord:
    id: str
    node_id: str
    video_name: str
    start_time: float | None
    end_time: float | None
    event_type: str
    summary: str
    state_signature: dict[str, Any]
    action_tags: list[str]
    related_segment_ids: list[str]
    related_object_ids: list[str]
    related_track_ids: list[str]
    previous_event_ids: list[str] = field(default_factory=list)
    next_event_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class AdaptiveEventViewRecord:
    id: str
    node_id: str
    event_id: str
    event_type: str
    event_segmentation_mode: str
    event_source: str
    video_name: str
    start_time: float
    end_time: float
    duration: float
    boundary_reason: str
    change_score: float
    dominant_signals: list[str]
    scene_tags: list[str]
    action_tags: list[str]
    object_counts: dict[str, int]
    related_segment_ids: list[str]
    related_keyframes: list[str]
    related_tracks: list[str]
    related_relations: list[str]
    summary: str
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class EvidenceNode:
    id: str
    type: str
    view_id: str
    text: str
    provenance: dict[str, Any]


@dataclass
class EvidenceRelation:
    source: str
    target: str
    relation: str
    provenance: dict[str, Any] = field(default_factory=dict)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
