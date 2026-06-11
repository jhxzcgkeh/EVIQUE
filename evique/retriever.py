from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cost_planner import CostBasedViewPlanner, get_cost_planner_config
from .evidence_packer import EvidencePacker, get_evidence_packer_config
from .planner import QueryPlanner
from .utils import (
    CANONICAL_VISUAL_RELATION_FILE,
    LEGACY_VISUAL_RELATION_FILE,
    format_seconds,
    overlap_score,
    read_json,
    read_jsonl,
    read_visual_relations,
    shorten,
    visual_relations_enabled,
)
from .view_stats import build_view_stats
from .video_identity import EVIQUE_VERSION, EVIQUE_VERSION_LABEL, metadata_video_values, video_identity_values

def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


DEFAULT_VIEW_TOPK = {
    "scope": 6,
    "target": 8,
    "track": 4,
    "event": 4,
}

QUERY_TYPE_VIEW_TOPK = {
    "object_list": {"target": 8, "scope": 4, "track": 0, "event": 0},
    "object_grounding": {"target": 8, "scope": 4, "track": 0, "event": 0},
    "trajectory": {"target": 6, "scope": 4, "track": 6, "event": 2},
    "interaction": {"target": 6, "scope": 4, "track": 6, "event": 2},
    "before_after": {"scope": 5, "target": 4, "track": 4, "event": 3},
    "temporal": {"scope": 5, "target": 4, "track": 4, "event": 3},
    "state_change": {"event": 5, "scope": 4, "track": 4, "target": 4},
    "event": {"event": 5, "scope": 4, "target": 3, "track": 3},
    "general_description": {"scope": 6, "target": 3, "track": 0, "event": 0},
    "scene": {"scope": 6, "target": 3, "track": 0, "event": 0},
    "speech": {"scope": 6, "target": 0, "track": 0, "event": 0},
}

TRACK_QUERY_TYPES = {"trajectory", "interaction", "before_after", "temporal", "state_change", "event"}
EVENT_QUERY_TYPES = {"before_after", "temporal", "state_change", "event"}
TEMPORAL_QUERY_TERMS = {"after", "before", "enter", "entered", "entering", "leave", "leaving", "left", "then", "when"}
VISUAL_EVIDENCE_VIEWS = {"visual_object", "visual_track", "visual_relation", "visual_event"}
SPATIAL_EVIDENCE_TERMS = {
    "adjacent",
    "around",
    "behind",
    "beside",
    "close",
    "front",
    "left",
    "near",
    "nearby",
    "nearest",
    "neighbor",
    "relative",
    "right",
    "side",
}
TEMPORAL_EVIDENCE_TERMS = {
    "after",
    "before",
    "change",
    "direction",
    "enter",
    "exit",
    "first",
    "later",
    "move",
    "moves",
    "moving",
    "next",
    "order",
    "sequence",
    "subsequent",
    "then",
    "trajectory",
    "transition",
}
INTERACTION_EVIDENCE_TERMS = {
    "cluster",
    "clustered",
    "follow",
    "following",
    "interaction",
    "multiple",
    "near",
    "parallel",
    "together",
    "wait",
    "waiting",
}
EVENT_EVIDENCE_TERMS = {
    "area",
    "center",
    "congestion",
    "event",
    "middle",
    "region",
    "slow",
    "stopped",
    "stopping",
    "window",
}
CAPTION_FALLBACK_TERMS = {
    "accumulation",
    "calm",
    "congested",
    "congestion",
    "flow",
    "flows",
    "light",
    "lights",
    "normal",
    "phase",
    "proceed",
    "queue",
    "queued",
    "queueing",
    "queuing",
    "red",
    "signal",
    "signals",
    "stop",
    "stopped",
    "stopping",
    "traffic",
    "wait",
    "waiting",
}
CAPTION_FALLBACK_CONTEXT_TERMS = {
    "car",
    "cars",
    "intersection",
    "road",
    "scene",
    "traffic",
    "vehicle",
    "vehicles",
}
OPEN_ENDED_CONTEXT_TERMS = {
    "after",
    "around",
    "before",
    "enter",
    "entering",
    "enters",
    "how",
    "located",
    "location",
    "movement",
    "moving",
    "near",
    "nearby",
    "relative",
    "through",
    "typical",
    "typically",
    "what",
    "when",
    "where",
}
SPATIAL_CONTEXT_TERMS = {
    "around",
    "center",
    "intersection",
    "middle",
    "near",
    "nearby",
    "relative",
    "spatial",
}
TEMPORAL_CONTEXT_TERMS = {
    "after",
    "before",
    "change",
    "during",
    "later",
    "phase",
    "then",
    "transition",
    "when",
}
SCENE_TRAFFIC_CONTEXT_TERMS = {
    "accumulation",
    "calm",
    "congestion",
    "flow",
    "intersection",
    "road",
    "scene",
    "traffic",
}
SIGNAL_CAPTION_TERMS = {
    "green",
    "light",
    "proceed",
    "proceeding",
    "red",
    "signal",
    "signals",
    "stop",
    "stopped",
    "wait",
    "waiting",
}
CAPTION_FALLBACK_PHRASES = {
    "calm period",
    "normal traffic flow",
    "red light",
    "red-light",
    "traffic flow",
    "traffic light",
    "traffic signal",
}
TEMPORAL_RELATION_STRICT_WINDOW_SECONDS = 5.0
TEMPORAL_RELATION_FALLBACK_WINDOW_SECONDS = 10.0
TEMPORAL_SEQUENCE_INTENT_KEYS = {"temporal_interaction", "temporal_ordering", "transition"}
TEMPORAL_SEQUENCE_MAX_ITEMS = 5
TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS = 120.0
TEMPORAL_SEQUENCE_MAX_SEGMENT_SPAN = 2
TEMPORAL_REFINEMENT_INTENTS = {
    "event_localization",
    "multi_object_interaction",
    "temporal_interaction",
    "temporal_ordering",
    "temporal_trajectory",
    "transition",
}
TEMPORAL_REFINEMENT_QUERY_TYPES = {
    "before_after",
    "event_localization",
    "interaction",
    "instance_spatial_temporal",
    "state_change",
    "temporal",
    "trajectory",
}
TEMPORAL_REFINEMENT_PHRASES = {
    "immediately after",
    "immediately before",
    "near crosswalk",
    "signal changes",
    "start moving",
    "starts moving",
    "traffic starts",
}
RELATION_SUPPLEMENT_TERMS = {
    "around",
    "beside",
    "close",
    "near",
    "nearby",
    "relative",
    "surrounding",
    "together",
}
EVENT_SUPPLEMENT_TERMS = {
    "after",
    "afterward",
    "afterwards",
    "before",
    "change",
    "moving",
    "next",
    "start",
    "starts",
    "then",
    "transition",
}
PEDESTRIAN_CROSSWALK_QUERY_TERMS = {
    "after",
    "before",
    "crosswalk",
    "pedestrian",
    "person",
    "people",
    "reaction",
    "react",
    "sidewalk",
    "vehicle",
    "vehicles",
    "wait",
    "waiting",
    "yield",
    "yielding",
}
PEDESTRIAN_LABEL_TERMS = {"pedestrian", "person", "people"}
CROSSWALK_LABEL_TERMS = {"crosswalk", "intersection", "road", "sidewalk"}
VEHICLE_LABEL_TERMS = {"bus", "car", "truck", "vehicle", "vehicles"}
VIEW_ALIASES = {
    "scope": {"scope", "segment", "caption_context", "coverage"},
    "object": {"object", "target", "visual_object"},
    "target": {"target", "object", "visual_object"},
    "visual_object": {"visual_object", "target", "object"},
    "track": {"track", "visual_track"},
    "visual_track": {"visual_track", "track"},
    "event": {"event", "adaptive_event", "visual_event", "fixed_window_event"},
    "adaptive_event": {"adaptive_event", "event", "visual_event", "fixed_window_event"},
    "visual_event": {"visual_event", "event", "adaptive_event", "fixed_window_event"},
    "fixed_window_event": {"fixed_window_event", "event", "adaptive_event", "visual_event"},
}
PLANNED_VIEW_NAMES = {
    "scope",
    "target",
    "track",
    "event",
    "adaptive_event",
    "fixed_window_event",
    "visual_object",
    "visual_track",
    "visual_event",
    "visual_relation",
}
LEGACY_EVENT_FAMILY_VIEWS = {"event", "adaptive_event", "visual_event"}
EVENT_FAMILY_VIEWS = LEGACY_EVENT_FAMILY_VIEWS | {"fixed_window_event"}
EVENT_REPLACEMENT_VIEWS = set(LEGACY_EVENT_FAMILY_VIEWS)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default or [])
    return [part.strip() for part in value.split(",") if part.strip()]


def _expand_view_aliases(values: list[str]) -> set[str]:
    expanded: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        expanded.update(VIEW_ALIASES.get(token, {token}))
    return expanded


def _view_disabled(view: str, disabled_views: set[str]) -> bool:
    if not disabled_views:
        return False
    aliases = VIEW_ALIASES.get(str(view or "").lower(), {str(view or "").lower()})
    return bool(aliases & disabled_views)


class EvidenceRetriever:
    def __init__(self, index_dir: Path, *, max_evidence: int = 18, token_budget: int = 12000):
        self.index_dir = Path(index_dir)
        self.max_evidence = max_evidence
        self.token_budget = token_budget
        self.planner = QueryPlanner(max_evidence=max_evidence, token_budget=token_budget)
        self.scopes = read_jsonl(self.index_dir / "scope_view.jsonl")
        self.targets = read_jsonl(self.index_dir / "target_view.jsonl")
        self.tracks = read_jsonl(self.index_dir / "track_view.jsonl")
        self.events = read_jsonl(self.index_dir / "event_view.jsonl")
        self.adaptive_events = read_jsonl(self.index_dir / "adaptive_event_view.jsonl")
        self.keyframes = read_jsonl(self.index_dir / "keyframe_view.jsonl")
        self.visual_objects = read_jsonl(self.index_dir / "visual_object_view.jsonl")
        self.visual_tracks = read_jsonl(self.index_dir / "visual_track_view.jsonl")
        self.visual_events = read_jsonl(self.index_dir / "visual_event_view.jsonl")
        self.scopes_by_segment = {row.get("segment_id"): row for row in self.scopes if row.get("segment_id")}
        self.nodes = {row["id"]: row for row in read_jsonl(self.index_dir / "evidence_nodes.jsonl")}
        self.relations = read_jsonl(self.index_dir / "evidence_relations.jsonl")
        self.index_manifest = read_json(self.index_dir / "index_manifest.json") if (self.index_dir / "index_manifest.json").exists() else {}
        self.video_identities = [
            row for row in self.index_manifest.get("video_identities", [])
            if isinstance(row, dict)
        ]
        self.graph_stats = read_json(self.index_dir / "graph_stats.json") if (self.index_dir / "graph_stats.json").exists() else {}
        manifest_relation_enabled = self.index_manifest.get("visual_relations_enabled")
        if manifest_relation_enabled is None:
            manifest_relation_enabled = self.graph_stats.get("visual_relations_enabled")
        self.visual_relations_enabled = visual_relations_enabled() and (
            bool(manifest_relation_enabled) if manifest_relation_enabled is not None else True
        )
        relation_file_exists = (
            (self.index_dir / CANONICAL_VISUAL_RELATION_FILE).exists()
            or (self.index_dir / LEGACY_VISUAL_RELATION_FILE).exists()
        )
        manifest_relation_generated = self.index_manifest.get("visual_relations_file_generated")
        if manifest_relation_generated is None:
            manifest_relation_generated = self.graph_stats.get("visual_relations_file_generated")
        self.visual_relations_file_generated = bool(
            self.visual_relations_enabled
            and (manifest_relation_generated if manifest_relation_generated is not None else relation_file_exists)
        )
        self.visual_relations = read_visual_relations(self.index_dir) if self.visual_relations_enabled else []
        self.view_stats = read_json(self.index_dir / "view_stats.json") if (self.index_dir / "view_stats.json").exists() else build_view_stats(self.index_dir)
        self.cost_planner = CostBasedViewPlanner(self.view_stats, config=get_cost_planner_config())
        self.relations_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in self.relations:
            self.relations_by_node[relation["source"]].append(relation)
            self.relations_by_node[relation["target"]].append(relation)
        self.visual_objects_by_id = {
            row.get("object_id") or row.get("id"): row
            for row in self.visual_objects
            if row.get("object_id") or row.get("id")
        }
        self.visual_tracks_by_id = {
            row.get("track_id") or row.get("id"): row
            for row in self.visual_tracks
            if row.get("track_id") or row.get("id")
        }
        self.visual_events_by_id = {
            row.get("event_id") or row.get("id"): row
            for row in self.visual_events
            if row.get("event_id") or row.get("id")
        }
        self.visual_relations_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.visual_relations_by_neighbor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.visual_relations:
            target_id = row.get("target_object_id")
            related_id = row.get("related_object_id") or row.get("neighbor_object_id")
            if target_id:
                self.visual_relations_by_target[str(target_id)].append(row)
            if related_id:
                self.visual_relations_by_neighbor[str(related_id)].append(row)
        self.visual_tracks_by_object_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for track in self.visual_tracks:
            for object_id in track.get("object_ids", []) or []:
                self.visual_tracks_by_object_id[str(object_id)].append(track)
        self.ablation_event_mode = os.getenv("EVIQUE_ABLATION_EVENT_MODE", "").strip().lower()
        self.ablation_fixed_window_size = max(0.1, _env_float("EVIQUE_ABLATION_FIXED_WINDOW_SIZE", 8.0))
        self.ablation_fixed_window_stride = max(0.1, _env_float("EVIQUE_ABLATION_FIXED_WINDOW_STRIDE", 4.0))
        self.fixed_window_events = (
            self._build_fixed_window_events(
                window_size=self.ablation_fixed_window_size,
                stride=self.ablation_fixed_window_stride,
            )
            if self.ablation_event_mode == "fixed_window"
            else []
        )

    def _ablation_settings(self) -> dict[str, Any]:
        disabled_views = _expand_view_aliases(_env_list("EVIQUE_ABLATION_DISABLED_VIEWS"))
        fixed_order = [
            view
            for view in _env_list(
                "EVIQUE_ABLATION_FIXED_VIEW_ORDER",
                ["scope", "visual_object", "visual_track", "adaptive_event", "visual_event"],
            )
            if view in PLANNED_VIEW_NAMES and not _view_disabled(view, disabled_views)
        ]
        event_mode = self.ablation_event_mode
        metadata = {
            "enabled": bool(
                _env_bool("EVIQUE_ABLATION_DISABLE_PLANNER", False)
                or _env_bool("EVIQUE_ABLATION_DISABLE_PACKAGING", False)
                or event_mode
                or disabled_views
            ),
            "disable_planner": _env_bool("EVIQUE_ABLATION_DISABLE_PLANNER", False),
            "fixed_view_order": fixed_order,
            "disable_packaging": _env_bool("EVIQUE_ABLATION_DISABLE_PACKAGING", False),
            "packing_mode": "raw_topk_truncate" if _env_bool("EVIQUE_ABLATION_DISABLE_PACKAGING", False) else "utility_aware",
            "event_mode": event_mode or "adaptive",
            "fixed_window_size": self.ablation_fixed_window_size if event_mode == "fixed_window" else None,
            "fixed_window_stride": self.ablation_fixed_window_stride if event_mode == "fixed_window" else None,
            "disabled_views": sorted(disabled_views),
        }
        return {
            "disable_planner": metadata["disable_planner"],
            "disable_packaging": metadata["disable_packaging"],
            "event_mode": event_mode,
            "fixed_view_order": fixed_order,
            "disabled_views": disabled_views,
            "metadata": metadata,
        }

    def _row_count_for_view(self, view: str) -> int:
        if view == "scope":
            return len(self.scopes)
        if view == "target":
            return len(self.targets)
        if view == "track":
            return len(self.tracks)
        if view == "event":
            return len(self.fixed_window_events) if self.ablation_event_mode == "fixed_window" else len(self.events)
        if view == "adaptive_event":
            return len(self.fixed_window_events) if self.ablation_event_mode == "fixed_window" else len(self.adaptive_events)
        if view == "visual_object":
            return len(self.visual_objects)
        if view == "visual_track":
            return len(self.visual_tracks)
        if view == "visual_event":
            return len(self.fixed_window_events) if self.ablation_event_mode == "fixed_window" else len(self.visual_events)
        if view == "visual_relation":
            return len(self.visual_relations) if self.visual_relations_enabled else 0
        if view == "fixed_window_event":
            return len(self.fixed_window_events)
        return 0

    def _is_fixed_window_event_mode(self) -> bool:
        return self.ablation_event_mode == "fixed_window"

    def _fixed_window_view_name(self, view: str) -> str:
        view = str(view or "")
        if self._is_fixed_window_event_mode() and view in LEGACY_EVENT_FAMILY_VIEWS:
            return "fixed_window_event"
        return view

    def _normalize_fixed_window_view_order(self, views: list[str]) -> list[str]:
        if not self._is_fixed_window_event_mode():
            return list(views)
        normalized: list[str] = []
        seen: set[str] = set()
        for view in views:
            mapped = self._fixed_window_view_name(str(view or ""))
            if not mapped or mapped in seen:
                continue
            normalized.append(mapped)
            seen.add(mapped)
        return normalized

    def _normalize_fixed_window_cost_plan(self, cost_plan: dict[str, Any]) -> dict[str, Any]:
        if not self._is_fixed_window_event_mode():
            return cost_plan
        refined = dict(cost_plan)
        view_order = self._normalize_fixed_window_view_order(list(refined.get("view_order") or []))

        row_budgets: dict[str, int] = {}
        for raw_view, raw_count in dict(refined.get("max_rows_per_view") or {}).items():
            view = self._fixed_window_view_name(str(raw_view or ""))
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                count = 0
            if view:
                row_budgets[view] = max(row_budgets.get(view, 0), count)
        for view in view_order:
            if view not in row_budgets:
                row_budgets[view] = 2 if view == "fixed_window_event" else 1

        estimated_costs: dict[str, dict[str, Any]] = {}
        for raw_view, raw_value in dict(refined.get("estimated_costs") or {}).items():
            view = self._fixed_window_view_name(str(raw_view or ""))
            if not view:
                continue
            value = dict(raw_value or {})
            value["row_count"] = self._row_count_for_view(view)
            value["available"] = self._row_count_for_view(view) > 0
            estimated_costs[view] = value
        for view in view_order:
            estimated_costs.setdefault(
                view,
                {
                    "row_count": self._row_count_for_view(view),
                    "estimated_cost": 0.0,
                    "expected_benefit": 0.0,
                    "planner_score": 0.0,
                    "available": self._row_count_for_view(view) > 0,
                },
            )

        anchor_view = self._fixed_window_view_name(str(refined.get("anchor_view") or ""))
        refined["event_mode"] = "fixed_window"
        refined["view_order"] = view_order
        refined["max_rows_per_view"] = row_budgets
        refined["estimated_costs"] = estimated_costs
        refined["anchor_view"] = anchor_view if anchor_view in view_order else (view_order[0] if view_order else "scope")
        refined["reason"] = f"{refined.get('reason') or ''}; event_mode=fixed_window".strip("; ")
        return refined

    def _fixed_view_cost_plan(self, plan: Any, view_order: list[str]) -> dict[str, Any]:
        view_order = self._normalize_fixed_window_view_order(view_order)
        view_order = [view for view in view_order if self._row_count_for_view(view) > 0]
        if not view_order:
            view_order = ["scope"]
        max_rows_total = max(1, _env_int("EVIQUE_COST_PLANNER_MAX_ROWS_TOTAL", max(16, self.max_evidence)))
        defaults = {
            "scope": 4,
            "target": 6,
            "track": 4,
            "event": 4,
            "adaptive_event": 5,
            "fixed_window_event": 5,
            "visual_object": 6,
            "visual_track": 4,
            "visual_event": 4,
            "visual_relation": 6,
        }
        remaining = max_rows_total
        budgets: dict[str, int] = {}
        for view in view_order:
            budget = min(defaults.get(view, 3), remaining)
            budgets[view] = max(0, budget)
            remaining -= budget
            if remaining <= 0:
                break
        view_order = [view for view in view_order if budgets.get(view, 0) > 0]
        estimates = {
            view: {
                "row_count": self._row_count_for_view(view),
                "estimated_cost": 0.0,
                "expected_benefit": 0.0,
                "planner_score": 0.0,
                "available": self._row_count_for_view(view) > 0,
            }
            for view in view_order
        }
        return {
            "planner_version": "ablation_fixed_view_order",
            "enabled": False,
            "ablation_disable_planner": True,
            "query_type": str(getattr(plan, "query_type", "default")),
            "anchor_view": view_order[0] if view_order else "scope",
            "view_order": view_order,
            "max_rows_per_view": budgets,
            "max_rows_total": max_rows_total,
            "min_confidence": 2.0,
            "stop_condition": "fixed_view_order_exhausted",
            "estimated_costs": estimates,
            "reason": f"ablation fixed view order: {' -> '.join(view_order)}",
            "debug_trace": [],
        }

    def _filter_disabled_views_from_cost_plan(self, cost_plan: dict[str, Any], disabled_views: set[str]) -> dict[str, Any]:
        if not disabled_views:
            return cost_plan
        refined = dict(cost_plan)
        view_order = [view for view in list(refined.get("view_order") or []) if not _view_disabled(view, disabled_views)]
        row_budgets = {
            view: count
            for view, count in dict(refined.get("max_rows_per_view") or {}).items()
            if not _view_disabled(view, disabled_views)
        }
        estimated_costs = {
            view: value
            for view, value in dict(refined.get("estimated_costs") or {}).items()
            if not _view_disabled(view, disabled_views)
        }
        refined["view_order"] = view_order
        refined["max_rows_per_view"] = row_budgets
        refined["estimated_costs"] = estimated_costs
        refined["disabled_views"] = sorted(disabled_views)
        if refined.get("anchor_view") not in view_order:
            refined["anchor_view"] = view_order[0] if view_order else "scope"
        return refined

    def _filter_disabled_view_hits(
        self,
        view_hits: dict[str, list[dict[str, Any]]],
        disabled_views: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not disabled_views:
            return view_hits
        return {
            view: self._filter_disabled_evidence(rows, disabled_views)
            for view, rows in view_hits.items()
            if not _view_disabled(view, disabled_views)
        }

    def _evidence_view_names(self, item: dict[str, Any]) -> set[str]:
        names = {str(item.get("view") or ""), str(item.get("type") or ""), str(item.get("source_view") or "")}
        names.update(str(value) for value in item.get("source_views") or [])
        record = item.get("record") or {}
        if isinstance(record, dict):
            names.update({str(record.get("view") or ""), str(record.get("type") or ""), str(record.get("source_view") or "")})
            names.update(str(value) for value in record.get("source_views") or [])
        return {name.lower() for name in names if name}

    def _filter_disabled_evidence(self, evidence: list[dict[str, Any]], disabled_views: set[str]) -> list[dict[str, Any]]:
        if not disabled_views:
            return evidence
        out: list[dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                out.append(item)
                continue
            if any(_view_disabled(view, disabled_views) for view in self._evidence_view_names(item)):
                continue
            out.append(item)
        return out

    def _expand_fixed_event_selection(self, views: list[str]) -> list[str]:
        if self.ablation_event_mode != "fixed_window":
            return views
        expanded = list(views)
        if any(view in EVENT_REPLACEMENT_VIEWS for view in views) and "fixed_window_event" not in expanded:
            expanded.append("fixed_window_event")
        return self._normalize_fixed_window_view_order(expanded)

    def _raw_topk_truncate_package(
        self,
        candidates: list[dict[str, Any]],
        *,
        max_items: int,
        token_budget: int,
        char_budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        used_tokens = 0
        used_chars = 0
        for index, raw in enumerate(candidates):
            if len(selected) >= max_items:
                dropped.append({**raw, "drop_reason": "max_items_reached", "original_rank": index})
                continue
            item = dict(raw)
            text = str(item.get("text") or item.get("short_text") or "")
            item_chars = len(text)
            item_tokens = max(1, int(item_chars / 4))
            remaining_tokens = max(0, token_budget - used_tokens)
            remaining_chars = max(0, char_budget - used_chars)
            if remaining_tokens <= 0 or remaining_chars <= 0:
                dropped.append({**item, "drop_reason": "token_budget_exhausted", "original_rank": index})
                continue
            if item_tokens > remaining_tokens or item_chars > remaining_chars:
                allowed_chars = min(remaining_chars, max(24, remaining_tokens * 4))
                if allowed_chars < 24:
                    dropped.append({**item, "drop_reason": "token_budget_exhausted", "original_rank": index})
                    continue
                item["text"] = shorten(text, allowed_chars)
                item["short_text"] = shorten(item["text"], min(260, allowed_chars))
                item_chars = len(str(item.get("text") or item.get("short_text") or ""))
                item_tokens = max(1, int(item_chars / 4))
            selected.append(item)
            used_chars += item_chars
            used_tokens += item_tokens
        metadata = {
            "evidence_packer_enabled": False,
            "packing_strategy": "raw_topk_truncate",
            "ablation_disable_packaging": True,
            "candidate_evidence_count": len(candidates),
            "packed_evidence_count": len(selected),
            "dropped_evidence_count": len(dropped),
            "estimated_packed_tokens": used_tokens,
            "estimated_candidate_tokens": sum(max(1, int(len(str(item.get("text") or item.get("short_text") or "")) / 4)) for item in candidates),
            "estimated_packed_chars": used_chars,
            "estimated_candidate_chars": sum(len(str(item.get("text") or item.get("short_text") or "")) for item in candidates),
            "evidence_token_budget": token_budget,
            "evidence_char_budget": char_budget,
            "evidence_max_items": max_items,
            "packing_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in selected)),
            "dropped_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in dropped)),
            "budget_fill_ratio": round(used_tokens / max(float(token_budget), 1.0), 6),
            "budget_exhausted": bool(dropped and used_tokens >= token_budget),
            "temporal_aware_packing_used": False,
            "temporal_anchor_segment": "",
            "temporal_before_count": 0,
            "temporal_focal_count": 0,
            "temporal_after_count": 0,
            "temporal_supplement_count": 0,
            "relation_supplement_used": False,
            "relation_supplement_count": 0,
            "event_supplement_used": False,
            "event_supplement_count": 0,
            "pedestrian_crosswalk_expansion_used": False,
            "caption_fallback_used": False,
        }
        return selected, dropped, metadata

    def _build_fixed_window_events(self, *, window_size: float, stride: float) -> list[dict[str, Any]]:
        source_rows: list[tuple[str, dict[str, Any], float, float, str]] = []
        for view, rows in (("visual_object", self.visual_objects), ("visual_track", self.visual_tracks)):
            for row in rows:
                span = self._fixed_window_row_span(row)
                if span is None:
                    continue
                start, end = span
                video_key = self._video_key_for_fixed_window(row)
                source_rows.append((view, row, start, end, video_key))
        if not source_rows:
            return []
        by_video: dict[str, list[tuple[str, dict[str, Any], float, float, str]]] = defaultdict(list)
        for item in source_rows:
            by_video[item[4]].append(item)
        events: list[dict[str, Any]] = []
        for video_key, rows in by_video.items():
            min_start = min(start for _, _, start, _, _ in rows)
            max_end = max(end for _, _, _, end, _ in rows)
            window_start = min_start
            while window_start <= max_end:
                window_end = window_start + window_size
                in_window = [
                    (view, row, start, end)
                    for view, row, start, end, _ in rows
                    if start < window_end and end > window_start
                ]
                if in_window:
                    object_rows = [row for view, row, _, _ in in_window if view == "visual_object"]
                    track_rows = [row for view, row, _, _ in in_window if view == "visual_track"]
                    labels = Counter(
                        str(row.get("label") or row.get("category") or "object")
                        for row in object_rows
                        if row.get("label") or row.get("category")
                    )
                    track_labels = Counter(
                        str(row.get("label") or row.get("category") or "track")
                        for row in track_rows
                        if row.get("label") or row.get("category")
                    )
                    object_text = ", ".join(f"{label}={count}" for label, count in labels.most_common(8)) or "none"
                    track_text = ", ".join(f"{label}={count}" for label, count in track_labels.most_common(8)) or "none"
                    track_ids = [
                        str(row.get("track_id") or row.get("id"))
                        for row in track_rows[:8]
                        if row.get("track_id") or row.get("id")
                    ]
                    text = (
                        f"Fixed-window event from {window_start:.1f}s to {window_end:.1f}s: "
                        f"objects include {object_text}; tracks include {track_text}; "
                        f"track_ids={', '.join(track_ids) or 'none'}."
                    )
                    events.append(
                        {
                            "id": f"fixed_event:{video_key}:{window_start:.2f}:{window_end:.2f}",
                            "node_id": f"fixed_event:{video_key}:{window_start:.2f}:{window_end:.2f}",
                            "view": "fixed_window_event",
                            "video_id": video_key,
                            "source_vid": video_key,
                            "start_time": round(window_start, 3),
                            "end_time": round(window_end, 3),
                            "text": text,
                            "summary": text,
                            "source_views": ["visual_object", "visual_track"],
                            "metadata": {
                                "window_size": window_size,
                                "stride": stride,
                                "object_count": len(object_rows),
                                "track_count": len(track_rows),
                            },
                            "provenance": {
                                "video_id": video_key,
                                "source_vid": video_key,
                                "start_time": round(window_start, 3),
                                "end_time": round(window_end, 3),
                                "source": "ablation_fixed_window_event",
                            },
                        }
                    )
                window_start += stride
        return events

    def _fixed_window_row_span(self, row: dict[str, Any]) -> tuple[float, float] | None:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        start_value = row.get("start_time", row.get("timestamp", provenance.get("start_time", provenance.get("timestamp"))))
        end_value = row.get("end_time", row.get("timestamp", provenance.get("end_time", provenance.get("timestamp"))))
        try:
            start = float(start_value)
            end = float(end_value)
        except (TypeError, ValueError):
            return None
        if end <= start:
            end = start + 0.5
        return start, end

    def _video_key_for_fixed_window(self, row: dict[str, Any]) -> str:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        for key in ("video_id", "source_vid", "video_name"):
            value = row.get(key) or provenance.get(key)
            if value:
                return str(value)
        return "unknown_video"

    def _retrieve_fixed_window_events(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        temporal_bonus = 0.35 if query_tokens & (TEMPORAL_QUERY_TERMS | EVENT_SUPPLEMENT_TERMS) else 0.0
        for record in self.fixed_window_events:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            text = str(record.get("text") or record.get("summary") or "")
            score = overlap_score(query_tokens, text) + temporal_bonus
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hit = self._hit("fixed_window_event", record, score, text)
                hit["source_views"] = list(record.get("source_views") or [])
                hit["metadata"] = dict(record.get("metadata") or {})
                hits.append(hit)
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _fixed_window_event_fallback_reason(self, video_filter: set[str] | None) -> str:
        visual_source_rows = [
            row
            for rows in (self.visual_objects, self.visual_tracks)
            for row in rows
            if self._row_matches_video_filter(row, video_filter)
        ]
        if not visual_source_rows:
            return "no_visual_object_or_visual_track_rows_available"
        rows_with_spans = [row for row in visual_source_rows if self._fixed_window_row_span(row) is not None]
        if not rows_with_spans:
            return "visual_object_or_visual_track_rows_have_no_valid_time_spans"
        if not self.fixed_window_events:
            return "no_fixed_window_events_constructed_from_visual_object_or_visual_track"
        if not self._filter_rows_by_video(self.fixed_window_events, video_filter):
            return "no_fixed_window_events_match_query_video_filter"
        return "fixed_window_event_unavailable"

    def _fixed_window_candidate_pool(
        self,
        pools: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for pool in pools:
            for item in pool:
                if not isinstance(item, dict) or self._evidence_type(item) != "fixed_window_event":
                    continue
                key = (
                    "fixed_window_event",
                    str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)
        return candidates

    def _enforce_fixed_window_event_evidence(
        self,
        query: str,
        plan: Any,
        query_tokens: set[str],
        evidence: list[dict[str, Any]],
        candidate_pools: list[list[dict[str, Any]]],
        *,
        max_items: int,
        max_chars: int,
        disabled_views: set[str],
        video_filter: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self._is_fixed_window_event_mode():
            return evidence, {}

        original_count = len(evidence)
        cleaned = [item for item in evidence if self._evidence_type(item) not in LEGACY_EVENT_FAMILY_VIEWS]
        removed_count = original_count - len(cleaned)
        fixed_disabled = _view_disabled("fixed_window_event", disabled_views)
        fixed_candidates = self._fixed_window_candidate_pool([cleaned, *candidate_pools])
        fallback_candidates: list[dict[str, Any]] = []
        if not fixed_candidates and not fixed_disabled:
            fallback_candidates = self._retrieve_fixed_window_events(
                query,
                query_tokens,
                limit=max(1, max_items),
                fallback=True,
                video_filter=video_filter,
            )
            fixed_candidates = self._fixed_window_candidate_pool([fallback_candidates])

        injected = False
        fallback_failed = False
        fallback_reason = ""
        if not fixed_candidates:
            fallback_failed = True
            fallback_reason = (
                "fixed_window_event_disabled_by_ablation"
                if fixed_disabled
                else self._fixed_window_event_fallback_reason(video_filter)
            )
        else:
            fixed_candidates.sort(
                key=lambda item: (
                    self._intent_evidence_priority(plan, item),
                    _safe_float(item.get("score"), 0.0),
                ),
                reverse=True,
            )
            chosen = fixed_candidates[0]
            chosen_key = str(chosen.get("id") or chosen.get("node_id") or self._compact_evidence_preview(chosen))
            has_chosen = any(
                self._evidence_type(item) == "fixed_window_event"
                and str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)) == chosen_key
                for item in cleaned
            )
            injected = not has_chosen
            cleaned = [
                chosen,
                *[
                    item
                    for item in cleaned
                    if not (
                        self._evidence_type(item) == "fixed_window_event"
                        and str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)) == chosen_key
                    )
                ],
            ]

        cleaned = self._dedupe_evidence_items(cleaned)
        trimmed = self._trim_evidence(cleaned, max_items=max_items, max_chars=max_chars)
        if fixed_candidates and not any(self._evidence_type(item) == "fixed_window_event" for item in trimmed):
            trimmed = self._trim_evidence([fixed_candidates[0], *trimmed], max_items=max_items, max_chars=max_chars)

        final_event_views = sorted(
            {
                self._evidence_type(item)
                for item in trimmed
                if self._evidence_type(item) in EVENT_FAMILY_VIEWS
            }
        )
        fixed_in_final = "fixed_window_event" in final_event_views
        if fixed_candidates and not fixed_in_final:
            fallback_failed = True
            fallback_reason = "fixed_window_event_trimmed_from_final_evidence"

        metadata = {
            "event_mode": "fixed_window",
            "fixed_window_event_required": True,
            "fixed_window_event_in_final": fixed_in_final,
            "fixed_window_event_injected": injected,
            "fixed_window_event_fallback_failed": fallback_failed,
            "fixed_window_event_fallback_reason": fallback_reason,
            "fixed_window_event_candidates": len(fixed_candidates),
            "fixed_window_event_fallback_candidates": len(fallback_candidates),
            "fixed_window_event_constructed_count": len(self.fixed_window_events),
            "legacy_event_family_removed_count": removed_count,
            "final_event_family_views": final_event_views,
            "packed_evidence_count": len(trimmed),
            "packing_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in trimmed)),
            "estimated_packed_chars": sum(len(str(item.get("text") or item.get("short_text") or "")) for item in trimmed),
            "estimated_packed_tokens": sum(
                max(1, int(len(str(item.get("text") or item.get("short_text") or "")) / 4))
                for item in trimmed
            ),
        }
        return trimmed, metadata

    def retrieve(self, query: str, query_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        plan = self.planner.plan(query)
        query_tokens = set(plan.query_terms)
        video_filter = self._video_filter_from_metadata(query_metadata)
        if video_filter and not self._video_filter_matches_index(video_filter):
            requested = ", ".join(sorted(video_filter))
            available = ", ".join(sorted(self._available_video_values())[:20])
            raise ValueError(
                "strict video filter did not match any video_id/video_name/video_path in this EVIQUE workdir: "
                f"requested=[{requested}] available_sample=[{available}]"
            )
        ablation = self._ablation_settings()
        cost_plan = (
            self._fixed_view_cost_plan(plan, ablation["fixed_view_order"])
            if ablation["disable_planner"]
            else self.cost_planner.plan(query, plan, video_filter=video_filter)
            if self.cost_planner.enabled
            else None
        )
        temporal_refinement_query = self._temporal_refinement_query(query, plan, query_tokens)
        pedestrian_crosswalk_query = (
            _env_bool("EVIQUE_PEDESTRIAN_CROSSWALK_EXPAND", True)
            and self._pedestrian_crosswalk_query(query, plan, query_tokens)
        )
        if cost_plan and not ablation["disable_planner"] and (temporal_refinement_query or pedestrian_crosswalk_query):
            cost_plan = self._refine_cost_plan_for_temporal(
                cost_plan,
                require_pedestrian_crosswalk=pedestrian_crosswalk_query,
            )
        if cost_plan:
            cost_plan = self._filter_visual_relation_cost_plan(cost_plan)
        if cost_plan:
            cost_plan = self._filter_disabled_views_from_cost_plan(cost_plan, ablation["disabled_views"])
        if cost_plan:
            cost_plan = self._normalize_fixed_window_cost_plan(cost_plan)
        if cost_plan:
            view_hits, execution_metrics = self._execute_cost_plan(
                query,
                plan,
                cost_plan,
                query_tokens,
                video_filter=video_filter,
            )
            selected_views = list(cost_plan.get("view_order") or [])
        else:
            selected_views = [
                view for view in list(plan.selected_views)
                if not _view_disabled(view, ablation["disabled_views"])
            ]
            selected_views = self._normalize_fixed_window_view_order(selected_views)
            event_view = "fixed_window_event" if ablation["event_mode"] == "fixed_window" else "event"
            event_hits = (
                self._retrieve_fixed_window_events(query, query_tokens, limit=self.max_evidence, video_filter=video_filter)
                if event_view == "fixed_window_event"
                else self._retrieve_events(query, query_tokens, limit=self.max_evidence, video_filter=video_filter)
            )
            view_hits = {
                "scope": self._retrieve_scopes(query, query_tokens, limit=self.max_evidence, video_filter=video_filter),
                "target": self._retrieve_targets(query, query_tokens, limit=self.max_evidence, video_filter=video_filter),
                "track": self._retrieve_tracks(query, query_tokens, limit=self.max_evidence, video_filter=video_filter),
                event_view: event_hits,
            }
            view_hits = self._filter_disabled_view_hits(view_hits, ablation["disabled_views"])
            execution_metrics = {
                "event_mode": "fixed_window" if ablation["event_mode"] == "fixed_window" else "adaptive",
                "views_queried": ["scope", "target", "track", event_view],
                "views_skipped": sorted(ablation["disabled_views"]),
                "stop_reason": "v4_all_core_views_queried",
                "evidence_confidence": 0.0,
                "evidence_coverage": 0.0,
                "planner_debug_trace": [],
            }
        view_hits = self._filter_disabled_view_hits(view_hits, ablation["disabled_views"])
        selected_views = [
            view for view in selected_views
            if not _view_disabled(view, ablation["disabled_views"])
        ]
        fallback_retrievers = {
            "scope": self._retrieve_scopes,
            "target": self._retrieve_targets,
            "track": self._retrieve_tracks,
        }
        if ablation["event_mode"] == "fixed_window":
            fallback_retrievers["fixed_window_event"] = self._retrieve_fixed_window_events
        else:
            fallback_retrievers["event"] = self._retrieve_events
        visual_chain = None
        if not cost_plan and plan.query_type == "instance_spatial_temporal":
            visual_chain = self._retrieve_instance_spatial_temporal(query, plan, query_tokens, video_filter=video_filter)

        for view in selected_views:
            if view not in fallback_retrievers:
                continue
            if not view_hits.get(view):
                view_hits[view] = fallback_retrievers[view](
                    query,
                    query_tokens,
                    limit=1,
                    fallback=True,
                    video_filter=video_filter,
                )

        hits: list[dict[str, Any]] = []
        for view in selected_views:
            hits.extend(view_hits.get(view, []))

        if plan.query_type == "object_list":
            coverage = self._target_label_coverage(video_filter=video_filter)
            if coverage:
                hits.append(coverage)

        if not hits:
            hits.extend(
                self._retrieve_scopes(
                    query,
                    query_tokens,
                    limit=min(4, self.max_evidence),
                    fallback=True,
                    video_filter=video_filter,
                )
            )

        retrieved_count = len(hits)
        ranked = self._prioritize(hits, plan, query_tokens)
        if cost_plan:
            anchor_view = str(cost_plan.get("anchor_view") or "")
            required_views = [anchor_view] if anchor_view else []
            optional_views = [view for view in selected_views if view != anchor_view]
        else:
            required_views = [view for view in plan.required_views if view in {"scope", "target", "track", "event"}]
            optional_views = [view for view in plan.optional_views if view in {"scope", "target", "track", "event"}]
        if ablation["event_mode"] == "fixed_window":
            required_views = self._expand_fixed_event_selection(required_views)
            optional_views = self._expand_fixed_event_selection(optional_views)
        optional_threshold = float(plan.constraints.get("optional_score_threshold", 0.15))
        required_segment_ids = {
            segment_id
            for hit in ranked
            if hit.get("view") in required_views
            for segment_id in self._candidate_segment_ids(hit)
        }
        filtered: list[dict[str, Any]] = []
        for hit in ranked:
            view = str(hit.get("view"))
            if view in required_views:
                filtered.append(hit)
            elif view in optional_views and float(hit.get("score", 0.0)) >= optional_threshold:
                if cost_plan or not required_segment_ids or self._is_adjacent_to_segments(hit, required_segment_ids):
                    filtered.append(hit)
        filtered_count = len(filtered)
        minimal = self.build_minimal_evidence_package(query, plan, filtered, plan.constraints)
        evidence = self._filter_disabled_evidence(minimal["evidence"], ablation["disabled_views"])
        visual_chain_found = bool(visual_chain and visual_chain.get("found"))
        visual_chain_evidence = (
            self._filter_disabled_evidence(list(visual_chain.get("evidence") or []), ablation["disabled_views"])
            if visual_chain_found
            else []
        )
        cost_visual_evidence_count = sum(
            len(view_hits.get(view, []))
            for view in ("visual_object", "visual_track", "visual_relation", "visual_event", "fixed_window_event")
        )
        visual_chain_attempted = self._visual_chain_attempted(plan) or bool(cost_visual_evidence_count)
        visual_intent_evidence = (
            self._retrieve_visual_intent_evidence(query, plan, query_tokens, video_filter=video_filter)
            if visual_chain_attempted and not cost_plan
            else []
        )
        visual_intent_evidence = self._filter_disabled_evidence(visual_intent_evidence, ablation["disabled_views"])
        visual_failure_reason = visual_chain.get("failure_reason") if visual_chain is not None else None
        combined_visual_evidence = visual_chain_evidence + visual_intent_evidence
        merged_evidence = list(evidence)
        if combined_visual_evidence:
            combined_visual_evidence = self._dedupe_visual_evidence(combined_visual_evidence, max_per_pair=2)
            combined_visual_evidence.sort(
                key=lambda item: self._intent_evidence_priority(plan, item),
                reverse=True,
            )
            merged_evidence: list[dict[str, Any]] = []
            seen_evidence: set[tuple[str, str]] = set()
            for item in combined_visual_evidence + evidence:
                key = (
                    str(item.get("view") or ""),
                    str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)),
                )
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                merged_evidence.append(item)
        fused_evidence, fusion_metrics = self._fuse_and_naturalize_evidence(
            query,
            plan,
            merged_evidence,
            video_filter=video_filter,
        )
        fused_evidence = self._filter_disabled_evidence(fused_evidence, ablation["disabled_views"])
        candidate_evidence = list(fused_evidence)
        final_max_items = min(self.max_evidence, int(plan.constraints.get("max_evidence", self.max_evidence)))
        final_max_chars = int(plan.constraints.get("max_evidence_chars", min(self.token_budget, 8000)))
        v5_evidence = self._trim_evidence(
            fused_evidence,
            max_items=final_max_items,
            max_chars=final_max_chars,
        )
        v5_evidence = self._ensure_nearby_context_in_final(
            v5_evidence,
            fused_evidence,
            max_items=final_max_items,
            max_chars=final_max_chars,
        )
        v5_evidence = self._filter_disabled_evidence(v5_evidence, ablation["disabled_views"])
        temporal_sequence_evidence, temporal_context_reason = self._augment_temporal_context(
            query,
            plan,
            v5_evidence,
            fused_evidence,
        )
        temporal_sequence_evidence = self._filter_disabled_evidence(temporal_sequence_evidence, ablation["disabled_views"])
        evidence = v5_evidence
        dropped_evidence: list[dict[str, Any]] = []
        packer_config = get_evidence_packer_config()
        packer_config["max_items"] = max(1, int(self.max_evidence))
        evidence_packer = EvidencePacker(packer_config)
        relation_supplement_used = False
        relation_supplement_count = 0
        event_supplement_used = False
        event_supplement_count = 0
        pedestrian_crosswalk_expansion_used = False
        if ablation["disable_packaging"]:
            raw_source = candidate_evidence or v5_evidence
            evidence, dropped_evidence, packing_metadata = self._raw_topk_truncate_package(
                raw_source,
                max_items=final_max_items,
                token_budget=max(1, int(packer_config.get("token_budget") or self.token_budget or 3200)),
                char_budget=max(1, int(packer_config.get("char_budget") or max(final_max_chars, 1))),
            )
        elif evidence_packer.enabled:
            if temporal_sequence_evidence:
                candidate_evidence.extend(temporal_sequence_evidence)

            def append_unique(additions: list[dict[str, Any]], *, limit: int | None = None) -> int:
                existing = {
                    (
                        self._evidence_type(item),
                        str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)),
                    )
                    for item in candidate_evidence
                }
                added = 0
                for item in additions:
                    key = (
                        self._evidence_type(item),
                        str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)),
                    )
                    if key in existing:
                        continue
                    candidate_evidence.append(item)
                    existing.add(key)
                    added += 1
                    if limit is not None and added >= limit:
                        break
                return added

            relation_query = (
                self._relation_supplement_query(query, plan, query_tokens)
                or "visual_relation" in ((cost_plan or {}).get("view_order") or [])
            )
            current_relation_count = sum(
                1 for item in candidate_evidence if self._evidence_type(item) == "visual_relation"
            )
            relation_limit = int(packer_config.get("relation_supplement_min") or 0)
            if relation_query and self.visual_relations and current_relation_count < relation_limit:
                relation_supplement = self._retrieve_visual_view(
                    "visual_relation",
                    plan,
                    query_tokens,
                    limit=max(1, relation_limit - current_relation_count + int(packer_config.get("temporal_max_supplement") or 0)),
                    video_filter=video_filter,
                )
                if relation_supplement:
                    added = append_unique(relation_supplement, limit=max(0, relation_limit - current_relation_count))
                    relation_supplement_used = added > 0
                    relation_supplement_count += added

            event_query = (
                self._event_supplement_query(query, plan, query_tokens)
                or "adaptive_event" in ((cost_plan or {}).get("view_order") or [])
                or "event" in ((cost_plan or {}).get("view_order") or [])
                or "visual_event" in ((cost_plan or {}).get("view_order") or [])
                or "fixed_window_event" in ((cost_plan or {}).get("view_order") or [])
            )
            current_event_count = sum(
                1 for item in candidate_evidence if self._evidence_type(item) in EVENT_FAMILY_VIEWS
            )
            event_limit = int(packer_config.get("event_supplement_min") or 0)
            if event_query and current_event_count < event_limit:
                needed = max(0, event_limit - current_event_count)
                if ablation["event_mode"] == "fixed_window":
                    temporal_supplement = self._retrieve_fixed_window_events(
                        query,
                        query_tokens,
                        limit=needed + int(packer_config.get("temporal_max_supplement") or 0),
                        fallback=True,
                        video_filter=video_filter,
                    )
                else:
                    temporal_supplement = self._retrieve_visual_view(
                        "visual_event",
                        plan,
                        query_tokens,
                        limit=needed + int(packer_config.get("temporal_max_supplement") or 0),
                        video_filter=video_filter,
                    )
                    temporal_supplement.extend(
                        self._retrieve_adaptive_events(
                            query,
                            query_tokens,
                            limit=needed + int(packer_config.get("temporal_max_supplement") or 0),
                            video_filter=video_filter,
                        )
                    )
                    temporal_supplement.extend(
                        self._retrieve_events(
                            query,
                            query_tokens,
                            limit=needed,
                            fallback=True,
                            video_filter=video_filter,
                        )
                    )
                if temporal_supplement:
                    temporal_supplement.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
                    added = append_unique(temporal_supplement, limit=needed)
                    event_supplement_used = added > 0
                    event_supplement_count += added

            if temporal_refinement_query:
                if not any(self._evidence_type(item) in {"track", "visual_track"} for item in candidate_evidence):
                    append_unique(
                        self._retrieve_tracks(
                            query,
                            query_tokens,
                            limit=1,
                            fallback=True,
                            video_filter=video_filter,
                        ),
                        limit=1,
                    )
                if not any(self._evidence_type(item) in {"target", "visual_object"} for item in candidate_evidence):
                    append_unique(
                        self._retrieve_targets(
                            query,
                            query_tokens,
                            limit=1,
                            fallback=True,
                            video_filter=video_filter,
                        ),
                        limit=1,
                    )

            if pedestrian_crosswalk_query:
                expansion = self._pedestrian_crosswalk_expansion(
                    query,
                    plan,
                    query_tokens,
                    video_filter=video_filter,
                )
                append_unique(
                    expansion,
                    limit=max(1, int(packer_config.get("temporal_max_supplement") or 6)),
                )
                pedestrian_crosswalk_expansion_used = True

            pack_result = evidence_packer.pack(
                query,
                candidate_evidence or v5_evidence,
                cost_plan={
                    **(cost_plan or {}),
                    "views_queried": execution_metrics.get("views_queried", []),
                    "views_skipped": execution_metrics.get("views_skipped", []),
                    "stop_reason": execution_metrics.get("stop_reason", ""),
                },
                query_plan=plan.to_dict(),
                package_metadata={
                    "video_filter_source": "explicit" if video_filter else "disabled",
                    "query_video_filter": sorted(video_filter) if video_filter else [],
                    "retrieved_count": retrieved_count,
                    "filtered_count": filtered_count,
                    "final_max_items": final_max_items,
                    "final_max_chars": final_max_chars,
                    "spatial_relation_supplement_used": relation_supplement_used,
                    "spatial_relation_supplement_count": relation_supplement_count,
                    "temporal_event_supplement_used": event_supplement_used,
                    "temporal_event_supplement_count": event_supplement_count,
                    "relation_supplement_used": relation_supplement_used,
                    "relation_supplement_count": relation_supplement_count,
                    "event_supplement_used": event_supplement_used,
                    "event_supplement_count": event_supplement_count,
                    "pedestrian_crosswalk_expansion_used": pedestrian_crosswalk_expansion_used,
                    "caption_fallback_used": bool(fusion_metrics.get("caption_fallback_used")),
                },
            )
            evidence = list(pack_result.get("packed_evidence") or [])
            dropped_evidence = list(pack_result.get("dropped_evidence") or [])
            packing_metadata = dict(pack_result.get("metadata") or {})
        else:
            pack_result = evidence_packer.pack(
                query,
                v5_evidence,
                cost_plan=cost_plan,
                query_plan=plan.to_dict(),
                package_metadata={"query_video_filter": sorted(video_filter) if video_filter else []},
            )
            packing_metadata = dict(pack_result.get("metadata") or {})
        if ablation["event_mode"] == "fixed_window":
            evidence, fixed_window_metadata = self._enforce_fixed_window_event_evidence(
                query,
                plan,
                query_tokens,
                evidence,
                [candidate_evidence, v5_evidence, fused_evidence, temporal_sequence_evidence],
                max_items=final_max_items,
                max_chars=final_max_chars,
                disabled_views=ablation["disabled_views"],
                video_filter=video_filter,
            )
            packing_metadata.update(fixed_window_metadata)
        print(
            f"[{EVIQUE_VERSION_LABEL}] evidence_packer="
            f"{'on' if packing_metadata.get('evidence_packer_enabled') else 'off'}"
        )
        print(f"[{EVIQUE_VERSION_LABEL}] evidence_token_budget={packing_metadata.get('evidence_token_budget')}")
        print(f"[{EVIQUE_VERSION_LABEL}] candidate_evidence_count={packing_metadata.get('candidate_evidence_count')}")
        print(f"[{EVIQUE_VERSION_LABEL}] packed_evidence_count={packing_metadata.get('packed_evidence_count')}")
        print(f"[{EVIQUE_VERSION_LABEL}] dropped_evidence_count={packing_metadata.get('dropped_evidence_count')}")
        print(f"[{EVIQUE_VERSION_LABEL}] estimated_packed_tokens={packing_metadata.get('estimated_packed_tokens')}")
        temporal_context_used = bool(temporal_sequence_evidence)
        traffic_flow_transition_query = (
            temporal_refinement_query
            and bool(query_tokens & (SCENE_TRAFFIC_CONTEXT_TERMS | SIGNAL_CAPTION_TERMS | VEHICLE_LABEL_TERMS | {"car", "cars"}))
            and bool(query_tokens & (TEMPORAL_CONTEXT_TERMS | EVENT_SUPPLEMENT_TERMS | {"start", "starts", "change", "changes"}))
        )
        insufficient_due_to_missing_visual_event = bool(
            visual_chain_attempted
            and not any(self._evidence_type(item) in {"visual_event", "fixed_window_event"} for item in evidence)
            and not fusion_metrics.get("caption_fallback_used")
        )
        subgraph = self._build_evidence_subgraph(evidence)
        answer_constraints = [
            f"This is {EVIQUE_VERSION_LABEL} evidence with video-aware retrieval enabled.",
            "Use only the supplied EVIQUE evidence package.",
            "Cite timestamps or segment ids when they are available.",
            "When video_id/source_vid is present, do not mix evidence from different videos.",
            "Start with a concise direct answer.",
            "Use the sections: Most likely matching moment; Supported visual/text details; Uncertain or weakly supported details.",
            "Give the strongest supported moment description before limitations.",
            "Prioritize people, clothing, actions, held objects, scene, motion, and interactions when present.",
            "If evidence supports only part of the query, state the confirmed part first and keep uncertainty brief.",
            "Fuse direct visual evidence with caption/scope context before answering.",
            "If direct visual evidence is limited but caption/scope evidence supports a general conclusion, answer generally and identify it as caption-supported.",
            "Keep limitations clear and non-repetitive, but do not over-compress when important evidence is needed.",
            "For spatial/nearby questions, explicitly use nearby_object_context evidence when present.",
            "If nearby_object_context says same-time objects are available, include that in the direct visual evidence section.",
            "If nearby_object_context says same-time nearby object unavailable, state that limitation clearly.",
            "Do not infer unsupported visual details beyond the evidence.",
        ]
        if temporal_refinement_query or temporal_context_used:
            answer_constraints.extend(
                [
                    "For temporal-ordering questions, answer first from Temporal sequence evidence in BEFORE / FOCAL EVENT / AFTER order.",
                    "Give a best-effort local temporal answer before uncertainty when the sequence evidence contains relevant local actions.",
                    "Treat caption_segment evidence as caption-supported best effort, not direct object tracking.",
                    "Only describe before/after order when the evidence provides timestamps, segment ids, or adjacent segment order.",
                    "If evidence only shows co-presence, do not turn it into interaction, yielding, or causality.",
                    "If identity tracking is weak, avoid claiming the same object persists across time.",
                    "If before/after evidence is missing or timestamps are absent, say the exact order cannot be reliably determined.",
                ]
            )
        if pedestrian_crosswalk_query:
            answer_constraints.extend(
                [
                    "For pedestrian, crosswalk, or vehicle-reaction questions, separate visually supported interaction from same-segment co-presence.",
                    "Do not claim yielding, reaction, or causality unless visual relation or event evidence directly supports it.",
                ]
            )
        return {
            "evique_version": EVIQUE_VERSION,
            "query": query,
            "query_video_filter": sorted(video_filter) if video_filter else [],
            "video_aware_retrieval": True,
            "plan": plan.to_dict(),
            "query_intents": dict(getattr(plan, "query_intents", {}) or {}),
            "route_reason": str(getattr(plan, "route_reason", "")),
            "visual_trigger_reason": str(getattr(plan, "visual_trigger_reason", "")),
            "cost_plan": cost_plan,
            "cost_planner_enabled": bool(cost_plan) and not ablation["disable_planner"],
            "ablation": ablation["metadata"],
            "event_mode": ablation["metadata"].get("event_mode"),
            "anchor_view": (cost_plan or {}).get("anchor_view"),
            "view_order": (cost_plan or {}).get("view_order") or selected_views,
            "max_rows_per_view": (cost_plan or {}).get("max_rows_per_view", {}),
            "max_rows_total": (cost_plan or {}).get("max_rows_total"),
            "views_queried": execution_metrics.get("views_queried", []),
            "views_skipped": execution_metrics.get("views_skipped", []),
            "stop_reason": execution_metrics.get("stop_reason", ""),
            "evidence_confidence": execution_metrics.get("evidence_confidence", 0.0),
            "evidence_coverage": execution_metrics.get("evidence_coverage", 0.0),
            "planner_debug_trace": execution_metrics.get("planner_debug_trace", []),
            "execution_event_mode": execution_metrics.get("event_mode") or (cost_plan or {}).get("event_mode") or ablation["metadata"].get("event_mode"),
            "evidence": evidence,
            "dropped_evidence": dropped_evidence,
            "evidence_packing_metadata": packing_metadata,
            **packing_metadata,
            "temporal_sequence_evidence": temporal_sequence_evidence,
            "relations": subgraph,
            "graph_stats": self.graph_stats,
            "retrieved_count": retrieved_count,
            "used_count": len(evidence),
            "filtered_count": filtered_count,
            "view_hit_counts": {view: len(rows) for view, rows in view_hits.items()},
            "visual_used": bool(visual_chain_found or visual_intent_evidence or cost_visual_evidence_count),
            "visual_chain_attempted": visual_chain_attempted,
            "visual_relations_enabled": self.visual_relations_enabled,
            "visual_relations_file_generated": self.visual_relations_file_generated,
            "visual_object_candidates": len(self.visual_objects),
            "visual_track_candidates": len(self.visual_tracks),
            "visual_relation_candidates": len(self.visual_relations),
            "visual_event_candidates": len(self.visual_events),
            "fixed_window_event_constructed_candidates": len(self.fixed_window_events),
            "visual_instance_chain_found": visual_chain_found,
            "visual_chain_evidence_count": len(visual_chain_evidence),
            "visual_intent_evidence_count": len(visual_intent_evidence) + cost_visual_evidence_count,
            "visual_failure_reason": visual_failure_reason,
            "caption_fallback_used": bool(fusion_metrics.get("caption_fallback_used")),
            "caption_context_evidence_count": int(fusion_metrics.get("caption_context_evidence_count", 0)),
            "temporal_relation_aligned_count": int(fusion_metrics.get("temporal_relation_aligned_count", 0)),
            "temporal_relation_fallback_count": int(fusion_metrics.get("temporal_relation_fallback_count", 0)),
            "temporal_context_used": temporal_context_used,
            "temporal_context_count": len(temporal_sequence_evidence),
            "temporal_context_reason": temporal_context_reason,
            "traffic_flow_transition_query": traffic_flow_transition_query,
            "insufficient_due_to_missing_visual_event": insufficient_due_to_missing_visual_event,
            "nearby_object_context_used": bool(fusion_metrics.get("nearby_object_context_used")),
            "nearby_object_context_candidate_count": int(fusion_metrics.get("nearby_object_context_candidate_count", 0)),
            "nearby_object_context_in_final_count": int(
                any(self._evidence_type(item) == "nearby_object_context" for item in evidence)
            ),
            "density_prompt_used": True,
            "caption_context_temporal_diversity": float(fusion_metrics.get("caption_context_temporal_diversity", 0.0)),
            "guidance": plan.guidance,
            "answer_constraints": answer_constraints,
        }

    def format_package(self, package: dict[str, Any]) -> str:
        plan = package.get("plan", {})
        lines = [
            "EVIQUE Compact Evidence Package",
            f"query_type: {plan.get('query_type', '')}",
            f"query_intents: {package.get('query_intents') or plan.get('query_intents', {})}",
            f"route_reason: {package.get('route_reason') or plan.get('route_reason', '')}",
            f"visual_trigger_reason: {package.get('visual_trigger_reason') or plan.get('visual_trigger_reason', '')}",
            f"selected_views: {', '.join(plan.get('selected_views', []))}",
            f"dependency_order: {', '.join(plan.get('dependency_order', []))}",
            f"view_weights: {plan.get('view_weights', {})}",
            f"cost_planner_enabled: {package.get('cost_planner_enabled', False)}",
            f"event_mode: {package.get('event_mode') or package.get('execution_event_mode') or ''}",
            f"anchor_view: {package.get('anchor_view') or ''}",
            f"view_order: {', '.join(package.get('view_order') or [])}",
            f"views_queried: {', '.join(package.get('views_queried') or [])}",
            f"stop_reason: {package.get('stop_reason', '')}",
            f"evidence_confidence: {float(package.get('evidence_confidence') or 0.0):.4f}",
            f"evidence_coverage: {float(package.get('evidence_coverage') or 0.0):.4f}",
            f"evidence_packer_enabled: {package.get('evidence_packer_enabled', False)}",
            f"packing_strategy: {package.get('packing_strategy', '')}",
            f"candidate_evidence_count: {package.get('candidate_evidence_count', 0)}",
            f"packed_evidence_count: {package.get('packed_evidence_count', 0)}",
            f"dropped_evidence_count: {package.get('dropped_evidence_count', 0)}",
            f"estimated_packed_tokens: {package.get('estimated_packed_tokens', 0)}",
            f"evidence_token_budget: {package.get('evidence_token_budget', 0)}",
            f"evidence_max_items: {package.get('evidence_max_items', 0)}",
            f"temporal_aware_packing_used: {package.get('temporal_aware_packing_used', False)}",
            f"temporal_anchor_segment: {package.get('temporal_anchor_segment') or ''}",
            f"temporal_before_count: {package.get('temporal_before_count', 0)}",
            f"temporal_focal_count: {package.get('temporal_focal_count', 0)}",
            f"temporal_after_count: {package.get('temporal_after_count', 0)}",
            f"relation_supplement_count: {package.get('relation_supplement_count', 0)}",
            f"event_supplement_count: {package.get('event_supplement_count', 0)}",
            f"pedestrian_crosswalk_expansion_used: {package.get('pedestrian_crosswalk_expansion_used', False)}",
            f"caption_fallback_used: {package.get('caption_fallback_used', False)}",
            f"guidance: {package.get('guidance', '')}",
            f"temporal_context_used: {package.get('temporal_context_used', False)}",
            f"temporal_context_count: {package.get('temporal_context_count', 0)}",
            f"temporal_context_reason: {package.get('temporal_context_reason', '')}",
            "",
            "Graph statistics:",
            str(package.get("graph_stats", {})),
            "",
            "Evidence:",
        ]
        for i, item in enumerate(package.get("evidence", []), start=1):
            provenance = item.get("provenance") or {}
            if isinstance(provenance, dict):
                video_name = provenance.get("video_name", "")
                video_id = item.get("video_id") or provenance.get("video_id") or ""
                source_vid = item.get("source_vid") or provenance.get("source_vid") or ""
                start = provenance.get("start_time")
                end = provenance.get("end_time")
                time_text = self._format_time_text(start, end)
                identity_parts = [
                    f"video_id={video_id}" if video_id else "",
                    f"source_vid={source_vid}" if source_vid else "",
                    f"video_name={video_name}" if video_name else "",
                ]
                provenance_text = f"{' '.join(part for part in identity_parts if part)} {time_text}".strip()
            else:
                provenance_text = str(provenance)
            lines.append(
                f"[{i}] view={item.get('view')} id={item.get('id')} score={_safe_float(item.get('score'), 0.0):.4f} "
                f"provenance={provenance_text}\n{item.get('text', '')}"
            )
        temporal_sequence = package.get("temporal_sequence_evidence") or []
        if temporal_sequence:
            lines.extend(
                [
                    "",
                    "Temporal sequence evidence:",
                    "Use this section first for a local before/focal/after answer. Caption sources are caption-supported best effort, not direct object tracks.",
                ]
            )
            for i, item in enumerate(temporal_sequence, start=1):
                role = item.get("temporal_role") or "UNKNOWN"
                source = item.get("source_view") or item.get("source") or item.get("view")
                segment_ids = item.get("segment_ids") or ([item.get("segment_id")] if item.get("segment_id") else [])
                segment_text = ",".join(str(segment_id) for segment_id in segment_ids if segment_id)
                start = item.get("start_time")
                end = item.get("end_time")
                timestamp = item.get("timestamp")
                time_text = self._format_time_text(start, end, timestamp)
                label = item.get("label") or item.get("category") or ""
                action = item.get("action") or item.get("event_type") or ""
                score = item.get("score")
                score_text = ""
                if score is not None:
                    try:
                        score_text = f" score={float(score):.4f}"
                    except (TypeError, ValueError):
                        score_text = ""
                support = item.get("support_text") or item.get("text") or ""
                lines.append(
                    f"[T{i}] ROLE={role}\n"
                    f"source_view={source} time={time_text or 'unknown'} segment_id={segment_text or 'unknown'} "
                    f"label={label or 'unknown'} action={action or 'unknown'}{score_text}\n"
                    f"support_text: {support}"
                )
        relations = package.get("relations", [])
        if relations:
            lines.extend(["", "Evidence relations:"])
            for relation in relations[:40]:
                lines.append(f"- {relation.get('source')} --{relation.get('relation')}--> {relation.get('target')}")
        lines.extend(["", "Answer constraints:"])
        lines.extend(f"- {constraint}" for constraint in package.get("answer_constraints", []))
        return "\n".join(lines)

    def _temporal_refinement_query(self, query: str, plan: Any, query_tokens: set[str]) -> bool:
        intents = getattr(plan, "query_intents", None) or {}
        query_type = str(getattr(plan, "query_type", ""))
        lowered = query.lower()
        return (
            query_type in TEMPORAL_REFINEMENT_QUERY_TYPES
            or any(bool(intents.get(key)) for key in TEMPORAL_REFINEMENT_INTENTS)
            or bool(query_tokens & (TEMPORAL_QUERY_TERMS | TEMPORAL_EVIDENCE_TERMS | EVENT_SUPPLEMENT_TERMS))
            or any(phrase in lowered for phrase in TEMPORAL_REFINEMENT_PHRASES)
        )

    def _relation_supplement_query(self, query: str, plan: Any, query_tokens: set[str]) -> bool:
        intents = getattr(plan, "query_intents", None) or {}
        lowered = query.lower()
        return (
            bool(intents.get("spatial_relation"))
            or bool(intents.get("multi_object_interaction"))
            or bool(query_tokens & RELATION_SUPPLEMENT_TERMS)
            or "relative position" in lowered
            or "close to each other" in lowered
        )

    def _event_supplement_query(self, query: str, plan: Any, query_tokens: set[str]) -> bool:
        intents = getattr(plan, "query_intents", None) or {}
        return (
            bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or bool(query_tokens & EVENT_SUPPLEMENT_TERMS)
            or self._temporal_refinement_query(query, plan, query_tokens)
        )

    def _pedestrian_crosswalk_query(self, query: str, plan: Any, query_tokens: set[str]) -> bool:
        lowered = query.lower()
        if "near crosswalk" in lowered or "nearby vehicle" in lowered:
            return True
        return bool(query_tokens & PEDESTRIAN_CROSSWALK_QUERY_TERMS) and bool(
            query_tokens & (PEDESTRIAN_LABEL_TERMS | CROSSWALK_LABEL_TERMS | VEHICLE_LABEL_TERMS)
        )

    def _refine_cost_plan_for_temporal(
        self,
        cost_plan: dict[str, Any],
        *,
        require_pedestrian_crosswalk: bool,
    ) -> dict[str, Any]:
        refined = dict(cost_plan)
        view_order = list(refined.get("view_order") or [])
        row_budgets = dict(refined.get("max_rows_per_view") or {})
        essential_views = ["adaptive_event", "event", "visual_event"]
        if self.visual_relations_enabled:
            essential_views.append("visual_relation")
        if require_pedestrian_crosswalk:
            essential_views = ["visual_object", "visual_event", "adaptive_event", "target"]
            if self.visual_relations_enabled:
                essential_views.insert(1, "visual_relation")
        for view in essential_views:
            if view not in view_order:
                view_order.append(view)
            row_budgets[view] = max(int(row_budgets.get(view) or 0), 2)
        supplement_budget = max(0, _env_int("EVIQUE_TEMPORAL_MAX_SUPPLEMENT", 6))
        refined["view_order"] = view_order
        refined["max_rows_per_view"] = row_budgets
        refined["max_rows_total"] = int(refined.get("max_rows_total") or 16) + supplement_budget
        refined["temporal_refinement"] = True
        refined["pedestrian_crosswalk_refinement"] = bool(require_pedestrian_crosswalk)
        return refined

    def _filter_visual_relation_cost_plan(self, cost_plan: dict[str, Any]) -> dict[str, Any]:
        if self.visual_relations_enabled:
            return cost_plan
        refined = dict(cost_plan)
        view_order = [view for view in list(refined.get("view_order") or []) if view != "visual_relation"]
        row_budgets = dict(refined.get("max_rows_per_view") or {})
        row_budgets.pop("visual_relation", None)
        estimated_costs = dict(refined.get("estimated_costs") or {})
        estimated_costs.pop("visual_relation", None)
        refined["view_order"] = view_order
        refined["max_rows_per_view"] = row_budgets
        refined["estimated_costs"] = estimated_costs
        if refined.get("anchor_view") == "visual_relation":
            refined["anchor_view"] = view_order[0] if view_order else "scope"
        return refined

    def _execute_cost_plan(
        self,
        query: str,
        plan: Any,
        cost_plan: dict[str, Any],
        query_tokens: set[str],
        *,
        video_filter: set[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        view_hits: dict[str, list[dict[str, Any]]] = {}
        views_queried: list[str] = []
        views_skipped: list[str] = []
        view_order = list(cost_plan.get("view_order") or [])
        max_rows_per_view = cost_plan.get("max_rows_per_view") or {}
        max_rows_total = int(cost_plan.get("max_rows_total") or self.max_evidence)
        min_confidence = float(cost_plan.get("min_confidence") or 0.65)
        selected: list[dict[str, Any]] = []
        stop_reason = "view_order_exhausted"
        print(f"[{EVIQUE_VERSION_LABEL}] cost_based_view_planner=on")
        print(f"[{EVIQUE_VERSION_LABEL}] anchor_view={cost_plan.get('anchor_view')}")
        print(f"[{EVIQUE_VERSION_LABEL}] view_order={view_order}")
        for view in view_order:
            if view == "visual_relation" and not self.visual_relations_enabled:
                views_skipped.append(view)
                continue
            estimated = (cost_plan.get("estimated_costs") or {}).get(view, {})
            if int(estimated.get("row_count") or 0) <= 0:
                views_skipped.append(view)
                continue
            remaining = max_rows_total - len(selected)
            if remaining <= 0:
                stop_reason = "max_rows_total_reached"
                break
            limit = min(int(max_rows_per_view.get(view) or remaining), remaining)
            if limit <= 0:
                views_skipped.append(view)
                continue
            rows = self._retrieve_planned_view(
                view,
                query,
                plan,
                query_tokens,
                limit=limit,
                video_filter=video_filter,
            )
            view_hits[view] = rows
            views_queried.append(view)
            selected.extend(rows)
            confidence, coverage = self._partial_evidence_quality(selected, view_hits, view_order)
            needs_boundary_views = bool(cost_plan.get("temporal_refinement"))
            needs_relation_view = bool(cost_plan.get("pedestrian_crosswalk_refinement") and self.visual_relations_enabled)
            queried_views = set(view_hits)
            has_event_view = bool(queried_views & EVENT_FAMILY_VIEWS)
            has_relation_view = "visual_relation" in queried_views
            missing_boundary_view = needs_boundary_views and (
                not has_event_view or (needs_relation_view and not has_relation_view)
            )
            if rows and confidence >= min_confidence and not missing_boundary_view:
                stop_reason = f"high_confidence_after_{view}"
                break
            if len(selected) >= max_rows_total:
                stop_reason = "max_rows_total_reached"
                break
        confidence, coverage = self._partial_evidence_quality(selected, view_hits, view_order)
        if not selected and views_queried:
            stop_reason = "no_positive_hits_after_planned_views"
        return view_hits, {
            "event_mode": cost_plan.get("event_mode") or ("fixed_window" if self._is_fixed_window_event_mode() else "adaptive"),
            "views_queried": views_queried,
            "views_skipped": views_skipped,
            "stop_reason": stop_reason,
            "evidence_confidence": confidence,
            "evidence_coverage": coverage,
            "planner_debug_trace": cost_plan.get("debug_trace") or [],
        }

    def _retrieve_planned_view(
        self,
        view: str,
        query: str,
        plan: Any,
        query_tokens: set[str],
        *,
        limit: int,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if view == "scope":
            return self._retrieve_scopes(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "target":
            return self._retrieve_targets(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "track":
            return self._retrieve_tracks(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "event":
            if self._is_fixed_window_event_mode():
                return self._retrieve_fixed_window_events(query, query_tokens, limit=limit, video_filter=video_filter)
            return self._retrieve_events(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "adaptive_event":
            return self._retrieve_adaptive_events(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "fixed_window_event":
            return self._retrieve_fixed_window_events(query, query_tokens, limit=limit, video_filter=video_filter)
        if view == "visual_relation" and not self.visual_relations_enabled:
            return []
        if view in {"visual_object", "visual_track", "visual_event", "visual_relation"}:
            return self._retrieve_visual_view(view, plan, query_tokens, limit=limit, video_filter=video_filter)
        return []

    def _partial_evidence_quality(
        self,
        items: list[dict[str, Any]],
        view_hits: dict[str, list[dict[str, Any]]],
        view_order: list[str],
    ) -> tuple[float, float]:
        if not items:
            return 0.0, 0.0
        scores = [max(0.0, _safe_float(item.get("score"), 0.0)) for item in items]
        normalized = [score / (score + 1.0) for score in scores]
        max_score = max(normalized) if normalized else 0.0
        avg_score = sum(normalized) / len(normalized) if normalized else 0.0
        confidence = min(1.0, 0.60 * max_score + 0.30 * avg_score + 0.10 * min(1.0, len(items) / 4.0))
        nonempty_views = {view for view, rows in view_hits.items() if rows}
        coverage = min(1.0, len(nonempty_views) / max(1, len(view_order)))
        return round(confidence, 6), round(coverage, 6)

    def build_minimal_evidence_package(
        self,
        query: str,
        plan: Any,
        ranked_candidates: list[dict[str, Any]],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        max_items = min(self.max_evidence, int(constraints.get("max_evidence", self.max_evidence)))
        max_chars = int(constraints.get("max_evidence_chars", min(self.token_budget, 8000)))
        max_item_chars = int(constraints.get("max_item_chars", 900))
        max_segments = int(constraints.get("max_segment_items", 6))
        max_tracks = int(constraints.get("max_track_items", 3))
        max_events = int(constraints.get("max_event_items", 3))
        allow_track = plan.query_type in TRACK_QUERY_TYPES
        allow_event = plan.query_type in EVENT_QUERY_TYPES

        segments: dict[str, dict[str, Any]] = {}
        tracks: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        direct_items: list[dict[str, Any]] = []
        seen_targets: dict[str, set[str]] = defaultdict(set)
        seen_tracks: set[str] = set()
        seen_events: set[str] = set()
        seen_direct: set[str] = set()

        def ensure_segment(segment_id: str, source: dict[str, Any]) -> dict[str, Any] | None:
            if not segment_id:
                return None
            if segment_id not in segments:
                if len(segments) >= max_segments:
                    return None
                scope = self.scopes_by_segment.get(segment_id, {})
                provenance = scope.get("provenance") or source.get("provenance") or {}
                text = scope.get("caption") or scope.get("text") or source.get("text", "")
                video_fields = self._video_fields_for_sources(scope, source, provenance)
                segments[segment_id] = {
                    "view": "segment",
                    "id": f"segment:{segment_id}",
                    "node_id": scope.get("node_id") or source.get("node_id"),
                    "segment_id": segment_id,
                    "segment_ids": [segment_id],
                    "start_time": scope.get("start_time") or provenance.get("start_time"),
                    "end_time": scope.get("end_time") or provenance.get("end_time"),
                    "short_text": shorten(text, max_item_chars),
                    "text": shorten(text, max_item_chars),
                    "targets": [],
                    "track_annotations": [],
                    "event_annotations": [],
                    "source_views": [],
                    "role": self._candidate_role(source, plan),
                    "score": _safe_float(source.get("score"), 0.0),
                    "provenance": provenance,
                    **video_fields,
                }
            item = segments[segment_id]
            item["score"] = max(_safe_float(item.get("score"), 0.0), _safe_float(source.get("score"), 0.0))
            source_view = source.get("view")
            if source_view and source_view not in item["source_views"]:
                item["source_views"].append(source_view)
            return item

        for candidate in ranked_candidates:
            view = str(candidate.get("view"))
            record = candidate.get("record") or {}
            segment_ids = self._candidate_segment_ids(candidate)
            if view in {"scope", "target"}:
                segment_id = segment_ids[0] if segment_ids else ""
                segment = ensure_segment(segment_id, candidate)
                if segment and view == "target":
                    label = str(record.get("label") or "").lower()
                    if label and label not in seen_targets[segment_id]:
                        seen_targets[segment_id].add(label)
                        segment["targets"].append(label)
                continue
            if view == "track" and allow_track and len(tracks) < max_tracks:
                track_id = str(candidate.get("id") or "")
                if track_id in seen_tracks:
                    continue
                seen_tracks.add(track_id)
                compact_segments = self._compact_track_segments(segment_ids)
                anchor = compact_segments[1] if len(compact_segments) == 3 else (compact_segments[0] if compact_segments else "")
                segment = ensure_segment(anchor, candidate) if anchor else None
                item = {
                    "view": "track",
                    "id": track_id,
                    "node_id": candidate.get("node_id"),
                    "track_id": track_id,
                    "label": record.get("label"),
                    "segment_ids": compact_segments,
                    "anchor_segment": anchor,
                    "short_text": shorten(self._track_text(record), max_item_chars),
                    "text": shorten(self._track_text(record), max_item_chars),
                    "role": self._candidate_role(candidate, plan),
                    "score": float(candidate.get("score", 0.0)),
                    "provenance": candidate.get("provenance") or {},
                    **self._video_fields_for_sources(record, candidate, candidate.get("provenance") or {}),
                }
                tracks.append(item)
                if segment:
                    segment["track_annotations"].append({"track_id": track_id, "label": record.get("label"), "segments": compact_segments})
                continue
            if view == "event" and allow_event and len(events) < max_events:
                event_id = str(candidate.get("id") or "")
                if event_id in seen_events:
                    continue
                seen_events.add(event_id)
                supporting_segments = segment_ids[:3]
                for segment_id in supporting_segments[:2]:
                    segment = ensure_segment(segment_id, candidate)
                    if segment:
                        segment["event_annotations"].append({"event_id": event_id, "event_type": record.get("event_type")})
                events.append(
                    {
                        "view": "event",
                        "id": event_id,
                        "node_id": candidate.get("node_id"),
                        "event_id": event_id,
                        "segment_ids": supporting_segments,
                        "short_text": shorten(self._event_text(record), max_item_chars),
                        "text": shorten(self._event_text(record), max_item_chars),
                        "role": self._candidate_role(candidate, plan),
                        "score": float(candidate.get("score", 0.0)),
                        "provenance": candidate.get("provenance") or {},
                        **self._video_fields_for_sources(record, candidate, candidate.get("provenance") or {}),
                    }
                )
                continue
            if view in {"adaptive_event", "fixed_window_event", "visual_object", "visual_track", "visual_event", "visual_relation"}:
                item_id = str(candidate.get("id") or candidate.get("node_id") or self._compact_evidence_preview(candidate))
                if item_id in seen_direct:
                    continue
                seen_direct.add(item_id)
                direct_items.append(self._compact_candidate(candidate, max_item_chars))

        evidence = sorted(segments.values(), key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)
        evidence.extend(sorted(tracks, key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True))
        evidence.extend(sorted(events, key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True))
        evidence.extend(sorted(direct_items, key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True))
        if not evidence:
            evidence = [self._compact_candidate(candidate, max_item_chars) for candidate in ranked_candidates[:max_items]]
        evidence = self._trim_evidence(evidence, max_items=max_items, max_chars=max_chars)
        return {
            "evidence": evidence,
            "segment_count": sum(1 for item in evidence if item.get("view") == "segment"),
            "track_count": sum(1 for item in evidence if item.get("view") == "track"),
            "event_count": sum(1 for item in evidence if item.get("view") == "event"),
        }

    def _video_fields_for_sources(self, *sources: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in ("video_id", "source_vid", "video_path", "video_name"):
                if source.get(key) and not fields.get(key):
                    fields[key] = source.get(key)
            provenance = source.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in ("video_id", "source_vid", "video_path", "video_name"):
                    if provenance.get(key) and not fields.get(key):
                        fields[key] = provenance.get(key)
        return fields

    def _video_filter_from_metadata(self, metadata: dict[str, Any] | None) -> set[str]:
        if self._disable_strict_video_filter(metadata):
            return set()
        values = metadata_video_values(metadata)
        if not values:
            return set()
        expanded = set(values)
        for identity in self.video_identities:
            identity_values = video_identity_values(identity)
            if values & identity_values:
                expanded.update(identity_values)
        return expanded

    def _disable_strict_video_filter(self, metadata: dict[str, Any] | None) -> bool:
        if _env_bool("EVIQUE_ABLATION_DISABLE_STRICT_VIDEO_FILTER", False):
            return True
        if not isinstance(metadata, dict):
            return False
        for key in (
            "disable_strict_video_filter",
            "disable_video_filter",
            "strict_video_filter_disabled",
            "evique_ablation_disable_strict_video_filter",
        ):
            value = metadata.get(key)
            if isinstance(value, bool):
                if value:
                    return True
            elif str(value or "").strip().lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _available_video_values(self) -> set[str]:
        values: set[str] = set()
        for identity in self.video_identities:
            values.update(video_identity_values(identity))
        for rows in (
            self.scopes,
            self.targets,
            self.tracks,
            self.events,
            self.adaptive_events,
            self.visual_objects,
            self.visual_tracks,
            self.visual_events,
            self.visual_relations,
            self.fixed_window_events,
        ):
            for row in rows:
                values.update(self._row_video_values(row))
        return values

    def _video_filter_matches_index(self, video_filter: set[str]) -> bool:
        if not video_filter:
            return True
        return bool(video_filter & self._available_video_values())

    def _row_video_values(self, row: dict[str, Any] | None) -> set[str]:
        if not isinstance(row, dict):
            return set()
        values = video_identity_values(row)
        record = row.get("record")
        if isinstance(record, dict):
            values.update(video_identity_values(record))
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            values.update(video_identity_values(provenance))
        return values

    def _row_matches_video_filter(self, row: dict[str, Any], video_filter: set[str] | None) -> bool:
        if not video_filter:
            return True
        row_values = self._row_video_values(row)
        return bool(row_values and row_values & video_filter)

    def _filter_rows_by_video(self, rows: list[dict[str, Any]], video_filter: set[str] | None) -> list[dict[str, Any]]:
        if not video_filter:
            return rows
        return [row for row in rows if self._row_matches_video_filter(row, video_filter)]

    def _rows_same_video(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_values = self._row_video_values(left)
        right_values = self._row_video_values(right)
        if left_values and right_values:
            return bool(left_values & right_values)
        return True

    def _retrieve_instance_spatial_temporal(
        self,
        query: str,
        plan: Any,
        query_tokens: set[str],
        *,
        video_filter: set[str] | None = None,
    ) -> dict[str, Any]:
        visual_objects = self._filter_rows_by_video(self.visual_objects, video_filter)
        visual_tracks = self._filter_rows_by_video(self.visual_tracks, video_filter)
        visual_relations = self._filter_rows_by_video(self.visual_relations, video_filter)
        visual_events = self._filter_rows_by_video(
            self.fixed_window_events if self._is_fixed_window_event_mode() else self.visual_events,
            video_filter,
        )
        debug: dict[str, Any] = {
            "event_mode": "fixed_window" if self._is_fixed_window_event_mode() else "adaptive",
            "visual_objects": len(visual_objects),
            "visual_tracks": len(visual_tracks),
            "visual_relations": len(visual_relations),
            "visual_events": len(visual_events),
        }

        def fail(reason: str) -> dict[str, Any]:
            return {"found": False, "evidence": [], "failure_reason": reason, "debug": debug}

        def instance_candidate_topk() -> int:
            try:
                return int(os.getenv("EVIQUE_INSTANCE_CANDIDATE_TOPK", "50"))
            except ValueError:
                return 50

        def row_timestamp(row: dict[str, Any]) -> float | None:
            for key in ("timestamp", "time", "frame_timestamp", "start_time"):
                value = row.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            provenance = row.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in ("timestamp", "start_time"):
                    value = provenance.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            pass
            return None

        def row_attributes(row: dict[str, Any]) -> list[str]:
            raw_attrs = row.get("attributes") or []
            if isinstance(raw_attrs, dict):
                raw_attrs = list(raw_attrs.values())
            if isinstance(raw_attrs, str):
                raw_attrs = raw_attrs.replace(",", " ").split()
            attrs = [str(attr).strip().lower() for attr in raw_attrs if str(attr).strip()]
            color = str(row.get("color") or "").strip().lower()
            if color and color != "unknown":
                attrs.append(color)
            return self._dedupe(attrs)

        def add_failure(
            stage: str,
            target: dict[str, Any] | None = None,
            anchor_event: dict[str, Any] | None = None,
            anchor_target: dict[str, Any] | None = None,
            detail: str | None = None,
        ) -> None:
            failures = debug.setdefault("sample_failures", [])
            if len(failures) >= 10:
                return
            target = target or {}
            anchor_event = anchor_event or {}
            anchor_target = anchor_target or {}
            failures.append(
                {
                    "stage": stage,
                    "detail": detail,
                    "target_object_id": target.get("object_id") or target.get("id"),
                    "target_track_id": target.get("track_id"),
                    "target_timestamp": row_timestamp(target),
                    "target_attrs": sorted(row_attributes(target)) if target else [],
                    "anchor_event_id": anchor_event.get("event_id") or anchor_event.get("id"),
                    "anchor_event_type": anchor_event.get("event_type"),
                    "anchor_timestamp": row_timestamp(anchor_event),
                    "anchor_target_object_id": anchor_target.get("object_id") or anchor_target.get("id"),
                    "anchor_target_track_id": anchor_target.get("track_id"),
                    "anchor_target_timestamp": row_timestamp(anchor_target),
                    "anchor_target_attrs": sorted(row_attributes(anchor_target)) if anchor_target else [],
                }
            )

        def lookup_object(object_id: Any) -> dict[str, Any] | None:
            if object_id is None:
                return None
            row = self.visual_objects_by_id.get(object_id) or self.visual_objects_by_id.get(str(object_id))
            return row if row and self._row_matches_video_filter(row, video_filter) else None

        def lookup_track(track_id: Any) -> dict[str, Any] | None:
            if track_id is None:
                return None
            row = self.visual_tracks_by_id.get(track_id) or self.visual_tracks_by_id.get(str(track_id))
            return row if row and self._row_matches_video_filter(row, video_filter) else None

        def id_set(row: dict[str, Any], scalar_keys: tuple[str, ...], list_keys: tuple[str, ...]) -> set[str]:
            values = {str(row.get(key)) for key in scalar_keys if row.get(key)}
            for key in list_keys:
                values.update(str(value) for value in row.get(key) or [] if value)
            return values

        def query_attributes() -> set[str]:
            return query_tokens & {
                "red",
                "blue",
                "white",
                "black",
                "gray",
                "grey",
                "green",
                "yellow",
                "brown",
                "purple",
                "dark",
                "light",
                "small",
                "large",
            }

        required_attrs = query_attributes()

        def attributes_compatible(row: dict[str, Any], required: set[str]) -> bool:
            if not required:
                return True
            attrs = set(row_attributes(row))
            color = str(row.get("color") or "").lower()
            if color and color != "unknown":
                attrs.add(color)
            return bool(attrs & required)

        def time_close(a: float | None, b: float | None, max_gap: float = 8.0) -> bool:
            if a is None or b is None:
                return True
            return abs(a - b) <= max_gap

        def resolve_anchor_target_object(target: dict[str, Any], anchor_event: dict[str, Any]) -> dict[str, Any] | None:
            anchor_timestamp = row_timestamp(anchor_event)
            for key in ("actor_object_id", "object_id", "target_object_id"):
                anchor_object = lookup_object(anchor_event.get(key))
                if anchor_object and attributes_compatible(anchor_object, required_attrs):
                    return anchor_object

            event_track_ids = id_set(
                anchor_event,
                ("track_id", "target_track_id", "actor_track_id", "related_track_id"),
                ("track_ids", "involved_track_ids", "related_track_ids"),
            )
            if target.get("track_id"):
                event_track_ids.add(str(target.get("track_id")))
            track_candidates = [
                row
                for row in visual_objects
                if row.get("track_id")
                and str(row.get("track_id")) in event_track_ids
                and attributes_compatible(row, required_attrs)
            ]

            def time_distance(row: dict[str, Any]) -> float:
                row_time = row_timestamp(row)
                if row_time is None or anchor_timestamp is None:
                    return float("inf")
                return abs(row_time - anchor_timestamp)

            if track_candidates:
                track_candidates.sort(key=time_distance)
                if time_close(row_timestamp(track_candidates[0]), anchor_timestamp):
                    return track_candidates[0]

            target_timestamp = row_timestamp(target)
            if time_close(target_timestamp, anchor_timestamp) and attributes_compatible(target, required_attrs):
                return target
            return None

        def object_score(row: dict[str, Any]) -> float:
            label = str(row.get("label") or "").lower()
            color = str(row.get("color") or "").lower()
            attrs = set(row_attributes(row))
            text = " ".join(
                str(value)
                for value in [
                    label,
                    color,
                    " ".join(attrs),
                    row.get("text"),
                    row.get("evidence_text"),
                    row.get("summary"),
                ]
                if value
            )
            score = overlap_score(query_tokens, text)
            if label and label in query_tokens:
                score += 3.0
            if set(label.replace("_", " ").split()) & query_tokens:
                score += 1.5
            if color and color != "unknown" and color in query_tokens:
                score += 2.0
            score += 2.0 * len(attrs & query_tokens)
            if {"object", "item", "thing"} & query_tokens:
                score += 0.5
            if label == "person" and {"people", "player", "defender", "performer"} & query_tokens:
                score += 1.5
            if label in {"car", "truck", "bus", "bicycle", "motorcycle"} and "vehicle" in query_tokens:
                score += 1.5
            if label in {"dog", "cat", "bird", "horse", "sheep", "cow"} and "animal" in query_tokens:
                score += 1.5
            if label == "sports ball" and "ball" in query_tokens:
                score += 1.5
            return score

        def choose_event(target: dict[str, Any]) -> dict[str, Any] | None:
            target_track_ids = {str(target.get("track_id"))} if target.get("track_id") else set()
            target_object_ids = {str(value) for value in (target.get("object_id"), target.get("id")) if value}
            target_timestamp = row_timestamp(target)
            scored: list[tuple[float, dict[str, Any]]] = []
            for event in visual_events:
                event_track_ids = id_set(
                    event,
                    ("track_id", "target_track_id", "actor_track_id", "related_track_id"),
                    ("track_ids", "involved_track_ids", "related_track_ids"),
                )
                event_object_ids = id_set(
                    event,
                    ("object_id", "target_object_id", "actor_object_id", "related_object_id"),
                    ("object_ids", "involved_object_ids", "related_object_ids"),
                )
                event_text = " ".join(
                    str(value)
                    for value in [
                        event.get("event_type"),
                        event.get("summary"),
                        event.get("evidence_text"),
                        " ".join(str(tag) for tag in event.get("action_tags") or []),
                    ]
                    if value
                )
                score = overlap_score(query_tokens, event_text)
                if target_track_ids & event_track_ids:
                    score += 3.0
                if target_object_ids & event_object_ids:
                    score += 2.0
                event_timestamp = row_timestamp(event)
                if target_timestamp is not None and event_timestamp is not None:
                    score += max(0.0, 1.0 - abs(target_timestamp - event_timestamp) / 10.0)
                if score > 0 or target_track_ids & event_track_ids or target_object_ids & event_object_ids:
                    scored.append((score, event))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                return scored[0][1]
            if target_timestamp is None:
                return None
                timed = [
                    (abs(row_timestamp(event) - target_timestamp), event)
                    for event in visual_events
                    if row_timestamp(event) is not None
                ]
            timed.sort(key=lambda item: item[0])
            return timed[0][1] if timed else None

        def choose_relation(target: dict[str, Any], anchor_event: dict[str, Any]) -> dict[str, Any] | None:
            target_object_id = target.get("object_id") or target.get("id")
            target_track_id = target.get("track_id")
            relations = list(self.visual_relations_by_target.get(str(target_object_id), [])) if target_object_id else []
            relations = [
                relation for relation in relations
                if self._row_matches_video_filter(relation, video_filter) and self._rows_same_video(target, relation)
            ]
            if not relations and target_track_id:
                relations = [
                    relation
                    for relation in visual_relations
                    if str(relation.get("target_track_id") or "") == str(target_track_id)
                    and self._rows_same_video(target, relation)
                ]
            if not relations:
                return None
            anchor_timestamp = row_timestamp(anchor_event)
            proximity_terms = {"nearest", "closest", "nearby", "near", "next", "beside", "close"}
            scored: list[tuple[float, dict[str, Any]]] = []
            for relation in relations:
                relation_type = str(relation.get("relation_type") or "").lower()
                score = 0.0
                if relation_type == "nearest_to":
                    score += 2.0
                elif relation_type == "overlap_or_near":
                    score += 1.5
                elif relation_type in {"left_of", "right_of", "above", "below"}:
                    score += 1.0
                elif relation_type == "same_frame":
                    score += 0.5
                if proximity_terms & query_tokens and relation_type in {"nearest_to", "overlap_or_near"}:
                    score += 2.0
                relation_timestamp = row_timestamp(relation)
                if anchor_timestamp is not None and relation_timestamp is not None:
                    gap = abs(anchor_timestamp - relation_timestamp)
                    if gap > 12.0:
                        continue
                    score += max(0.0, 2.0 - gap / 3.0)
                else:
                    score += 0.1
                scored.append((score, relation))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1] if scored else None

        def find_anchor_target_with_relation(
            target: dict[str, Any],
            anchor_event: dict[str, Any],
            current_anchor_target: dict[str, Any] | None,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
            if current_anchor_target and attributes_compatible(current_anchor_target, required_attrs):
                relation = choose_relation(current_anchor_target, anchor_event)
                if relation:
                    debug["relation_fallback_candidate_count_last"] = 1
                    debug["relation_fallback_scored_count_last"] = 1
                    debug["relation_fallback_candidate_samples_last"] = [
                        {
                            "object_id": current_anchor_target.get("object_id") or current_anchor_target.get("id"),
                            "track_id": current_anchor_target.get("track_id"),
                            "timestamp": row_timestamp(current_anchor_target),
                            "attrs": sorted(row_attributes(current_anchor_target)),
                            "has_relation": True,
                        }
                    ]
                    return current_anchor_target, relation, False

            anchor_timestamp = row_timestamp(anchor_event)
            candidates_by_id: dict[str, dict[str, Any]] = {}

            def add_candidate(row: dict[str, Any] | None) -> None:
                if not row or not attributes_compatible(row, required_attrs):
                    return
                if not time_close(row_timestamp(row), anchor_timestamp, max_gap=12.0):
                    return
                object_id = row.get("object_id") or row.get("id")
                if object_id:
                    candidates_by_id[str(object_id)] = row

            add_candidate(current_anchor_target)
            for key in ("actor_object_id", "object_id", "target_object_id"):
                add_candidate(lookup_object(anchor_event.get(key)))

            event_track_ids = id_set(
                anchor_event,
                ("track_id", "target_track_id", "actor_track_id", "related_track_id"),
                ("track_ids", "involved_track_ids", "related_track_ids"),
            )
            for row in (target, current_anchor_target):
                if row and row.get("track_id"):
                    event_track_ids.add(str(row.get("track_id")))
            for row in visual_objects:
                if row.get("track_id") and str(row.get("track_id")) in event_track_ids:
                    add_candidate(row)

            debug["relation_fallback_candidate_count_last"] = len(candidates_by_id)
            scored: list[tuple[float, float, float, dict[str, Any], dict[str, Any]]] = []
            candidate_samples: list[dict[str, Any]] = []
            for candidate in candidates_by_id.values():
                relation = choose_relation(candidate, anchor_event)
                if len(candidate_samples) < 5:
                    candidate_samples.append(
                        {
                            "object_id": candidate.get("object_id") or candidate.get("id"),
                            "track_id": candidate.get("track_id"),
                            "timestamp": row_timestamp(candidate),
                            "attrs": sorted(row_attributes(candidate)),
                            "has_relation": bool(relation),
                        }
                    )
                if not relation:
                    continue
                candidate_timestamp = row_timestamp(candidate)
                time_gap = (
                    abs(candidate_timestamp - anchor_timestamp)
                    if candidate_timestamp is not None and anchor_timestamp is not None
                    else 0.0
                )
                relation_type = str(relation.get("relation_type") or "").lower()
                relation_bonus = 2.0 if relation_type == "nearest_to" else 1.0 if relation_type == "overlap_or_near" else 0.5
                try:
                    distance = float(relation.get("distance_pixels"))
                except (TypeError, ValueError):
                    distance = float("inf")
                scored.append((-time_gap, relation_bonus, -distance, candidate, relation))

            debug["relation_fallback_scored_count_last"] = len(scored)
            debug["relation_fallback_candidate_samples_last"] = candidate_samples
            if not scored:
                return None, None, False
            scored.sort(key=lambda item: item[:3], reverse=True)
            best_target = scored[0][3]
            best_relation = scored[0][4]
            current_id = (current_anchor_target.get("object_id") or current_anchor_target.get("id")) if current_anchor_target else None
            best_id = best_target.get("object_id") or best_target.get("id")
            return best_target, best_relation, str(best_id) != str(current_id)

        def choose_track(related: dict[str, Any], related_object_id: Any, relation: dict[str, Any]) -> dict[str, Any] | None:
            track_ids = [
                related.get("track_id"),
                relation.get("related_track_id"),
                relation.get("neighbor_track_id"),
            ]
            track_ids.extend(related.get("track_ids") or [])
            for track_id in track_ids:
                track = lookup_track(track_id)
                if track:
                    return track
            for track in self.visual_tracks_by_object_id.get(str(related_object_id), []):
                if not self._row_matches_video_filter(track, video_filter):
                    continue
                return track
            return None

        def compact_points(track: dict[str, Any], anchor_timestamp: float | None) -> list[dict[str, Any]]:
            raw_points = []
            for key in ("points", "positions", "bbox_sequence", "observations"):
                value = track.get(key)
                if isinstance(value, list) and value:
                    raw_points = value
                    break
            points: list[dict[str, Any]] = []
            for point in raw_points:
                if isinstance(point, dict):
                    points.append(
                        {
                            "timestamp": point.get("timestamp") or point.get("time"),
                            "frame_id": point.get("frame_id"),
                            "bbox": point.get("bbox"),
                            "center": point.get("center") or point.get("bbox_center"),
                            "segment_id": point.get("segment_id"),
                            "image_path": point.get("image_path"),
                            "video_id": point.get("video_id"),
                            "source_vid": point.get("source_vid"),
                            "video_path": point.get("video_path"),
                        }
                    )
                else:
                    points.append({"value": point})
            if len(points) <= 3:
                return points
            anchor_point = None
            if anchor_timestamp is not None:
                timed_points = []
                for point in points:
                    try:
                        timed_points.append((abs(float(point.get("timestamp")) - anchor_timestamp), point))
                    except (TypeError, ValueError):
                        pass
                if timed_points:
                    timed_points.sort(key=lambda item: item[0])
                    anchor_point = timed_points[0][1]
            selected = [points[0], anchor_point or points[len(points) // 2], points[-1]]
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for point in selected:
                key = repr(point)
                if key not in seen:
                    seen.add(key)
                    deduped.append(point)
            return deduped

        def object_item(row: dict[str, Any], role: str, score: float, prefix: str) -> dict[str, Any]:
            object_id = row.get("object_id") or row.get("id")
            attrs = row_attributes(row)
            text = (
                f"{prefix}: attributes={attrs or 'unknown'}, label={row.get('label') or 'unknown'}, "
                f"object_id={object_id}, track_id={row.get('track_id') or 'unknown'}, "
                f"frame_id={row.get('frame_id') or 'unknown'}, timestamp={row.get('timestamp') or 'unknown'}."
            )
            return {
                "view": "visual_object",
                "id": object_id,
                "node_id": object_id,
                "score": float(score),
                "text": shorten(text, 950),
                "short_text": shorten(text, 260),
                "role": role,
                "object_id": object_id,
                "track_id": row.get("track_id"),
                "label": row.get("label"),
                "attributes": attrs,
                "bbox": row.get("bbox"),
                "center": row.get("center") or row.get("bbox_center"),
                "timestamp": row.get("timestamp"),
                "frame_id": row.get("frame_id"),
                "segment_id": row.get("segment_id"),
                "image_path": row.get("image_path"),
                "provenance": row.get("provenance") or {},
                **self._video_fields_for_sources(row),
            }

        def event_item(row: dict[str, Any]) -> dict[str, Any]:
            event_id = row.get("event_id") or row.get("id")
            event_type = row.get("event_type") or "unknown"
            summary = row.get("summary") or row.get("evidence_text") or row.get("text") or ""
            event_view = "fixed_window_event" if self._is_fixed_window_event_mode() else "visual_event"
            event_label = "fixed-window event" if event_view == "fixed_window_event" else "visual event"
            text = (
                f"Anchor {event_label}: event_type={event_type}, event_id={event_id}, "
                f"track_id={row.get('track_id') or row.get('target_track_id') or 'unknown'}, "
                f"timestamp={row.get('timestamp') or row.get('start_time') or 'unknown'}, summary={summary or 'unknown'}."
            )
            return {
                "view": event_view,
                "id": event_id,
                "node_id": event_id,
                "score": max(1.0, overlap_score(query_tokens, f"{event_type} {summary}")),
                "text": shorten(text, 950),
                "short_text": shorten(text, 260),
                "role": "event_binding",
                "event_id": event_id,
                "track_id": row.get("track_id") or row.get("target_track_id"),
                "label": row.get("label"),
                "attributes": row.get("attributes"),
                "bbox": row.get("bbox"),
                "center": row.get("center") or row.get("bbox_center"),
                "timestamp": row.get("timestamp"),
                "frame_id": row.get("frame_id"),
                "segment_id": row.get("segment_id"),
                "image_path": row.get("image_path"),
                "provenance": row.get("provenance") or {},
                **self._video_fields_for_sources(row),
            }

        def relation_item(row: dict[str, Any], target: dict[str, Any], related: dict[str, Any]) -> dict[str, Any]:
            relation_id = row.get("relation_id") or row.get("id")
            related_object_id = row.get("related_object_id") or row.get("neighbor_object_id")
            target_object_id = row.get("target_object_id") or target.get("object_id") or target.get("id")
            relation_type = row.get("relation_type") or "unknown"
            text = (
                f"Spatial relation near anchor: related object {related_object_id} is {relation_type} "
                f"target object {target_object_id}, timestamp={row.get('timestamp') or 'unknown'}, "
                f"distance_pixels={row.get('distance_pixels') or 'unknown'}."
            )
            return {
                "view": "visual_relation",
                "id": relation_id,
                "node_id": relation_id,
                "score": 1.0,
                "text": shorten(text, 950),
                "short_text": shorten(text, 260),
                "role": "spatial_relation",
                "object_id": target_object_id,
                "track_id": row.get("target_track_id"),
                "relation_id": relation_id,
                "label": related.get("label"),
                "attributes": row_attributes(related),
                "bbox": related.get("bbox"),
                "center": related.get("center") or related.get("bbox_center"),
                "timestamp": row.get("timestamp"),
                "frame_id": row.get("frame_id"),
                "segment_id": row.get("segment_id"),
                "image_path": row.get("image_path") or related.get("image_path"),
                "provenance": row.get("provenance") or {},
                **self._video_fields_for_sources(row, target, related),
            }

        def track_item(track: dict[str, Any], related: dict[str, Any], anchor_timestamp: float | None) -> dict[str, Any]:
            track_id = track.get("track_id") or track.get("id")
            points = compact_points(track, anchor_timestamp)
            motion_summary = track.get("motion_summary") or track.get("direction") or track.get("summary") or "unknown"
            text = (
                f"Related object motion evidence after/around anchor: track_id={track_id}, "
                f"motion_summary={motion_summary}, compact_points={points or 'unknown'}."
            )
            return {
                "view": "visual_track",
                "id": track_id,
                "node_id": track_id,
                "score": 1.0,
                "text": shorten(text, 950),
                "short_text": shorten(text, 260),
                "role": "motion_evidence",
                "object_id": related.get("object_id") or related.get("id"),
                "track_id": track_id,
                "label": track.get("label") or related.get("label"),
                "attributes": row_attributes(related),
                "bbox": related.get("bbox"),
                "center": related.get("center") or related.get("bbox_center"),
                "timestamp": anchor_timestamp,
                "frame_id": related.get("frame_id"),
                "segment_id": related.get("segment_id") or track.get("segment_id"),
                "image_path": related.get("image_path") or track.get("image_path"),
                "compact_points": points,
                "provenance": track.get("provenance") or related.get("provenance") or {},
                **self._video_fields_for_sources(track, related),
            }

        if not visual_objects or not visual_tracks or not visual_relations:
            return fail("visual_evidence_unavailable")
        if not visual_events:
            return fail("anchor_event_not_found")

        candidates = [(object_score(row), row) for row in visual_objects]
        candidates = [(score, row) for score, row in candidates if score > 0]
        candidates.sort(key=lambda item: item[0], reverse=True)
        debug["target_candidates"] = len(candidates)
        if not candidates:
            return fail("target_object_not_found")

        candidate_topk = instance_candidate_topk()
        debug["candidate_topk"] = candidate_topk
        debug.update(
            {
                "targets_checked": 0,
                "anchor_events_found": 0,
                "anchor_targets_found": 0,
                "relations_found": 0,
                "related_objects_found": 0,
                "related_tracks_found": 0,
                "sample_failures": [],
            }
        )
        saw_anchor = False
        saw_relation = False
        saw_related = False
        for target_score, target in candidates[:candidate_topk]:
            debug["targets_checked"] += 1
            anchor_event = choose_event(target)
            if not anchor_event:
                continue
            saw_anchor = True
            debug["anchor_events_found"] += 1

            anchor_target = resolve_anchor_target_object(target, anchor_event)
            if not anchor_target:
                continue
            if not attributes_compatible(anchor_target, required_attrs):
                continue
            debug["anchor_targets_found"] += 1

            anchor_target, relation, anchor_target_relation_fallback_used = find_anchor_target_with_relation(
                target,
                anchor_event,
                anchor_target,
            )
            if not anchor_target or not relation:
                continue
            saw_relation = True
            debug["relations_found"] += 1

            related_object_id = relation.get("related_object_id") or relation.get("neighbor_object_id")
            related = lookup_object(related_object_id)
            if not related:
                continue
            saw_related = True
            debug["related_objects_found"] += 1

            related_track = choose_track(related, related_object_id, relation)
            if not related_track:
                continue
            debug["related_tracks_found"] += 1

            anchor_timestamp = row_timestamp(anchor_event)
            relation_timestamp = row_timestamp(relation)
            anchor_relation_time_gap = (
                abs(anchor_timestamp - relation_timestamp)
                if anchor_timestamp is not None and relation_timestamp is not None
                else None
            )

            debug.update(
                {
                    "target_candidate_object_id": target.get("object_id") or target.get("id"),
                    "anchor_target_object_id": anchor_target.get("object_id") or anchor_target.get("id"),
                    "target_object_id": anchor_target.get("object_id") or anchor_target.get("id"),
                    "anchor_event_id": anchor_event.get("event_id") or anchor_event.get("id"),
                    "anchor_timestamp": anchor_timestamp,
                    "anchor_target_timestamp": row_timestamp(anchor_target),
                    "anchor_target_relation_fallback_used": anchor_target_relation_fallback_used,
                    "relation_id": relation.get("relation_id") or relation.get("id"),
                    "relation_timestamp": relation_timestamp,
                    "anchor_relation_time_gap": anchor_relation_time_gap,
                    "related_object_id": related_object_id,
                    "related_track_id": related_track.get("track_id") or related_track.get("id"),
                }
            )

            return {
                "found": True,
                "evidence": [
                    object_item(anchor_target, "attribute_grounding", target_score, "Target object candidate"),
                    event_item(anchor_event),
                    relation_item(relation, anchor_target, related),
                    object_item(related, "spatial_relation", 1.0, "Related object candidate"),
                    track_item(related_track, related, anchor_timestamp),
                ],
                "failure_reason": None,
                "debug": debug,
            }

        if not saw_anchor:
            return fail("anchor_event_not_found")
        if not saw_relation:
            return fail("spatial_relation_not_found")
        if not saw_related:
            return fail("related_object_not_found")
        return fail("related_track_motion_not_found")

    def _fuse_and_naturalize_evidence(
        self,
        query: str,
        plan: Any,
        evidence: list[dict[str, Any]],
        *,
        video_filter: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metrics = {
            "caption_fallback_used": False,
            "caption_context_evidence_count": 0,
            "caption_context_temporal_diversity": 0.0,
            "temporal_relation_aligned_count": 0,
            "temporal_relation_fallback_count": 0,
            "nearby_object_context_used": False,
            "nearby_object_context_candidate_count": 0,
        }
        max_items = min(self.max_evidence, int(getattr(plan, "constraints", {}).get("max_evidence", self.max_evidence)))
        query_tokens = set(getattr(plan, "query_terms", []) or [])
        visual_events = [item for item in evidence if self._evidence_type(item) in {"visual_event", "fixed_window_event"}]

        aligned_relation_items: list[dict[str, Any]] = []
        limitation_items: list[dict[str, Any]] = []
        seen_relations: set[str] = set()
        for event in visual_events:
            relation, status = self._best_temporal_relation_for_event(event)
            if relation is None:
                limitation_items.append(self._temporal_limitation_item(event))
                metrics["temporal_relation_fallback_count"] += 1
                continue
            relation_item = self._visual_evidence_item("visual_relation", relation, 1.0)
            relation_item["temporal_alignment"] = status
            relation_item["anchor_event_id"] = event.get("event_id") or event.get("id")
            relation_item = self._naturalize_evidence_item(relation_item)
            relation_key = str(relation_item.get("id") or self._compact_evidence_preview(relation_item))
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            aligned_relation_items.append(relation_item)
            if status == "aligned":
                metrics["temporal_relation_aligned_count"] += 1
            else:
                metrics["temporal_relation_fallback_count"] += 1

        naturalized: list[dict[str, Any]] = []
        for item in evidence:
            if visual_events and self._evidence_type(item) == "visual_relation":
                continue
            naturalized.append(self._naturalize_evidence_item(item))

        nearby_items: list[dict[str, Any]] = []
        if self._is_spatial_context_query(query_tokens):
            for event in visual_events[:2]:
                nearby_item, candidate_count = self._nearby_object_context_for_event(event, video_filter=video_filter)
                metrics["nearby_object_context_candidate_count"] += candidate_count
                if nearby_item:
                    nearby_items.append(nearby_item)
                    if candidate_count > 0:
                        metrics["nearby_object_context_used"] = True

        caption_items: list[dict[str, Any]] = []
        if self._needs_caption_fallback(query, query_tokens):
            caption_items = self._caption_fallback_evidence(
                query_tokens,
                evidence,
                max_items=max_items,
                video_filter=video_filter,
            )
            if caption_items:
                metrics["caption_fallback_used"] = True
                metrics["caption_context_evidence_count"] = len(caption_items)
                buckets = {item.get("temporal_bucket") for item in caption_items if item.get("temporal_bucket") is not None}
                metrics["caption_context_temporal_diversity"] = len(buckets) / max(len(caption_items), 1)

        visual_core = [
            item
            for item in naturalized
            if self._evidence_type(item) in {"visual_event", "fixed_window_event", "visual_track", "visual_object"}
        ]
        non_visual_context = [
            item
            for item in naturalized
            if self._evidence_type(item) not in {"visual_event", "fixed_window_event", "visual_track", "visual_object", "visual_relation"}
        ]
        trailing_relations = [
            item
            for item in naturalized
            if self._evidence_type(item) == "visual_relation"
        ]
        return (
            self._dedupe_evidence_items(
                caption_items
                + visual_core
                + aligned_relation_items
                + nearby_items
                + limitation_items
                + non_visual_context
                + trailing_relations
            ),
            metrics,
        )

    def _needs_caption_fallback(self, query: str, query_tokens: set[str]) -> bool:
        lowered = query.lower()
        strong = bool(query_tokens & CAPTION_FALLBACK_TERMS) or any(
            phrase in lowered for phrase in CAPTION_FALLBACK_PHRASES
        )
        contextual = bool(query_tokens & CAPTION_FALLBACK_CONTEXT_TERMS) and bool(
            query_tokens & OPEN_ENDED_CONTEXT_TERMS
        )
        return strong or contextual

    def _is_spatial_context_query(self, query_tokens: set[str]) -> bool:
        return bool(query_tokens & SPATIAL_CONTEXT_TERMS)

    def _caption_fallback_evidence(
        self,
        query_tokens: set[str],
        evidence: list[dict[str, Any]],
        *,
        max_items: int,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        target_count = min(4, max(2, max_items // 4 if max_items >= 8 else max_items))
        if target_count <= 0:
            return []
        anchor_segments = {
            str(segment_id)
            for item in evidence
            for segment_id in self._candidate_segment_ids(item)
            if segment_id
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for scope in self.scopes:
            if not self._row_matches_video_filter(scope, video_filter):
                continue
            text = " ".join(
                str(value)
                for value in [
                    scope.get("caption"),
                    scope.get("text"),
                    " ".join(scope.get("scene_tags") or []),
                    scope.get("transcript"),
                ]
                if value
            )
            lowered = text.lower()
            score = overlap_score(query_tokens, text)
            score += 0.65 * len(CAPTION_FALLBACK_TERMS & set(lowered.replace("-", " ").split()))
            if any(phrase in lowered for phrase in CAPTION_FALLBACK_PHRASES):
                score += 1.5
            if str(scope.get("segment_id")) in anchor_segments:
                score += 1.25
            if score > 0.0:
                scored.append((score, scope))
        scored.sort(key=lambda row: row[0], reverse=True)

        items: list[dict[str, Any]] = []
        for score, scope in scored[:target_count]:
            segment_id = scope.get("segment_id")
            start = scope.get("start_time")
            end = scope.get("end_time")
            time_text = self._format_time_text(start, end)
            time_suffix = f" ({time_text})" if time_text else ""
            caption = scope.get("caption") or scope.get("text") or ""
            text = (
                f"Caption-supported context: Segment {segment_id or 'unknown'}"
                f"{time_suffix} describes {shorten(caption, 700)}"
            )
            items.append(
                {
                    "view": "caption_context",
                    "id": f"caption_context:{segment_id or len(items)}",
                    "node_id": scope.get("node_id"),
                    "segment_id": segment_id,
                    "segment_ids": [segment_id] if segment_id else [],
                    "start_time": start,
                    "end_time": end,
                    "score": float(score) + 5.0,
                    "role": "caption_supported_context",
                    "text": shorten(text, 950),
                    "short_text": shorten(text, 260),
                    "provenance": scope.get("provenance") or {},
                    "record": scope,
                    **self._video_fields_for_sources(scope),
                }
            )
        return items

    def _augment_temporal_context(
        self,
        query: str,
        plan: Any,
        evidence: list[dict[str, Any]],
        fused_evidence: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        if not self._has_temporal_sequence_intent(plan):
            return [], "not_temporal_intent"

        anchor_candidates = self._dedupe_evidence_items(list(evidence) + list(fused_evidence))
        anchor = self._select_temporal_anchor(anchor_candidates, plan)
        if anchor is None:
            return [], "temporal_intent_no_anchor_with_time_or_segment"

        sources: list[tuple[str, list[dict[str, Any]], str]] = [
            ("visual_event", self.visual_events, "temporal_intent_selected_visual_event"),
            ("event", self.events, "temporal_intent_selected_event"),
            ("caption_segment", self.scopes, "temporal_intent_selected_caption_segment"),
        ]
        candidates: list[tuple[float, int, list[dict[str, Any]], str, str]] = []
        visual_event_rejected_long = False

        for source_view, rows, reason in sources:
            if source_view == "visual_event":
                visual_event_rejected_long = (
                    visual_event_rejected_long
                    or self._temporal_source_has_rejected_long_rows(rows, source_view, anchor)
                )
            items = self._build_temporal_sequence_evidence(
                anchor,
                source_view,
                rows,
                max_items=TEMPORAL_SEQUENCE_MAX_ITEMS,
            )
            if not items:
                continue
            quality = self._temporal_sequence_quality(items, source_view)
            candidates.append((quality, -len(candidates), items, reason, source_view))

        if not candidates:
            if visual_event_rejected_long:
                return [], "temporal_intent_visual_event_rejected_long_duration"
            return [], "temporal_intent_no_local_temporal_rows"

        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best_quality, _, best_items, best_reason, best_source = candidates[0]
        if not self._has_temporal_neighbor(best_items):
            return best_items, "temporal_intent_no_adjacent_temporal_rows"
        if best_source == "caption_segment" and visual_event_rejected_long:
            return best_items, "temporal_intent_selected_caption_segment_due_to_better_granularity"
        return best_items, best_reason

    def _has_temporal_sequence_intent(self, plan: Any) -> bool:
        intents = getattr(plan, "query_intents", None) or {}
        if not isinstance(intents, dict) and isinstance(plan, dict):
            intents = plan.get("query_intents") or {}
        return any(bool(intents.get(key)) for key in TEMPORAL_SEQUENCE_INTENT_KEYS)

    def _select_temporal_anchor(self, evidence: list[dict[str, Any]], plan: Any) -> dict[str, Any] | None:
        scored: list[tuple[float, int, dict[str, Any]]] = []
        fallback_scored: list[tuple[float, int, dict[str, Any]]] = []
        type_priority = {
            "event": 9.0,
            "segment": 8.5,
            "caption_context": 8.5,
            "visual_event": 7.5,
            "visual_track": 5.5,
            "track": 5.0,
            "target": 4.0,
            "visual_object": 3.0,
            "scope": 3.0,
        }
        for order, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            if self._temporal_start(item) is None and not self._temporal_segment_ids(item):
                continue
            is_long = self._is_temporal_sequence_global_row(item)
            score = type_priority.get(self._evidence_type(item), 1.0)
            try:
                score += min(2.0, max(0.0, _safe_float(item.get("score"), 0.0)))
            except (TypeError, ValueError):
                pass
            score += 0.25 * self._intent_evidence_priority(plan, item)
            duration = self._temporal_duration(item)
            if duration is not None:
                score += max(0.0, 2.0 - min(duration, TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS) / 60.0)
            else:
                score += 0.8
            if self._temporal_start(item) is not None:
                score += 0.8
            if self._temporal_segment_ids(item):
                score += 0.5
            if is_long:
                fallback_scored.append((score - 8.0, -order, item))
            else:
                scored.append((score, -order, item))
        if not scored and fallback_scored:
            scored = fallback_scored
        if not scored:
            return None
        scored.sort(key=lambda row: row[:2], reverse=True)
        return scored[0][2]

    def _build_temporal_sequence_evidence(
        self,
        anchor: dict[str, Any],
        source_view: str,
        rows: list[dict[str, Any]],
        *,
        max_items: int,
    ) -> list[dict[str, Any]]:
        if max_items <= 0 or not rows:
            return []
        anchor_video = self._temporal_video_name(anchor)
        prepared: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            row_video = self._temporal_video_name(row)
            if anchor_video and row_video and row_video != anchor_video:
                continue
            if self._temporal_start(row) is None and not self._temporal_segment_ids(row):
                continue
            if not self._is_temporal_sequence_row_usable(row, source_view):
                continue
            prepared.append((self._temporal_sort_key(row, index), index, row))
        if not prepared:
            return []
        prepared.sort(key=lambda row: row[0])

        focal_index = min(
            range(len(prepared)),
            key=lambda idx: self._temporal_anchor_distance(anchor, prepared[idx][2]),
        )
        neighbor_count = max(1, min(2, (max_items - 1) // 2))
        before_indices = list(range(max(0, focal_index - neighbor_count), focal_index))
        after_indices = list(range(focal_index + 1, min(len(prepared), focal_index + 1 + neighbor_count)))
        selected = (
            [(idx, "BEFORE") for idx in before_indices]
            + [(focal_index, "FOCAL EVENT")]
            + [(idx, "AFTER") for idx in after_indices]
        )

        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for idx, role in selected[:max_items]:
            row = prepared[idx][2]
            row_key = (
                source_view,
                str(row.get("id") or row.get("event_id") or row.get("node_id") or self._compact_evidence_preview(row)),
            )
            if row_key in seen:
                continue
            seen.add(row_key)
            items.append(self._temporal_sequence_item(source_view, row, role))
        return items

    def _has_temporal_neighbor(self, items: list[dict[str, Any]]) -> bool:
        roles = {item.get("temporal_role") for item in items}
        return {"BEFORE", "FOCAL EVENT", "AFTER"}.issubset(roles)

    def _temporal_source_has_rejected_long_rows(
        self,
        rows: list[dict[str, Any]],
        source_view: str,
        anchor: dict[str, Any],
    ) -> bool:
        anchor_video = self._temporal_video_name(anchor)
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_video = self._temporal_video_name(row)
            if anchor_video and row_video and row_video != anchor_video:
                continue
            if self._temporal_start(row) is None and not self._temporal_segment_ids(row):
                continue
            if self._is_temporal_sequence_row_usable(row, source_view):
                continue
            duration = self._temporal_duration(row)
            segment_span = self._temporal_segment_span(row)
            if duration is not None and duration > TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS:
                return True
            if segment_span is not None and segment_span > TEMPORAL_SEQUENCE_MAX_SEGMENT_SPAN:
                return True
        return False

    def _is_temporal_sequence_row_usable(self, row: dict[str, Any], source_view: str) -> bool:
        duration = self._temporal_duration(row)
        if duration is not None and duration > TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS:
            return False

        segment_span = self._temporal_segment_span(row)
        if segment_span is not None and segment_span > TEMPORAL_SEQUENCE_MAX_SEGMENT_SPAN:
            return False

        if source_view in {"visual_event", "visual_track", "track"} and self._is_temporal_sequence_global_row(row):
            return False
        return True

    def _temporal_sequence_quality(self, items: list[dict[str, Any]], source_view: str) -> float:
        roles = {item.get("temporal_role") for item in items}
        quality = 0.0
        if "BEFORE" in roles:
            quality += 3.0
        if "FOCAL EVENT" in roles:
            quality += 3.0
        if "AFTER" in roles:
            quality += 3.0
        if self._has_temporal_neighbor(items):
            quality += 4.0

        durations = [
            duration
            for duration in (self._temporal_duration(item) for item in items)
            if duration is not None
        ]
        if durations:
            average_duration = sum(durations) / max(len(durations), 1)
            if average_duration <= 45.0:
                quality += 3.0
            elif average_duration <= TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS:
                quality += 1.5
            else:
                quality -= 8.0
        else:
            quality += 0.5

        segment_indices = [
            index
            for item in items
            for index in self._temporal_segment_index_values(item)
        ]
        if segment_indices:
            unique_indices = sorted(set(segment_indices))
            if len(unique_indices) >= 2:
                quality += 2.0
            if len(unique_indices) >= 3:
                quality += 1.0
            if unique_indices[-1] - unique_indices[0] <= max(TEMPORAL_SEQUENCE_MAX_SEGMENT_SPAN, len(unique_indices)):
                quality += 1.0

        starts = [self._temporal_start(item) for item in items if self._temporal_start(item) is not None]
        unique_starts = set(starts)
        if len(unique_starts) >= 2:
            quality += 2.0
        if len(unique_starts) >= 3:
            quality += 1.0

        if any(self._is_temporal_sequence_global_row(item) for item in items):
            quality -= 10.0
        if source_view == "caption_segment":
            quality += 1.2
        elif source_view == "event":
            quality += 0.8
        elif source_view == "visual_event":
            quality += 0.2
        return quality

    def _temporal_sequence_item(self, source_view: str, row: dict[str, Any], role: str) -> dict[str, Any]:
        timestamp = self._row_timestamp(row)
        start = self._temporal_raw_time(row, ("start_time",))
        end = self._temporal_raw_time(row, ("end_time",))
        if start is None and source_view in {"event", "caption_segment"}:
            start = self._temporal_start(row)
        segment_ids = self._temporal_segment_ids(row)
        label = self._temporal_label(row)
        action = self._temporal_action(row)
        support = self._temporal_support_text(row)
        confidence = self._temporal_raw_time(row, ("confidence",))
        score = row.get("score")
        if score is None:
            score = row.get("confidence")
        location_parts = []
        time_text = self._format_time_text(start, end, timestamp)
        if time_text:
            location_parts.append(f"time={time_text}")
        if segment_ids:
            location_parts.append(f"segment={','.join(segment_ids)}")
        location = "; ".join(location_parts) or "time/segment unavailable"
        details = [
            f"source={source_view}",
            location,
            f"label/category={label or 'unknown'}",
            f"action/event={action or 'unknown'}",
        ]
        if confidence is not None:
            details.append(f"confidence={confidence:.3f}")
        if score is not None:
            try:
                score_value = float(score)
                details.append(f"score={score_value:.3f}")
            except (TypeError, ValueError):
                score_value = None
        else:
            score_value = None
        text = (
            f"Temporal sequence evidence [{role}]: "
            f"{'; '.join(details)}. Support: {shorten(support, 420)}"
        )
        row_id = row.get("id") or row.get("event_id") or row.get("node_id") or len(text)
        return {
            "view": "temporal_sequence",
            "id": f"temporal_sequence:{source_view}:{role.lower().replace(' ', '_')}:{row_id}",
            "source_view": source_view,
            "source": source_view,
            "source_id": row_id,
            "temporal_role": role,
            "timestamp": timestamp,
            "start_time": start,
            "end_time": end,
            "segment_id": segment_ids[0] if segment_ids else None,
            "segment_ids": segment_ids,
            "label": label,
            "category": label,
            "action": action,
            "event_type": row.get("event_type") or row.get("relation_type"),
            "confidence": confidence,
            "score": score_value,
            "support_text": shorten(support, 700),
            "text": shorten(text, 780),
            "short_text": shorten(text, 260),
            "provenance": row.get("provenance") or {},
            **self._video_fields_for_sources(row),
        }

    def _temporal_anchor_distance(self, anchor: dict[str, Any], row: dict[str, Any]) -> tuple[float, float]:
        anchor_ids = self._temporal_id_values(anchor)
        row_ids = self._temporal_id_values(row)
        if anchor_ids and row_ids and anchor_ids & row_ids:
            return (0.0, 0.0)

        anchor_time = self._temporal_start(anchor)
        row_start = self._temporal_start(row)
        row_end = self._temporal_end(row)
        if anchor_time is not None and row_start is not None and row_end is not None:
            if row_start <= anchor_time <= row_end:
                return (1.0, 0.0)
            return (2.0, min(abs(row_start - anchor_time), abs(row_end - anchor_time)))
        if anchor_time is not None and row_start is not None:
            return (2.5, abs(row_start - anchor_time))

        segment_overlap = set(self._temporal_segment_ids(anchor)) & set(self._temporal_segment_ids(row))
        if segment_overlap:
            return (1.5, 0.0)
        segment_distance = self._temporal_segment_distance(
            self._temporal_segment_ids(anchor),
            self._temporal_segment_ids(row),
        )
        if segment_distance is not None:
            return (3.0, float(segment_distance))
        return (9.0, 999999.0)

    def _temporal_sort_key(self, row: dict[str, Any], fallback_index: int) -> tuple[Any, ...]:
        video_name = self._temporal_video_name(row) or ""
        start = self._temporal_start(row)
        if start is not None:
            return (video_name, 0, float(start), "", 0, fallback_index)
        segment_ids = self._temporal_segment_ids(row)
        if segment_ids:
            prefix, index = self._segment_index(segment_ids[0])
            return (video_name, 1, float("inf"), prefix, index if index is not None else fallback_index, fallback_index)
        return (video_name, 2, float("inf"), "", fallback_index, fallback_index)

    def _temporal_raw_time(self, item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        sources = [item]
        record = item.get("record")
        if isinstance(record, dict):
            sources.append(record)
        for source in sources:
            for key in keys:
                value = source.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            provenance = source.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in keys:
                    value = provenance.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            pass
        return None

    def _temporal_start(self, item: dict[str, Any]) -> float | None:
        return self._row_timestamp(item)

    def _temporal_end(self, item: dict[str, Any]) -> float | None:
        return self._temporal_raw_time(item, ("end_time",))

    def _temporal_duration(self, item: dict[str, Any]) -> float | None:
        start = self._temporal_raw_time(item, ("start_time",))
        end = self._temporal_raw_time(item, ("end_time",))
        if start is None:
            start = self._row_timestamp(item)
        if start is None or end is None:
            return None
        duration = end - start
        return duration if duration >= 0 else None

    def _temporal_segment_span(self, item: dict[str, Any]) -> int | None:
        indices = self._temporal_segment_index_values(item)
        if len(indices) < 2:
            return None
        return max(indices) - min(indices)

    def _temporal_segment_index_values(self, item: dict[str, Any]) -> list[int]:
        values: list[int] = []
        for segment_id in self._temporal_segment_ids(item):
            _, index = self._segment_index(segment_id)
            if index is not None and index not in values:
                values.append(index)

        def add_index(raw: Any) -> None:
            if raw is None:
                return
            try:
                index = int(raw)
            except (TypeError, ValueError):
                return
            if index not in values:
                values.append(index)

        for source in [item, item.get("record") or {}, item.get("provenance") or {}]:
            if not isinstance(source, dict):
                continue
            add_index(source.get("segment_index"))
        return values

    def _is_temporal_sequence_global_row(self, item: dict[str, Any]) -> bool:
        duration = self._temporal_duration(item)
        if duration is not None and duration > TEMPORAL_SEQUENCE_MAX_DURATION_SECONDS:
            return True
        segment_span = self._temporal_segment_span(item)
        return bool(segment_span is not None and segment_span > TEMPORAL_SEQUENCE_MAX_SEGMENT_SPAN)

    def _temporal_segment_ids(self, item: dict[str, Any]) -> list[str]:
        segment_ids: list[str] = []

        def add(raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, str):
                values = [raw]
            elif isinstance(raw, (list, tuple, set)):
                values = [str(value) for value in raw if value]
            else:
                values = [str(raw)]
            for value in values:
                if value and value not in segment_ids:
                    segment_ids.append(value)

        add(item.get("segment_id"))
        add(item.get("segment_ids"))
        add(item.get("related_segment_ids"))
        record = item.get("record") or {}
        if isinstance(record, dict):
            add(record.get("segment_id"))
            add(record.get("segment_ids"))
            add(record.get("related_segment_ids"))
        provenance = item.get("provenance") or {}
        if isinstance(provenance, dict):
            add(provenance.get("segment_id"))
        return segment_ids

    def _temporal_segment_distance(self, left: list[str], right: list[str]) -> int | None:
        best: int | None = None
        for left_id in left:
            left_prefix, left_index = self._segment_index(left_id)
            if left_index is None:
                continue
            for right_id in right:
                right_prefix, right_index = self._segment_index(right_id)
                if right_index is None or left_prefix != right_prefix:
                    continue
                distance = abs(left_index - right_index)
                best = distance if best is None else min(best, distance)
        return best

    def _temporal_video_name(self, item: dict[str, Any]) -> str:
        for source in [item, item.get("record") or {}, item.get("provenance") or {}]:
            if not isinstance(source, dict):
                continue
            for key in ("video_id", "source_vid", "video_name"):
                if source.get(key):
                    return str(source.get(key))
        return ""

    def _temporal_id_values(self, item: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        keys = (
            "id",
            "node_id",
            "event_id",
            "track_id",
            "object_id",
            "source_id",
            "actor_track_id",
            "actor_object_id",
        )
        for source in [item, item.get("record") or {}]:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if value:
                    values.add(str(value))
        return values

    def _temporal_label(self, row: dict[str, Any]) -> str:
        for source in [row, row.get("record") or {}]:
            if not isinstance(source, dict):
                continue
            for key in ("label", "category", "class_name", "object_label"):
                value = source.get(key)
                if value:
                    return str(value).replace("_", " ")
            state = source.get("state_signature")
            if isinstance(state, dict):
                counts = state.get("object_counts") or {}
                if isinstance(counts, dict) and counts:
                    return ", ".join(f"{label}={count}" for label, count in list(counts.items())[:4])
        return ""

    def _temporal_action(self, row: dict[str, Any]) -> str:
        for source in [row, row.get("record") or {}]:
            if not isinstance(source, dict):
                continue
            for key in ("event_type", "relation_type", "action", "direction_text"):
                value = source.get(key)
                if value:
                    return str(value).replace("_", " ")
            action_tags = source.get("action_tags")
            if action_tags:
                return ", ".join(str(tag) for tag in action_tags[:6])
        return ""

    def _temporal_support_text(self, row: dict[str, Any]) -> str:
        parts: list[str] = []
        for source in [row, row.get("record") or {}]:
            if not isinstance(source, dict):
                continue
            for key in ("evidence_text", "summary", "motion_summary", "caption", "text", "transcript"):
                value = source.get(key)
                if value:
                    parts.append(str(value))
            scene_tags = source.get("scene_tags")
            if scene_tags:
                parts.append("Scene tags: " + ", ".join(str(tag) for tag in scene_tags[:8]))
            state = source.get("state_signature")
            if state:
                parts.append(f"State signature: {state}")
        return shorten(" ".join(parts), 900)

    def _best_temporal_relation_for_event(self, event: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        event_time = self._row_timestamp(event)
        event_frame = event.get("frame_id")
        event_segment = event.get("segment_id")
        event_track_ids = self._track_id_set(event)
        scored: list[tuple[int, float, float, dict[str, Any], str]] = []
        for relation in self.visual_relations:
            if not self._rows_same_video(event, relation):
                continue
            relation_time = self._row_timestamp(relation)
            relation_frame = relation.get("frame_id")
            relation_segment = relation.get("segment_id")
            relation_track_ids = self._track_id_set(relation)
            same_track = bool(event_track_ids & relation_track_ids)
            same_keyframe = bool(event_frame is not None and relation_frame is not None and str(event_frame) == str(relation_frame))
            same_segment = bool(event_segment and relation_segment and str(event_segment) == str(relation_segment))
            time_gap = self._time_gap(event_time, relation_time)

            if same_keyframe:
                status = "aligned"
                bucket = 3
            elif same_track and time_gap is not None and time_gap <= TEMPORAL_RELATION_STRICT_WINDOW_SECONDS:
                status = "aligned"
                bucket = 3
            elif same_track and time_gap is not None and time_gap <= TEMPORAL_RELATION_FALLBACK_WINDOW_SECONDS:
                status = "near_time_fallback"
                bucket = 2
            elif same_segment:
                status = "same_segment_fallback"
                bucket = 1
            else:
                continue

            relation_type = str(relation.get("relation_type") or "").lower()
            relation_bonus = 0.3 if relation_type in {"nearest_to", "overlap_or_near"} else 0.1
            gap_score = -(time_gap if time_gap is not None else 9999.0)
            scored.append((bucket, relation_bonus, gap_score, relation, status))

        if not scored:
            return None, "same_time_relation_unavailable"
        scored.sort(key=lambda row: row[:3], reverse=True)
        return scored[0][3], scored[0][4]

    def _temporal_limitation_item(self, event: dict[str, Any]) -> dict[str, Any]:
        timestamp = self._row_timestamp(event)
        segment_id = event.get("segment_id")
        text = (
            f"Temporal limitation: same-time relation unavailable for the visual event"
            f"{self._time_phrase(timestamp)}. Falling back to same-segment caption/scope context"
            f"{f' for segment {segment_id}' if segment_id else ''}."
        )
        return {
            "view": "temporal_limitation",
            "id": f"temporal_limitation:{event.get('event_id') or event.get('id') or len(text)}",
            "node_id": event.get("node_id"),
            "segment_id": segment_id,
            "score": 0.8,
            "role": "uncertainty_limitations",
            "text": text,
            "short_text": shorten(text, 260),
            "provenance": event.get("provenance") or {},
            **self._video_fields_for_sources(event),
        }

    def _nearby_object_context_for_event(
        self,
        event: dict[str, Any],
        *,
        video_filter: set[str] | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        event_time = self._row_timestamp(event)
        event_frame = event.get("frame_id")
        event_segment = event.get("segment_id")
        event_tracks = self._track_id_set(event)
        event_object_ids = self._object_id_set(event)
        event_id = event.get("event_id") or event.get("id") or self._compact_evidence_preview(event)

        def is_event_object(row: dict[str, Any]) -> bool:
            row_object_ids = self._object_id_set(row)
            if event_object_ids and row_object_ids and event_object_ids & row_object_ids:
                return True
            row_tracks = self._track_id_set(row)
            return bool(event_tracks and row_tracks and event_tracks & row_tracks)

        def add_candidate(
            target: dict[str, Any] | None,
            candidates: dict[str, dict[str, Any]],
            *,
            relation: dict[str, Any] | None = None,
        ) -> None:
            if not target or is_event_object(target):
                return
            if not self._rows_same_video(event, target):
                return
            key = str(target.get("object_id") or target.get("id") or self._compact_evidence_preview(target))
            if key not in candidates:
                candidates[key] = target
            if relation:
                candidates[key] = {**candidates[key], "nearby_relation_type": relation.get("relation_type")}

        strict_candidates: dict[str, dict[str, Any]] = {}
        same_segment_candidates: dict[str, dict[str, Any]] = {}
        for relation in self.visual_relations:
            if not self._row_matches_video_filter(relation, video_filter) or not self._rows_same_video(event, relation):
                continue
            relation_time = self._row_timestamp(relation)
            relation_frame = relation.get("frame_id")
            relation_segment = relation.get("segment_id")
            relation_tracks = self._track_id_set(relation)
            relation_objects = self._object_id_set(relation)
            anchored = bool(event_tracks & relation_tracks or event_object_ids & relation_objects)
            if not anchored:
                continue

            same_keyframe = bool(event_frame is not None and relation_frame is not None and str(event_frame) == str(relation_frame))
            time_gap = self._time_gap(event_time, relation_time)
            same_time = same_keyframe or (time_gap is not None and time_gap <= TEMPORAL_RELATION_STRICT_WINDOW_SECONDS)
            same_segment = bool(event_segment and relation_segment and str(event_segment) == str(relation_segment))
            related_id = relation.get("related_object_id") or relation.get("neighbor_object_id")
            target_id = relation.get("target_object_id")
            if related_id and (not event_object_ids or str(related_id) not in event_object_ids):
                candidate = self.visual_objects_by_id.get(related_id) or self.visual_objects_by_id.get(str(related_id))
            else:
                candidate = self.visual_objects_by_id.get(target_id) or self.visual_objects_by_id.get(str(target_id))
            if same_time:
                add_candidate(candidate, strict_candidates, relation=relation)
            elif same_segment:
                add_candidate(candidate, same_segment_candidates, relation=relation)

        if not strict_candidates:
            for row in self.visual_objects:
                if not self._row_matches_video_filter(row, video_filter) or not self._rows_same_video(event, row):
                    continue
                if is_event_object(row):
                    continue
                row_time = self._row_timestamp(row)
                row_frame = row.get("frame_id")
                row_segment = row.get("segment_id")
                same_keyframe = bool(event_frame is not None and row_frame is not None and str(event_frame) == str(row_frame))
                time_gap = self._time_gap(event_time, row_time)
                same_time = same_keyframe or (time_gap is not None and time_gap <= TEMPORAL_RELATION_STRICT_WINDOW_SECONDS)
                same_segment = bool(event_segment and row_segment and str(event_segment) == str(row_segment))
                if same_time:
                    add_candidate(row, strict_candidates)
                elif same_segment:
                    add_candidate(row, same_segment_candidates)

        if strict_candidates:
            candidate_count = len(strict_candidates)
            label_text = self._label_count_text(strict_candidates.values())
            text = (
                f"Nearby-object context: {self._time_sentence_start(event_time)}"
                f"the event object has nearby same-time detected objects: {label_text}."
            )
            return (
                {
                    "view": "nearby_object_context",
                    "id": f"nearby_object_context:{event_id}",
                    "node_id": event.get("node_id"),
                    "segment_id": event_segment,
                    "timestamp": event_time,
                    "score": 4.6,
                    "role": "spatial_nearby_context",
                    "text": shorten(text, 950),
                    "short_text": shorten(text, 260),
                    "candidate_count": candidate_count,
                    "temporal_alignment": "same_time",
                    "provenance": event.get("provenance") or {},
                    **self._video_fields_for_sources(event),
                },
                candidate_count,
            )

        if same_segment_candidates:
            candidate_count = len(same_segment_candidates)
            label_text = self._label_count_text(same_segment_candidates.values())
            text = (
                f"Spatial limitation: same-time nearby object unavailable for the visual event"
                f"{self._time_phrase(event_time)}. Same-segment object context includes: {label_text}."
            )
            return (
                {
                    "view": "nearby_object_context",
                    "id": f"nearby_object_context:{event_id}:same_segment",
                    "node_id": event.get("node_id"),
                    "segment_id": event_segment,
                    "timestamp": event_time,
                    "score": 2.2,
                    "role": "uncertainty_limitations",
                    "text": shorten(text, 950),
                    "short_text": shorten(text, 260),
                    "candidate_count": candidate_count,
                    "temporal_alignment": "same_segment_fallback",
                    "provenance": event.get("provenance") or {},
                    **self._video_fields_for_sources(event),
                },
                candidate_count,
            )

        text = (
            f"Spatial limitation: same-time nearby object unavailable for the visual event"
            f"{self._time_phrase(event_time)}."
        )
        return (
            {
                "view": "nearby_object_context",
                "id": f"nearby_object_context:{event_id}:unavailable",
                "node_id": event.get("node_id"),
                "segment_id": event_segment,
                "timestamp": event_time,
                "score": 1.0,
                "role": "uncertainty_limitations",
                "text": text,
                "short_text": shorten(text, 260),
                "candidate_count": 0,
                "temporal_alignment": "unavailable",
                "provenance": event.get("provenance") or {},
                **self._video_fields_for_sources(event),
            },
            0,
        )

    def _naturalize_evidence_item(self, item: dict[str, Any]) -> dict[str, Any]:
        view = self._evidence_type(item)
        naturalized = dict(item)
        if view == "visual_event":
            timestamp = self._row_timestamp(item)
            record = item.get("record") or {}
            label = str(item.get("label") or (record.get("label") if isinstance(record, dict) else "") or "object").replace("_", " ")
            event_type = str(
                item.get("event_type")
                or item.get("relation_type")
                or (record.get("event_type") if isinstance(record, dict) else "")
                or (record.get("relation_type") if isinstance(record, dict) else "")
                or ""
            ).replace("_", " ")
            naturalized["text"] = shorten(
                f"Direct visual evidence: {self._time_sentence_start(timestamp)}a {label} track {self._event_phrase(event_type)}.",
                950,
            )
        elif view == "visual_relation":
            timestamp = self._row_timestamp(item)
            relation_type = str(item.get("relation_type") or "spatial relation").replace("_", " ")
            alignment = item.get("temporal_alignment")
            if alignment == "aligned":
                prefix = "Direct visual evidence"
                suffix = "This relation is aligned to the visual event timestamp."
            elif alignment == "near_time_fallback":
                prefix = "Temporal fallback"
                suffix = "The same-time relation was unavailable; this relation is within the wider time window."
            elif alignment == "same_segment_fallback":
                prefix = "Temporal fallback"
                suffix = "Same-time relation unavailable; this is same-segment relation context only."
            else:
                prefix = "Direct visual evidence"
                suffix = "Use this only as local spatial context."
            naturalized["text"] = shorten(
                f"{prefix}: {self._time_sentence_start(timestamp)}a neighboring object is {relation_type} the event object. {suffix}",
                950,
            )
        elif view == "visual_track":
            start = item.get("start_time")
            end = item.get("end_time")
            record = item.get("record") or {}
            if isinstance(record, dict):
                start = start if start is not None else record.get("start_time")
                end = end if end is not None else record.get("end_time")
            label = str(item.get("label") or "object").replace("_", " ")
            summary = item.get("motion_summary") or (record.get("motion_summary") if isinstance(record, dict) else "") or item.get("text") or "has visible motion"
            time_text = self._format_time_text(start, end)
            time_prefix = f"From {time_text}, " if time_text else ""
            naturalized["text"] = shorten(
                f"Direct visual evidence: {time_prefix}a {label} track {shorten(str(summary), 420)}.",
                950,
            )
        elif view == "visual_object":
            timestamp = self._row_timestamp(item)
            label = str(item.get("label") or "object").replace("_", " ")
            naturalized["text"] = shorten(
                f"Direct visual evidence: {self._time_sentence_start(timestamp)}a {label} is detected in the frame.",
                950,
            )
        elif view == "segment":
            segment_id = item.get("segment_id") or (item.get("segment_ids") or [""])[0]
            naturalized["text"] = shorten(
                f"Caption-supported context: Segment {segment_id or 'unknown'} describes {item.get('text') or item.get('short_text') or ''}",
                950,
            )
        return naturalized

    def _add_track_values(self, values: set[str], raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, str):
            if raw:
                values.add(raw)
            return
        if isinstance(raw, (list, tuple, set)):
            for value in raw:
                if value:
                    values.add(str(value))
            return
        values.add(str(raw))

    def _object_id_set(self, item: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for key in ("object_id", "target_object_id", "related_object_id", "neighbor_object_id", "actor_object_id"):
            value = item.get(key)
            if value:
                values.add(str(value))
        record = item.get("record") or {}
        if isinstance(record, dict):
            for key in ("object_id", "target_object_id", "related_object_id", "neighbor_object_id", "actor_object_id"):
                value = record.get(key)
                if value:
                    values.add(str(value))
        return values

    def _label_count_text(self, rows: Any) -> str:
        counts = Counter(str(row.get("label") or "object").replace("_", " ") for row in rows)
        if not counts:
            return "none"
        return ", ".join(f"{label}={count}" for label, count in counts.most_common(8))

    def _track_id_set(self, item: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for key in ("track_id", "target_track_id", "related_track_id", "neighbor_track_id", "actor_track_id"):
            self._add_track_values(values, item.get(key))
        for key in ("track_ids", "involved_track_ids", "related_track_ids"):
            self._add_track_values(values, item.get(key))
        record = item.get("record") or {}
        if isinstance(record, dict):
            for key in ("track_id", "target_track_id", "related_track_id", "neighbor_track_id", "actor_track_id"):
                self._add_track_values(values, record.get(key))
            for key in ("track_ids", "involved_track_ids", "related_track_ids"):
                self._add_track_values(values, record.get(key))
        return values

    def _row_timestamp(self, item: dict[str, Any]) -> float | None:
        sources = [item]
        record = item.get("record")
        if isinstance(record, dict):
            sources.append(record)
        for source in sources:
            for key in ("timestamp", "time", "frame_timestamp", "start_time"):
                value = source.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            provenance = source.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in ("timestamp", "start_time"):
                    value = provenance.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            pass
        return None

    def _time_gap(self, left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return abs(left - right)

    def _format_time_text(
        self,
        start: float | None,
        end: float | None = None,
        timestamp: float | None = None,
    ) -> str:
        if start is not None and end is not None:
            return f"{format_seconds(start)}-{format_seconds(end)}"
        if start is not None:
            return format_seconds(start)
        if end is not None:
            return format_seconds(end)
        if timestamp is not None:
            return format_seconds(timestamp)
        return ""

    def _time_phrase(self, timestamp: float | None) -> str:
        return f" at {timestamp:.1f}s" if timestamp is not None else ""

    def _time_sentence_start(self, timestamp: float | None) -> str:
        return f"At {timestamp:.1f}s, " if timestamp is not None else ""

    def _event_phrase(self, event_type: str) -> str:
        normalized = event_type.strip().lower()
        if normalized.startswith("enter "):
            return f"enters {normalized[len('enter '):]}"
        if normalized.startswith("move "):
            return f"moves {normalized[len('move '):]}"
        if normalized:
            return normalized
        return "has a visible event"

    def _dedupe_evidence_items(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in evidence:
            key = (
                self._evidence_type(item),
                str(item.get("id") or item.get("node_id") or self._compact_evidence_preview(item)),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _ensure_nearby_context_in_final(
        self,
        evidence,
        fused_evidence,
        *,
        max_items: int,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        if any(self._evidence_type(item) == "nearby_object_context" for item in evidence):
            return evidence

        nearby_item = next((item for item in fused_evidence if self._evidence_type(item) == "nearby_object_context"), None)
        if nearby_item is None:
            return evidence

        result = list(evidence)
        insert_at = 0
        for index, item in enumerate(result):
            if self._evidence_type(item) in {"visual_event", "visual_relation"}:
                insert_at = index + 1
        result.insert(insert_at, nearby_item)

        protected_caption_seen = False

        def removable_priority(item: dict[str, Any]) -> int:
            nonlocal protected_caption_seen
            view = self._evidence_type(item)
            if item is nearby_item or view in {"nearby_object_context", "visual_event", "fixed_window_event"}:
                return 100
            if view == "caption_context":
                if not protected_caption_seen:
                    protected_caption_seen = True
                    return 100
                return 4
            if view == "visual_track":
                return 0
            if view == "visual_object":
                return 1
            if view == "visual_relation":
                return 2
            if view in {"segment", "scope", "target", "track", "event", "fixed_window_event"}:
                return 5
            if view == "temporal_limitation":
                return 6
            return 5

        def total_chars(rows: list[dict[str, Any]]) -> int:
            return sum(len(str(row.get("text") or row.get("short_text") or "")) for row in rows)

        while len(result) > max_items or total_chars(result) > max_chars:
            priorities = [(removable_priority(item), index) for index, item in enumerate(result)]
            removable = [(priority, index) for priority, index in priorities if priority < 100]
            if not removable:
                break
            _, remove_index = min(removable, key=lambda row: (row[0], -row[1]))
            result.pop(remove_index)
            protected_caption_seen = False

        if total_chars(result) > max_chars:
            result = self._trim_evidence(result, max_items=max_items, max_chars=max_chars)
            if not any(self._evidence_type(item) == "nearby_object_context" for item in result):
                result = [nearby_item] + [item for item in result if self._evidence_type(item) != "nearby_object_context"]
                result = self._trim_evidence(result, max_items=max_items, max_chars=max_chars)
        return result[:max_items]

    def _evidence_type(self, evidence_item: Any) -> str:
        if isinstance(evidence_item, dict):
            view = evidence_item.get("view") or evidence_item.get("type") or ""
            if view:
                return str(view)
            record = evidence_item.get("record") or {}
            if isinstance(record, dict):
                return str(record.get("view") or record.get("type") or "unknown")
        return "text" if isinstance(evidence_item, str) else "unknown"

    def _evidence_text(self, evidence_item: Any) -> str:
        if isinstance(evidence_item, str):
            return evidence_item
        if not isinstance(evidence_item, dict):
            return str(evidence_item or "")
        parts: list[str] = []
        for key in ("text", "short_text", "evidence_text", "summary", "motion_summary", "relation_type", "event_type"):
            value = evidence_item.get(key)
            if value:
                parts.append(str(value))
        record = evidence_item.get("record") or {}
        if isinstance(record, dict):
            for key in ("text", "evidence_text", "summary", "motion_summary", "relation_type", "event_type", "label", "color"):
                value = record.get(key)
                if value:
                    parts.append(str(value))
        return " ".join(parts)

    def _intent_evidence_priority(self, plan: Any, evidence_item: Any) -> float:
        intents = getattr(plan, "query_intents", None) or {}
        if not isinstance(intents, dict) and isinstance(plan, dict):
            intents = plan.get("query_intents") or {}
        evidence_type = self._evidence_type(evidence_item)
        text = self._evidence_text(evidence_item).lower()
        score = 0.0
        if isinstance(evidence_item, dict):
            try:
                score += min(2.0, max(0.0, float(evidence_item.get("score", 0.0))))
            except (TypeError, ValueError):
                pass

        if intents.get("spatial_relation"):
            score += {
                "visual_relation": 4.0,
                "visual_object": 2.0,
                "visual_track": 1.4,
                "visual_event": 1.0,
                "fixed_window_event": 1.0,
                "adaptive_event": 0.9,
                "relation": 1.4,
                "target": 0.5,
                "scope": 0.3,
                "segment": 0.3,
            }.get(evidence_type, 0.0)
            score += 0.15 * len(SPATIAL_EVIDENCE_TERMS & set(text.split()))
        if intents.get("temporal_trajectory"):
            score += {
                "visual_track": 4.0,
                "visual_event": 2.4,
                "fixed_window_event": 2.4,
                "adaptive_event": 3.0,
                "visual_relation": 1.3,
                "visual_object": 0.7,
                "track": 1.6,
                "event": 1.0,
            }.get(evidence_type, 0.0)
            score += 0.15 * len(TEMPORAL_EVIDENCE_TERMS & set(text.split()))
        if any(intents.get(key) for key in TEMPORAL_SEQUENCE_INTENT_KEYS):
            score += {
                "temporal_sequence": 5.0,
                "adaptive_event": 3.8,
                "visual_event": 3.4,
                "fixed_window_event": 3.4,
                "event": 3.0,
                "visual_track": 2.6,
                "track": 2.0,
                "segment": 1.6,
                "caption_context": 1.4,
                "scope": 1.2,
                "visual_relation": 1.0,
            }.get(evidence_type, 0.0)
            score += 0.18 * len(TEMPORAL_EVIDENCE_TERMS & set(text.split()))
        if intents.get("multi_object_interaction"):
            score += {
                "visual_relation": 3.2,
                "adaptive_event": 2.8,
                "visual_track": 3.0,
                "visual_event": 2.4,
                "fixed_window_event": 2.4,
                "visual_object": 1.0,
                "track": 1.2,
                "event": 1.0,
            }.get(evidence_type, 0.0)
            score += 0.15 * len(INTERACTION_EVIDENCE_TERMS & set(text.split()))
        if intents.get("event_localization"):
            score += {
                "visual_event": 4.0,
                "fixed_window_event": 4.0,
                "adaptive_event": 4.2,
                "visual_track": 2.4,
                "visual_relation": 1.4,
                "visual_object": 0.8,
                "event": 1.8,
                "track": 1.0,
            }.get(evidence_type, 0.0)
            score += 0.15 * len(EVENT_EVIDENCE_TERMS & set(text.split()))

        if isinstance(evidence_item, dict):
            structured_keys = {
                "object_id",
                "track_id",
                "relation_id",
                "event_id",
                "timestamp",
                "frame_id",
                "segment_id",
                "bbox",
                "center",
                "distance_pixels",
                "start_time",
                "end_time",
                "compact_points",
            }
            score += 0.08 * sum(1 for key in structured_keys if evidence_item.get(key) is not None)
            provenance = evidence_item.get("provenance")
            if isinstance(provenance, dict) and provenance:
                score += 0.4
            try:
                confidence = float(evidence_item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if intents.get("evidence_grounded"):
                score += min(1.0, max(0.0, confidence))
                score += 0.3 if provenance else 0.0
        return score

    def _compact_evidence_preview(self, evidence_item: Any) -> str:
        if isinstance(evidence_item, dict):
            prefix = f"{self._evidence_type(evidence_item)}:{evidence_item.get('id') or evidence_item.get('node_id') or ''}".rstrip(":")
            text = self._evidence_text(evidence_item)
            return shorten(f"{prefix} {text}".strip(), 240)
        return shorten(str(evidence_item or ""), 240)

    def _visual_chain_attempted(self, plan: Any) -> bool:
        constraints = getattr(plan, "constraints", None) or {}
        selected_views = set(getattr(plan, "selected_views", []) or [])
        required_views = set(getattr(plan, "required_views", []) or [])
        query_type = str(getattr(plan, "query_type", ""))
        return (
            bool(constraints.get("visual_chain_priority"))
            or query_type == "instance_spatial_temporal"
            or bool((selected_views | required_views) & VISUAL_EVIDENCE_VIEWS)
        )

    def _dedupe_visual_evidence(self, evidence_items: list[dict[str, Any]], *, max_per_pair: int = 2) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        pair_counts: Counter[tuple[Any, ...]] = Counter()
        ranked = sorted(evidence_items, key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)
        for item in ranked:
            view = self._evidence_type(item)
            if view == "visual_relation":
                pair_key = self._relation_pair_key(item)
                if pair_counts[pair_key] >= max_per_pair:
                    continue
                dedupe_key = pair_key + (self._time_bucket(item.get("timestamp")),)
                pair_counts[pair_key] += 1
            else:
                dedupe_key = (view, item.get("id") or shorten(self._evidence_text(item), 120))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped.append(item)
        return deduped

    def _relation_pair_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        video_key = (
            item.get("video_id")
            or item.get("source_vid")
            or (item.get("provenance") or {}).get("video_id")
            or (item.get("provenance") or {}).get("source_vid")
            or (item.get("provenance") or {}).get("video_name")
            or "unknown_video"
        )
        left = item.get("target_track_id") or item.get("target_object_id") or item.get("object_id") or "unknown_left"
        right = item.get("related_track_id") or item.get("related_object_id") or "unknown_right"
        pair = tuple(sorted([str(left), str(right)]))
        return ("visual_relation", video_key, pair[0], pair[1], item.get("relation_type") or "unknown")

    def _time_bucket(self, value: Any, *, bucket_seconds: float = 10.0) -> int | None:
        try:
            return int(float(value) // bucket_seconds)
        except (TypeError, ValueError):
            return None

    def _visual_evidence_budgets(self, plan: Any, max_items: int) -> tuple[dict[str, int], list[str]]:
        intents = getattr(plan, "query_intents", None) or {}
        budgets = {"visual_relation": 0, "visual_object": 0, "visual_track": 0, "visual_event": 0}
        order: list[str] = []

        def add(view: str, count: int) -> None:
            if view == "visual_relation" and not self.visual_relations_enabled:
                return
            budgets[view] = max(budgets.get(view, 0), count)
            if view not in order:
                order.append(view)

        if intents.get("spatial_relation"):
            for view, count in [("visual_relation", 5), ("visual_object", 2), ("visual_track", 2), ("visual_event", 1)]:
                add(view, count)
        if intents.get("temporal_trajectory"):
            for view, count in [("visual_track", 5), ("visual_event", 3), ("visual_relation", 2), ("visual_object", 1)]:
                add(view, count)
        if any(intents.get(key) for key in TEMPORAL_SEQUENCE_INTENT_KEYS):
            for view, count in [("visual_event", 5), ("visual_track", 4), ("visual_relation", 2), ("visual_object", 1)]:
                add(view, count)
        if intents.get("multi_object_interaction"):
            for view, count in [("visual_relation", 4), ("visual_track", 4), ("visual_event", 3), ("visual_object", 1)]:
                add(view, count)
        if intents.get("event_localization"):
            for view, count in [("visual_event", 5), ("visual_track", 3), ("visual_relation", 2), ("visual_object", 1)]:
                add(view, count)
        if intents.get("evidence_grounded"):
            for view, count in [("visual_object", 3), ("visual_track", 2), ("visual_event", 2), ("visual_relation", 2)]:
                add(view, count)
        if not order:
            for view, count in [("visual_relation", 3), ("visual_track", 3), ("visual_event", 2), ("visual_object", 2)]:
                add(view, count)

        if sum(budgets.values()) > max_items:
            trimmed = {view: 0 for view in budgets}
            remaining = max_items
            while remaining > 0:
                progressed = False
                for view in order:
                    if trimmed[view] < budgets[view]:
                        trimmed[view] += 1
                        remaining -= 1
                        progressed = True
                        if remaining == 0:
                            break
                if not progressed:
                    break
            budgets = trimmed
        return budgets, order

    def _visual_record_text(self, view: str, row: dict[str, Any]) -> str:
        keys_by_view = {
            "visual_relation": ("evidence_text", "relation_type", "target_object_id", "related_object_id", "target_track_id", "related_track_id"),
            "visual_track": ("evidence_text", "motion_summary", "direction_text", "label", "track_id"),
            "visual_event": ("evidence_text", "summary", "event_type", "label", "track_id"),
            "visual_object": ("evidence_text", "text", "label", "color", "object_id", "track_id"),
        }
        return " ".join(str(row.get(key)) for key in keys_by_view.get(view, ()) if row.get(key))

    def _visual_record_score(self, view: str, row: dict[str, Any], plan: Any, query_tokens: set[str]) -> float:
        intents = getattr(plan, "query_intents", None) or {}
        text = self._visual_record_text(view, row)
        score = overlap_score(query_tokens, text)
        if view == "visual_relation":
            relation_type = str(row.get("relation_type") or "").lower()
            if relation_type in {"nearest_to", "overlap_or_near", "left_of", "right_of", "above", "below"}:
                score += 1.0
            if intents.get("spatial_relation") and relation_type:
                score += 2.0
            if row.get("distance_pixels") is not None:
                score += 0.4
        elif view == "visual_track":
            if row.get("direction_text") and row.get("direction_text") not in {"unknown", "stationary"}:
                score += 1.0
            if row.get("start_time") is not None and row.get("end_time") is not None:
                score += 0.8
            if row.get("compact_points") or row.get("bbox_sequence"):
                score += 0.7
            if intents.get("temporal_trajectory"):
                score += 2.0
            if any(intents.get(key) for key in TEMPORAL_SEQUENCE_INTENT_KEYS):
                score += 1.4
        elif view == "visual_event":
            if row.get("event_type"):
                score += 0.8
            if row.get("timestamp") is not None or row.get("start_time") is not None:
                score += 0.5
            if row.get("segment_id"):
                score += 0.3
            if intents.get("event_localization"):
                score += 2.0
            if any(intents.get(key) for key in TEMPORAL_SEQUENCE_INTENT_KEYS):
                score += 2.0
        elif view == "visual_object":
            if row.get("bbox") or row.get("bbox_center"):
                score += 0.4
            if row.get("timestamp") is not None:
                score += 0.3
            try:
                score += min(0.8, max(0.0, float(row.get("confidence", 0.0))))
            except (TypeError, ValueError):
                pass
        if intents.get("evidence_grounded") and row.get("provenance"):
            score += 0.5
        return score

    def _visual_evidence_item(self, view: str, row: dict[str, Any], score: float) -> dict[str, Any]:
        item_id = row.get("relation_id") or row.get("event_id") or row.get("track_id") or row.get("object_id") or row.get("id")
        role_by_view = {
            "visual_relation": "spatial_relation",
            "visual_track": "motion_evidence",
            "visual_event": "event_binding",
            "fixed_window_event": "event_binding",
            "visual_object": "attribute_grounding",
        }
        text = self._visual_record_text(view, row)
        if view == "visual_relation":
            text = (
                f"Visual relation: {text}. relation_type={row.get('relation_type') or 'unknown'}, "
                f"target={row.get('target_object_id') or 'unknown'}, related={row.get('related_object_id') or row.get('neighbor_object_id') or 'unknown'}, "
                f"target_track={row.get('target_track_id') or 'unknown'}, related_track={row.get('related_track_id') or row.get('neighbor_track_id') or 'unknown'}, "
                f"timestamp={row.get('timestamp') or 'unknown'}, distance_pixels={row.get('distance_pixels') or 'unknown'}."
            )
        elif view == "visual_track":
            text = (
                f"Visual trajectory: {text}. track_id={row.get('track_id') or 'unknown'}, "
                f"start_time={row.get('start_time') or 'unknown'}, end_time={row.get('end_time') or 'unknown'}, "
                f"compact_points={row.get('compact_points') or 'unknown'}."
            )
        elif view == "visual_event":
            text = (
                f"Visual event: {text}. event_id={row.get('event_id') or row.get('id') or 'unknown'}, "
                f"timestamp={row.get('timestamp') or row.get('start_time') or 'unknown'}, segment_id={row.get('segment_id') or 'unknown'}."
            )
        elif view == "visual_object":
            text = (
                f"Visual object: {text}. object_id={row.get('object_id') or row.get('id') or 'unknown'}, "
                f"track_id={row.get('track_id') or 'unknown'}, frame_id={row.get('frame_id') or 'unknown'}, "
                f"timestamp={row.get('timestamp') or 'unknown'}, confidence={row.get('confidence') or 'unknown'}."
            )
        return {
            "view": view,
            "id": item_id,
            "node_id": item_id,
            "score": float(score),
            "text": shorten(text, 950),
            "short_text": shorten(text, 260),
            "role": role_by_view.get(view, "visual_evidence"),
            "object_id": row.get("object_id") or row.get("target_object_id"),
            "track_id": row.get("track_id") or row.get("target_track_id"),
            "relation_id": row.get("relation_id"),
            "event_id": row.get("event_id"),
            "target_object_id": row.get("target_object_id"),
            "related_object_id": row.get("related_object_id") or row.get("neighbor_object_id"),
            "target_track_id": row.get("target_track_id"),
            "related_track_id": row.get("related_track_id") or row.get("neighbor_track_id"),
            "relation_type": row.get("relation_type"),
            "distance_pixels": row.get("distance_pixels"),
            "label": row.get("label"),
            "bbox": row.get("bbox"),
            "center": row.get("center") or row.get("bbox_center"),
            "timestamp": row.get("timestamp") or row.get("start_time"),
            "frame_id": row.get("frame_id"),
            "segment_id": row.get("segment_id"),
            "compact_points": row.get("compact_points"),
            "confidence": row.get("confidence"),
            "provenance": row.get("provenance") or {},
            "record": row,
            **self._video_fields_for_sources(row),
        }

    def _retrieve_visual_intent_evidence(
        self,
        query: str,
        plan: Any,
        query_tokens: set[str],
        *,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        constraints = getattr(plan, "constraints", {}) or {}
        max_items = min(self.max_evidence, int(constraints.get("max_evidence", self.max_evidence)), 12)
        if max_items <= 0:
            return []
        budgets, view_order = self._visual_evidence_budgets(plan, max_items)
        rows_by_view = {
            "visual_relation": self._filter_rows_by_video(self.visual_relations, video_filter),
            "visual_track": self._filter_rows_by_video(self.visual_tracks, video_filter),
            "visual_event": (
                self._filter_rows_by_video(self.fixed_window_events, video_filter)
                if self.ablation_event_mode == "fixed_window"
                else self._filter_rows_by_video(self.visual_events, video_filter)
            ),
            "visual_object": self._filter_rows_by_video(self.visual_objects, video_filter),
        }
        candidates_by_view: dict[str, list[dict[str, Any]]] = {}
        for view, rows in rows_by_view.items():
            if self.ablation_event_mode == "fixed_window" and view == "visual_event":
                candidates = [
                    self._hit(
                        "fixed_window_event",
                        row,
                        self._visual_record_score("visual_event", row, plan, query_tokens),
                        str(row.get("text") or row.get("summary") or ""),
                    )
                    for row in rows
                ]
            else:
                candidates = [
                    self._visual_evidence_item(view, row, self._visual_record_score(view, row, plan, query_tokens))
                    for row in rows
                ]
            candidates = [item for item in candidates if _safe_float(item.get("score"), 0.0) > 0.0]
            if view == "visual_relation":
                candidates = self._dedupe_visual_evidence(candidates, max_per_pair=2)
            candidates.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
            candidates_by_view[view] = candidates

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for view in view_order:
            limit = budgets.get(view, 0)
            if limit <= 0:
                continue
            for item in candidates_by_view.get(view, [])[:limit]:
                key = str(item.get("id") or self._compact_evidence_preview(item))
                if key in seen:
                    continue
                selected.append(item)
                seen.add(key)
                if len(selected) >= max_items:
                    break
            if len(selected) >= max_items:
                break

        if len(selected) < max_items:
            remaining = [
                item
                for candidates in candidates_by_view.values()
                for item in candidates
                if str(item.get("id") or self._compact_evidence_preview(item)) not in seen
            ]
            remaining.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
            for item in remaining:
                selected.append(item)
                seen.add(str(item.get("id") or self._compact_evidence_preview(item)))
                if len(selected) >= max_items:
                    break

        selected = self._dedupe_visual_evidence(selected, max_per_pair=2)
        selected.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
        return selected[:max_items]

    def _view_topk(self, query_type: str) -> dict[str, int]:
        topk = dict(DEFAULT_VIEW_TOPK)
        topk.update(QUERY_TYPE_VIEW_TOPK.get(query_type, {}))
        return topk

    def _retrieve_adaptive_events(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.ablation_event_mode == "fixed_window":
            return self._retrieve_fixed_window_events(
                query,
                query_tokens,
                limit=limit,
                fallback=fallback,
                video_filter=video_filter,
            )
        hits = []
        temporal_bonus = 0.35 if {"after", "before", "when", "then", "first", "last", "happen", "event"} & query_tokens else 0.0
        for record in self.adaptive_events:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            state = record.get("state_signature") or {}
            text = (
                f"{record.get('event_type', '')} {record.get('summary', '')} "
                f"{' '.join(record.get('action_tags') or [])} {' '.join(record.get('scene_tags') or [])} "
                f"{record.get('boundary_reason', '')} {state}"
            )
            score = overlap_score(query_tokens, text) + temporal_bonus
            if set(record.get("action_tags") or []) & query_tokens:
                score += 0.35
            if record.get("dominant_signals") and {"change", "transition", "then", "after", "before"} & query_tokens:
                score += 0.20
            score += min(0.20, float(record.get("change_score") or 0.0) * 0.20)
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hits.append(self._hit("adaptive_event", record, score, self._adaptive_event_text(record)))
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _retrieve_visual_view(
        self,
        view: str,
        plan: Any,
        query_tokens: set[str],
        *,
        limit: int,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.ablation_event_mode == "fixed_window" and view == "visual_event":
            return self._retrieve_fixed_window_events(
                "",
                query_tokens,
                limit=limit,
                video_filter=video_filter,
            )
        if view == "visual_relation" and not self.visual_relations_enabled:
            return []
        rows_by_view = {
            "visual_relation": self._filter_rows_by_video(self.visual_relations, video_filter),
            "visual_track": self._filter_rows_by_video(self.visual_tracks, video_filter),
            "visual_event": self._filter_rows_by_video(self.visual_events, video_filter),
            "visual_object": self._filter_rows_by_video(self.visual_objects, video_filter),
        }
        rows = rows_by_view.get(view, [])
        candidates = [
            self._visual_evidence_item(view, row, self._visual_record_score(view, row, plan, query_tokens))
            for row in rows
        ]
        candidates = [item for item in candidates if _safe_float(item.get("score"), 0.0) > 0.0]
        if view == "visual_relation":
            candidates = self._dedupe_visual_evidence(candidates, max_per_pair=2)
        candidates.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
        return candidates[:limit]

    def _pedestrian_crosswalk_expansion(
        self,
        query: str,
        plan: Any,
        query_tokens: set[str],
        *,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        def label_tokens(row: dict[str, Any]) -> set[str]:
            values = [row.get("label"), row.get("object_label"), row.get("category")]
            values.extend(row.get("aliases") or [])
            text = " ".join(str(value) for value in values if value)
            return set(text.lower().replace("_", " ").split())

        def target_rows(label_terms: set[str], *, limit: int, role: str) -> list[dict[str, Any]]:
            scored: list[tuple[float, dict[str, Any]]] = []
            for record in self.targets:
                if not self._row_matches_video_filter(record, video_filter):
                    continue
                tokens = label_tokens(record)
                text = f"{record.get('text', '')} {' '.join(record.get('attributes') or [])}"
                score = overlap_score(query_tokens | label_terms, text)
                if tokens & label_terms:
                    score += 2.0
                if score > 0:
                    hit = self._hit("target", record, score, record.get("text", ""))
                    hit["role"] = role
                    scored.append((score, hit))
            scored.sort(key=lambda row: row[0], reverse=True)
            return [item for _, item in scored[:limit]]

        def visual_object_rows(label_terms: set[str], *, limit: int, role: str) -> list[dict[str, Any]]:
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in self._filter_rows_by_video(self.visual_objects, video_filter):
                tokens = label_tokens(row)
                text = self._visual_record_text("visual_object", row)
                score = self._visual_record_score("visual_object", row, plan, query_tokens | label_terms)
                if tokens & label_terms:
                    score += 2.0
                if score > 0:
                    item = self._visual_evidence_item("visual_object", row, score)
                    item["role"] = role
                    scored.append((score, item))
            scored.sort(key=lambda row: row[0], reverse=True)
            return [item for _, item in scored[:limit]]

        items.extend(target_rows(PEDESTRIAN_LABEL_TERMS, limit=2, role="pedestrian_target"))
        items.extend(visual_object_rows(PEDESTRIAN_LABEL_TERMS, limit=2, role="pedestrian_visual_object"))
        items.extend(target_rows(CROSSWALK_LABEL_TERMS, limit=2, role="crosswalk_context_target"))
        items.extend(visual_object_rows(CROSSWALK_LABEL_TERMS, limit=1, role="crosswalk_visual_object"))
        items.extend(target_rows(VEHICLE_LABEL_TERMS, limit=2, role="nearby_vehicle_target"))
        items.extend(visual_object_rows(VEHICLE_LABEL_TERMS, limit=2, role="nearby_vehicle_visual_object"))

        relation_items = self._retrieve_visual_view(
            "visual_relation",
            plan,
            query_tokens | PEDESTRIAN_LABEL_TERMS | CROSSWALK_LABEL_TERMS | VEHICLE_LABEL_TERMS | {"near", "nearby"},
            limit=4,
            video_filter=video_filter,
        )
        for item in relation_items:
            item["role"] = "pedestrian_crosswalk_relation_context"
            item["score"] = _safe_float(item.get("score"), 0.0) + 1.0
        items.extend(relation_items)

        event_tokens = (
            query_tokens
            | PEDESTRIAN_LABEL_TERMS
            | CROSSWALK_LABEL_TERMS
            | VEHICLE_LABEL_TERMS
            | {"crossing", "moving", "slowing", "stopping", "waiting"}
        )
        event_items = self._retrieve_visual_view(
            "visual_event",
            plan,
            event_tokens,
            limit=3,
            video_filter=video_filter,
        )
        event_items.extend(
            self._retrieve_adaptive_events(
                query,
                event_tokens,
                limit=2,
                fallback=True,
                video_filter=video_filter,
            )
        )
        for item in event_items:
            item["role"] = "pedestrian_crosswalk_event_context"
            item["score"] = _safe_float(item.get("score"), 0.0) + 0.8
        items.extend(event_items)

        items = self._dedupe_evidence_items(items)
        items.sort(key=lambda item: self._intent_evidence_priority(plan, item), reverse=True)
        return items

    def _candidate_segment_ids(self, hit: dict[str, Any]) -> list[str]:
        record = hit.get("record") or {}
        provenance = hit.get("provenance") or {}
        if hit.get("segment_id"):
            return [str(hit["segment_id"])]
        if hit.get("segment_ids"):
            return [str(segment_id) for segment_id in hit.get("segment_ids") or []]
        if record.get("segment_id"):
            return [str(record["segment_id"])]
        if record.get("segment_ids"):
            return [str(segment_id) for segment_id in record.get("segment_ids") or []]
        if record.get("related_segment_ids"):
            return [str(segment_id) for segment_id in record.get("related_segment_ids") or []]
        if provenance.get("segment_id"):
            return [str(provenance["segment_id"])]
        return []

    def _segment_index(self, segment_id: str) -> tuple[str, int | None]:
        prefix, _, suffix = segment_id.rpartition("_")
        try:
            return prefix, int(suffix)
        except ValueError:
            return segment_id, None

    def _is_adjacent_to_segments(self, hit: dict[str, Any], anchors: set[str]) -> bool:
        candidate_ids = self._candidate_segment_ids(hit)
        if not candidate_ids:
            return False
        anchor_keys = [self._segment_index(segment_id) for segment_id in anchors]
        for segment_id in candidate_ids:
            if segment_id in anchors:
                return True
            prefix, index = self._segment_index(segment_id)
            if index is None:
                continue
            for anchor_prefix, anchor_index in anchor_keys:
                if prefix == anchor_prefix and anchor_index is not None and abs(index - anchor_index) <= 1:
                    return True
        return False

    def _compact_track_segments(self, segment_ids: list[str]) -> list[str]:
        unique = self._dedupe(segment_ids)
        if len(unique) <= 3:
            return unique
        return [unique[0], unique[len(unique) // 2], unique[-1]]

    def _compact_candidate(self, candidate: dict[str, Any], max_item_chars: int) -> dict[str, Any]:
        record = candidate.get("record") or {}
        segment_ids = self._candidate_segment_ids(candidate)
        return {
            "view": candidate.get("view"),
            "id": candidate.get("id"),
            "node_id": candidate.get("node_id"),
            "segment_ids": segment_ids,
            "short_text": shorten(candidate.get("text", ""), max_item_chars),
            "text": shorten(candidate.get("text", ""), max_item_chars),
            "role": candidate.get("role"),
            "score": float(candidate.get("score", 0.0)),
            "provenance": candidate.get("provenance") or record.get("provenance") or {},
            **self._video_fields_for_sources(record, candidate),
        }

    def _trim_evidence(self, evidence: list[dict[str, Any]], *, max_items: int, max_chars: int) -> list[dict[str, Any]]:
        if max_items <= 0 or max_chars <= 0:
            return []
        selected: list[dict[str, Any]] = []
        used_chars = 0
        for item in evidence:
            if len(selected) >= max_items:
                break
            text = str(item.get("text") or item.get("short_text") or "")
            item_chars = len(text)
            remaining_chars = max_chars - used_chars
            if item_chars > remaining_chars:
                if remaining_chars < 24:
                    continue
                item = dict(item)
                item["text"] = shorten(text, remaining_chars)
                item["short_text"] = shorten(item["text"], min(260, remaining_chars))
                item_chars = len(str(item.get("text") or item.get("short_text") or ""))
            selected.append(item)
            used_chars += item_chars
        return selected

    def _candidate_role(self, hit: dict[str, Any], plan: Any) -> str:
        roles = set(getattr(plan, "evidence_roles", []) or [])
        view = str(hit.get("view"))
        role_by_view = {
            "scope": ["supporting_segments", "key_segments", "scene_summary", "anchor_segment", "transcript_excerpt"],
            "target": ["target_presence", "target_context"],
            "track": ["motion_evidence", "before_context", "after_context"],
            "event": ["state_event", "event_summary", "before_context", "after_context"],
            "adaptive_event": ["event_summary", "state_event", "anchor_segment", "before_context", "after_context"],
            "fixed_window_event": ["event_summary", "state_event", "anchor_segment", "before_context", "after_context"],
            "visual_object": ["attribute_grounding", "target_presence"],
            "visual_track": ["motion_evidence"],
            "visual_event": ["event_binding", "state_event"],
            "visual_relation": ["spatial_relation"],
        }
        for role in role_by_view.get(view, []):
            if role in roles:
                return role
        return role_by_view.get(view, ["general_evidence"])[0]

    def _role_match_score(self, hit: dict[str, Any], plan: Any) -> float:
        return 1.0 if self._candidate_role(hit, plan) in set(getattr(plan, "evidence_roles", []) or []) else 0.35

    def _provenance_score(self, hit: dict[str, Any]) -> float:
        provenance = hit.get("provenance") or {}
        if self._candidate_segment_ids(hit):
            return 1.0
        if provenance.get("start_time") is not None or provenance.get("end_time") is not None:
            return 0.8
        return 0.3 if provenance else 0.0

    def _temporal_score(self, hit: dict[str, Any], query_tokens: set[str]) -> float:
        if not (query_tokens & TEMPORAL_QUERY_TERMS):
            return 0.0
        view = str(hit.get("view"))
        if view in {"track", "event"}:
            return 1.0
        provenance = hit.get("provenance") or {}
        return 0.6 if provenance.get("start_time") is not None or provenance.get("end_time") is not None else 0.2

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _retrieve_scopes(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        hits = []
        for record in self.scopes:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            text = f"{record.get('text', '')} {' '.join(record.get('scene_tags') or [])}"
            score = overlap_score(query_tokens, text)
            if record.get("transcript") and {"say", "said", "speak", "speech", "talk", "transcript"} & query_tokens:
                score += 0.35
            if record.get("scene_tags") and query_tokens & set(record.get("scene_tags") or []):
                score += 0.25
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hits.append(self._hit("scope", record, score, self._scope_text(record)))
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _retrieve_targets(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        hits = []
        for record in self.targets:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            label = str(record.get("label", "")).lower()
            aliases = set(str(alias).lower() for alias in record.get("aliases") or [])
            text = f"{label} {' '.join(aliases)} {record.get('text', '')} {' '.join(record.get('attributes') or [])}"
            score = overlap_score(query_tokens, text)
            if label in query_tokens or aliases & query_tokens:
                score += 0.75
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hits.append(self._hit("target", record, score, record.get("text", "")))
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _retrieve_tracks(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        hits = []
        for record in self.tracks:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            label = str(record.get("label", "")).lower()
            action_tags = set(record.get("action_tags") or [])
            text = f"{label} {' '.join(action_tags)} {record.get('motion_summary', '')}"
            score = overlap_score(query_tokens, text)
            if label in query_tokens:
                score += 0.55
            if action_tags & query_tokens:
                score += 0.45
            if len(record.get("segment_ids") or []) > 1 and {"after", "before", "track", "move", "trajectory"} & query_tokens:
                score += 0.35
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hits.append(self._hit("track", record, score, self._track_text(record)))
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _retrieve_events(
        self,
        query: str,
        query_tokens: set[str],
        *,
        limit: int,
        fallback: bool = False,
        video_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_fixed_window_event_mode():
            return self._retrieve_fixed_window_events(
                query,
                query_tokens,
                limit=limit,
                fallback=fallback,
                video_filter=video_filter,
            )
        hits = []
        temporal_bonus = 0.25 if {"after", "before", "when", "then", "first", "last", "happen", "event"} & query_tokens else 0.0
        for record in self.events:
            if not self._row_matches_video_filter(record, video_filter):
                continue
            state = record.get("state_signature") or {}
            text = f"{record.get('event_type', '')} {record.get('summary', '')} {' '.join(record.get('action_tags') or [])} {state}"
            score = overlap_score(query_tokens, text) + temporal_bonus
            if set(record.get("action_tags") or []) & query_tokens:
                score += 0.35
            if fallback:
                score += 0.01
            if score > 0 or fallback:
                hits.append(self._hit("event", record, score, self._event_text(record)))
        return sorted(hits, key=lambda row: row["score"], reverse=True)[:limit]

    def _target_label_coverage(self, *, video_filter: set[str] | None = None) -> dict[str, Any] | None:
        targets = [record for record in self.targets if self._row_matches_video_filter(record, video_filter)]
        counts = Counter(str(record.get("label", "")) for record in targets if record.get("label"))
        if not counts:
            return None
        labels = ", ".join(f"{label}={count}" for label, count in counts.most_common(20))
        return {
            "view": "coverage",
            "id": "target_label_coverage",
            "node_id": "coverage:target_label",
            "score": 3.0,
            "text": f"Target label coverage across the indexed video evidence: {labels}.",
            "provenance": {"source": "target_view_aggregate"},
            "record": {"label_counts": dict(counts.most_common())},
        }

    def _prioritize(self, hits: list[dict[str, Any]], plan: Any, query_tokens: set[str]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for hit in hits:
            key = str(hit.get("id"))
            keyword_score = min(1.0, max(0.0, float(hit.get("score", 0.0))))
            view_weight = min(1.0, max(0.0, float(plan.view_weights.get(str(hit.get("view")), 0.5))))
            weighted = (
                0.40 * keyword_score
                + 0.25 * view_weight
                + 0.20 * self._role_match_score(hit, plan)
                + 0.10 * self._provenance_score(hit)
                + 0.05 * self._temporal_score(hit, query_tokens)
            )
            hit = {**hit, "score": weighted, "weighted_score": weighted, "role": self._candidate_role(hit, plan)}
            if key not in deduped or weighted > float(deduped[key].get("weighted_score", 0.0)):
                deduped[key] = hit
        return sorted(deduped.values(), key=lambda row: row["weighted_score"], reverse=True)

    def _select_evidence_with_view_floor(
        self,
        ranked: list[dict[str, Any]],
        *,
        selected_views: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Keep all four views represented, then fill by weighted score."""

        if limit <= 0:
            return []
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()

        for view in selected_views[:limit]:
            view_candidates = [hit for hit in ranked if hit.get("view") == view and str(hit.get("id")) not in seen]
            if not view_candidates:
                continue
            hit = view_candidates[0]
            selected.append(hit)
            seen.add(str(hit.get("id")))

        for hit in ranked:
            if len(selected) >= limit:
                break
            key = str(hit.get("id"))
            if key in seen:
                continue
            selected.append(hit)
            seen.add(key)

        return sorted(selected, key=lambda row: row.get("weighted_score", 0.0), reverse=True)

    def _build_evidence_subgraph(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        node_ids = {item.get("node_id") for item in evidence if item.get("node_id")}
        subgraph = []
        seen = set()
        for relation in self.relations:
            source = relation.get("source")
            target = relation.get("target")
            if source in node_ids and target in node_ids:
                key = (source, relation.get("relation"), target)
                if key not in seen:
                    seen.add(key)
                    subgraph.append(relation)
        return subgraph

    def _hit(self, view: str, record: dict[str, Any], score: float, text: str) -> dict[str, Any]:
        return {
            "view": view,
            "id": record.get("id"),
            "node_id": record.get("node_id"),
            "score": float(score),
            "text": shorten(text, 950),
            "provenance": record.get("provenance") or {},
            "record": record,
            **self._video_fields_for_sources(record),
        }

    def _scope_text(self, record: dict[str, Any]) -> str:
        time_text = self._format_time_text(record.get("start_time"), record.get("end_time"))
        time_suffix = f" ({time_text})" if time_text else ""
        return (
            f"Scope evidence for segment {record.get('segment_id')} "
            f"{time_suffix}. "
            f"Scene tags: {', '.join(record.get('scene_tags') or []) or 'none'}. "
            f"Caption: {shorten(record.get('caption', ''), 520)} "
            f"Transcript: {shorten(record.get('transcript', ''), 260)}"
        )

    def _track_text(self, record: dict[str, Any]) -> str:
        return (
            f"{record.get('motion_summary', '')} "
            f"Related segments: {', '.join(record.get('segment_ids') or [])}. "
            f"Neighbor tracks: {', '.join(record.get('neighbor_track_ids') or []) or 'none'}."
        )

    def _event_text(self, record: dict[str, Any]) -> str:
        state = record.get("state_signature") or {}
        return (
            f"{record.get('summary', '')} "
            f"State signature: {state}. "
            f"Segments: {', '.join(record.get('related_segment_ids') or [])}."
        )

    def _adaptive_event_text(self, record: dict[str, Any]) -> str:
        return (
            f"{record.get('summary', '')} "
            f"Boundary reason: {record.get('boundary_reason', '')}. "
            f"Dominant signals: {', '.join(record.get('dominant_signals') or [])}. "
            f"Segments: {', '.join(record.get('related_segment_ids') or [])}."
        )


def build_prompt_package(package: dict[str, Any], formatted_context: str, *, response_type: str) -> str:
    temporal_instructions = ""
    answer_structure = """Preferred answer flow; adapt headings to the question instead of forcing a fixed template:
- Start with the most direct supported answer or most likely matching moment.
- Then explain the evidence that supports it.
- Put uncertainty, missing evidence, or evidence-boundary notes last."""
    temporal_query = bool(
        package.get("temporal_sequence_evidence")
        or package.get("temporal_aware_packing_used")
        or package.get("temporal_context_used")
    )
    if temporal_query:
        temporal_instructions = """
For temporal ordering, before/after interaction, or transition questions:
- Start with the best supported local answer, then qualify it against the evidence.
- Mention before/focal/after order when the evidence provides timestamps, segment ids, or adjacent segment order.
- Keep uncertainty, missing evidence, and evidence-boundary statements after the supported answer.
- For local temporal evidence, use the Temporal sequence evidence BEFORE / FOCAL EVENT / AFTER order.
- Treat caption_segment evidence as caption-supported best effort, not as direct object tracking.
- State order when timestamps, segment ids, or adjacent segment order support it.
- If labels or actions are named in support_text, use them as caption-supported details rather than replacing them with a generic summary.
- If evidence only shows co-presence, do not describe it as causality, yielding, or interaction.
- If identity tracking is insufficient, do not claim the same object persists across time.
- If before/after evidence is missing or timestamps are unclear, say that the precise order cannot be reliably determined.
"""
    return f"""---Role---

You are a grounded video database QA assistant using EVIQUE evidence.

---Goal---

Answer the user's question using only the compact EVIQUE evidence package.
The package is query-specific and contains Scope, Target, Track, Event, Adaptive Event, and optional visual evidence plus provenance relations.
Act as a confidence-calibrated moment description generator: give the most likely supported moment first, then calibrate details against evidence.
Fuse direct visual evidence with caption/scope context before answering.
When evidence supports the query, answer positively and directly before discussing uncertainty.
Prioritize concrete moment details when present: people, clothing, actions, held objects, scene, motion, and interactions.
If only part of the query is supported, describe the confirmed part first and keep the missing/uncertain part brief.
For open-ended traffic or scene semantics, caption/scope evidence can support a general answer when low-level visual events do not explicitly label the concept.
If direct visual evidence is limited but caption/scope evidence supports a general conclusion, give the general answer and identify it as caption-supported.
If the package does not support a detail, state that limitation clearly.
Do not hallucinate unsupported attributes, and do not spend long passages explaining evidence absence.
Respect video identity: never combine visual evidence from one video with caption/text evidence from another when video_id or source_vid is present.
{temporal_instructions}

---Target response length and format---

{response_type}

{answer_structure}

---Compact EVIQUE Evidence---

{formatted_context}
"""
