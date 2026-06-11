from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .utils import overlap_score, shorten, tokenize
from .video_identity import video_identity_values


PACKING_STRATEGY = "importance_budgeted_mmr"

SPATIAL_TERMS = {
    "above",
    "around",
    "behind",
    "below",
    "beside",
    "close",
    "closest",
    "front",
    "left",
    "near",
    "nearby",
    "nearest",
    "next",
    "right",
    "surrounding",
}
TEMPORAL_TERMS = {
    "after",
    "before",
    "change",
    "during",
    "first",
    "later",
    "last",
    "next",
    "order",
    "sequence",
    "then",
    "transition",
    "when",
}
MOTION_TERMS = {
    "approach",
    "direction",
    "enter",
    "exit",
    "leave",
    "move",
    "moving",
    "path",
    "route",
    "track",
    "trajectory",
    "turn",
}
TEMPORAL_AWARE_QUERY_TYPES = {
    "before_after",
    "event_localization",
    "interaction",
    "instance_spatial_temporal",
    "state_change",
    "temporal",
    "trajectory",
}
TEMPORAL_AWARE_INTENTS = {
    "event_localization",
    "instance_spatial_temporal",
    "multi_object_interaction",
    "temporal_interaction",
    "temporal_ordering",
    "temporal_trajectory",
    "transition",
}
TEMPORAL_TRIGGER_PHRASES = {
    "before",
    "after",
    "afterward",
    "afterwards",
    "immediately before",
    "immediately after",
    "then",
    "next",
    "start moving",
    "starts moving",
    "waiting",
    "stopped",
    "signal changes",
    "traffic starts",
    "turns",
    "enters",
    "crosses",
    "near crosswalk",
    "pedestrian",
}
RELATION_QUERY_TERMS = {
    "around",
    "beside",
    "close",
    "near",
    "nearby",
    "relative",
    "surrounding",
    "together",
}
EVENT_QUERY_TERMS = {
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
PEDESTRIAN_TERMS = {"crossing", "pedestrian", "person", "people", "walk", "walking"}
CROSSWALK_TERMS = {"crosswalk", "intersection", "road", "sidewalk"}
VEHICLE_TERMS = {"bus", "car", "truck", "vehicle", "vehicles"}
NEAR_RELATION_TERMS = {"near", "nearby", "nearest", "overlap", "same", "close"}


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


@dataclass
class EvidencePackerConfig:
    enabled: bool = True
    debug: bool = False
    token_budget: int = 3200
    char_budget: int = 12000
    core_ratio: float = 0.55
    support_ratio: float = 0.30
    context_ratio: float = 0.15
    min_core_items: int = 3
    min_packed_items: int = 6
    max_items: int = 12
    dedup_threshold: float = 0.75
    spatial_relation_min_items: int = 2
    temporal_event_min_items: int = 2
    temporal_aware_packing: bool = True
    temporal_window_segments: int = 1
    temporal_min_before: int = 2
    temporal_min_focal: int = 3
    temporal_min_after: int = 2
    temporal_max_supplement: int = 6
    pedestrian_crosswalk_expand: bool = True
    relation_supplement_min: int = 2
    event_supplement_min: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "token_budget": self.token_budget,
            "char_budget": self.char_budget,
            "core_ratio": self.core_ratio,
            "support_ratio": self.support_ratio,
            "context_ratio": self.context_ratio,
            "min_core_items": self.min_core_items,
            "min_packed_items": self.min_packed_items,
            "max_items": self.max_items,
            "dedup_threshold": self.dedup_threshold,
            "spatial_relation_min_items": self.spatial_relation_min_items,
            "temporal_event_min_items": self.temporal_event_min_items,
            "temporal_aware_packing": self.temporal_aware_packing,
            "temporal_window_segments": self.temporal_window_segments,
            "temporal_min_before": self.temporal_min_before,
            "temporal_min_focal": self.temporal_min_focal,
            "temporal_min_after": self.temporal_min_after,
            "temporal_max_supplement": self.temporal_max_supplement,
            "pedestrian_crosswalk_expand": self.pedestrian_crosswalk_expand,
            "relation_supplement_min": self.relation_supplement_min,
            "event_supplement_min": self.event_supplement_min,
        }


def get_evidence_packer_config() -> dict[str, Any]:
    return EvidencePackerConfig(
        enabled=_env_bool("EVIQUE_EVIDENCE_PACKER", True),
        debug=_env_bool("EVIQUE_EVIDENCE_PACKER_DEBUG", False),
        token_budget=max(1, _env_int("EVIQUE_EVIDENCE_TOKEN_BUDGET", 3200)),
        char_budget=max(1, _env_int("EVIQUE_EVIDENCE_CHAR_BUDGET", 12000)),
        core_ratio=max(0.0, _env_float("EVIQUE_EVIDENCE_CORE_RATIO", 0.55)),
        support_ratio=max(0.0, _env_float("EVIQUE_EVIDENCE_SUPPORT_RATIO", 0.30)),
        context_ratio=max(0.0, _env_float("EVIQUE_EVIDENCE_CONTEXT_RATIO", 0.15)),
        min_core_items=max(0, _env_int("EVIQUE_EVIDENCE_MIN_CORE_ITEMS", 3)),
        min_packed_items=max(1, _env_int("EVIQUE_EVIDENCE_MIN_PACKED_ITEMS", 6)),
        max_items=max(1, _env_int("EVIQUE_EVIDENCE_MAX_ITEMS", 12)),
        dedup_threshold=min(1.0, max(0.0, _env_float("EVIQUE_EVIDENCE_DEDUP_THRESHOLD", 0.75))),
        spatial_relation_min_items=max(0, _env_int("EVIQUE_EVIDENCE_SPATIAL_RELATION_MIN_ITEMS", 2)),
        temporal_event_min_items=max(0, _env_int("EVIQUE_EVIDENCE_TEMPORAL_EVENT_MIN_ITEMS", 2)),
        temporal_aware_packing=_env_bool("EVIQUE_TEMPORAL_AWARE_PACKING", True),
        temporal_window_segments=max(0, _env_int("EVIQUE_TEMPORAL_WINDOW_SEGMENTS", 1)),
        temporal_min_before=max(0, _env_int("EVIQUE_TEMPORAL_MIN_BEFORE", 2)),
        temporal_min_focal=max(0, _env_int("EVIQUE_TEMPORAL_MIN_FOCAL", 3)),
        temporal_min_after=max(0, _env_int("EVIQUE_TEMPORAL_MIN_AFTER", 2)),
        temporal_max_supplement=max(0, _env_int("EVIQUE_TEMPORAL_MAX_SUPPLEMENT", 6)),
        pedestrian_crosswalk_expand=_env_bool("EVIQUE_PEDESTRIAN_CROSSWALK_EXPAND", True),
        relation_supplement_min=max(0, _env_int("EVIQUE_RELATION_SUPPLEMENT_MIN", 2)),
        event_supplement_min=max(0, _env_int("EVIQUE_EVENT_SUPPLEMENT_MIN", 2)),
    ).to_dict()


class EvidencePacker:
    def __init__(self, config: dict[str, Any] | EvidencePackerConfig | None = None):
        if config is None:
            self.config = EvidencePackerConfig(**get_evidence_packer_config())
        elif isinstance(config, EvidencePackerConfig):
            self.config = config
        else:
            defaults = EvidencePackerConfig()
            values = defaults.to_dict()
            values.update(config)
            self.config = EvidencePackerConfig(**values)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def pack(
        self,
        query: str,
        candidate_evidence: list[dict[str, Any]],
        *,
        cost_plan: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
        package_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_plan = self._as_dict(query_plan)
        cost_plan = self._as_dict(cost_plan)
        package_metadata = self._as_dict(package_metadata)
        if not self.enabled:
            return {
                "packed_evidence": list(candidate_evidence),
                "dropped_evidence": [],
                "metadata": self._disabled_metadata(candidate_evidence),
            }

        query_tokens = set(tokenize(query))
        video_filter_values, video_filter_source, strict_video_filter_enabled = self._video_filter_info(package_metadata)
        anchor_video_values = video_filter_values if strict_video_filter_enabled else set()
        raw_items = [
            self._normalize_item(
                raw,
                index=index,
                query_tokens=query_tokens,
                cost_plan=cost_plan,
                query_plan=query_plan,
                anchor_video_values=anchor_video_values,
            )
            for index, raw in enumerate(candidate_evidence)
        ]
        estimated_candidate_tokens = sum(item["token_cost"] for item in raw_items)
        estimated_candidate_chars = sum(item["char_cost"] for item in raw_items)

        dropped: list[dict[str, Any]] = []
        usable: list[dict[str, Any]] = []
        for item in raw_items:
            if item["cross_video_mismatch"]:
                dropped.append(self._drop_record(item, "cross_video_mismatch"))
            else:
                usable.append(item)

        deduped, duplicate_drops = self._dedupe(usable)
        dropped.extend(duplicate_drops)
        self._mark_mandatory(deduped, query_tokens, cost_plan, query_plan)
        for item in deduped:
            item["layer"] = self._classify_layer(item, query_tokens, cost_plan, query_plan)
        temporal_state = self._temporal_pack_state(
            query,
            deduped,
            query_tokens=query_tokens,
            query_plan=query_plan,
        )

        min_packed_items_target = self._min_packed_items_target(query_plan, query_tokens, len(deduped))
        soft_token_target = self._soft_token_target(query_plan, query_tokens, estimated_candidate_tokens, len(deduped))
        deduped_tokens = sum(item["token_cost"] for item in deduped)
        deduped_chars = sum(item["char_cost"] for item in deduped)
        under_budget = (
            deduped_tokens <= self.config.token_budget
            and deduped_chars <= self.config.char_budget
            and len(deduped) <= self.config.max_items
        )
        if under_budget:
            selected, budget_drops, debug_trace = self._select_all_under_budget(deduped)
        else:
            selected, budget_drops, debug_trace = self._select_items(
                deduped,
                cost_plan,
                temporal_state=temporal_state,
                min_packed_items_target=min_packed_items_target,
                soft_token_target=soft_token_target,
            )
        dropped.extend(budget_drops)

        packed_evidence = [self._public_item(item) for item in selected]
        selected_ids = {item["stable_id"] for item in selected}
        for item in deduped:
            if item["stable_id"] not in selected_ids and not any(
                record.get("stable_id") == item["stable_id"] for record in dropped
            ):
                dropped.append(self._drop_record(item, "not_selected"))

        packed_chars = sum(self._item_chars(item) for item in packed_evidence)
        packed_tokens = sum(self._estimate_tokens_from_chars(self._item_chars(item)) for item in packed_evidence)
        temporal_metadata = self._temporal_metadata(temporal_state, selected)
        metadata = {
            "evidence_packer_enabled": True,
            "packing_strategy": PACKING_STRATEGY,
            "candidate_evidence_count": len(candidate_evidence),
            "packed_evidence_count": len(packed_evidence),
            "dropped_evidence_count": max(0, len(candidate_evidence) - len(packed_evidence)),
            "estimated_packed_tokens": packed_tokens,
            "estimated_candidate_tokens": estimated_candidate_tokens,
            "estimated_packed_chars": packed_chars,
            "estimated_candidate_chars": estimated_candidate_chars,
            "evidence_token_budget": self.config.token_budget,
            "evidence_char_budget": self.config.char_budget,
            "evidence_core_ratio": self.config.core_ratio,
            "evidence_support_ratio": self.config.support_ratio,
            "evidence_context_ratio": self.config.context_ratio,
            "evidence_min_core_items": self.config.min_core_items,
            "evidence_min_packed_items": self.config.min_packed_items,
            "evidence_max_items": self.config.max_items,
            "evidence_dedup_threshold": self.config.dedup_threshold,
            "evidence_spatial_relation_min_items": self.config.spatial_relation_min_items,
            "evidence_temporal_event_min_items": self.config.temporal_event_min_items,
            "evidence_temporal_aware_packing": self.config.temporal_aware_packing,
            "evidence_temporal_window_segments": self.config.temporal_window_segments,
            "evidence_temporal_min_before": self.config.temporal_min_before,
            "evidence_temporal_min_focal": self.config.temporal_min_focal,
            "evidence_temporal_min_after": self.config.temporal_min_after,
            "evidence_temporal_max_supplement": self.config.temporal_max_supplement,
            "packing_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in packed_evidence)),
            "dropped_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in dropped)),
            "video_filter_source": video_filter_source,
            "strict_video_filter_enabled": strict_video_filter_enabled,
            "packed_video_count": self._packed_video_count(packed_evidence),
            "dropped_cross_video_count": sum(1 for item in dropped if item.get("drop_reason") == "cross_video_mismatch"),
            "spatial_relation_supplement_used": bool(package_metadata.get("spatial_relation_supplement_used", False)),
            "spatial_relation_supplement_count": int(package_metadata.get("spatial_relation_supplement_count") or 0),
            "temporal_event_supplement_used": bool(package_metadata.get("temporal_event_supplement_used", False)),
            "temporal_event_supplement_count": int(package_metadata.get("temporal_event_supplement_count") or 0),
            "relation_supplement_used": bool(
                package_metadata.get("relation_supplement_used")
                or package_metadata.get("spatial_relation_supplement_used")
            ),
            "relation_supplement_count": int(
                package_metadata.get("relation_supplement_count")
                or package_metadata.get("spatial_relation_supplement_count")
                or 0
            ),
            "event_supplement_used": bool(
                package_metadata.get("event_supplement_used")
                or package_metadata.get("temporal_event_supplement_used")
            ),
            "event_supplement_count": int(
                package_metadata.get("event_supplement_count")
                or package_metadata.get("temporal_event_supplement_count")
                or 0
            ),
            "pedestrian_crosswalk_expansion_used": bool(
                package_metadata.get("pedestrian_crosswalk_expansion_used", False)
            ),
            "pedestrian_evidence_count": self._semantic_count(packed_evidence, PEDESTRIAN_TERMS),
            "crosswalk_evidence_count": self._semantic_count(packed_evidence, CROSSWALK_TERMS),
            "vehicle_near_pedestrian_evidence_count": self._vehicle_near_pedestrian_count(packed_evidence),
            "yielding_supported_by_visual_relation": self._yielding_supported_by_visual_relation(packed_evidence),
            "caption_fallback_used": bool(package_metadata.get("caption_fallback_used", False)),
            "min_packed_items_target": min_packed_items_target,
            "budget_fill_ratio": round(packed_tokens / max(float(self.config.token_budget), 1.0), 6),
            "mandatory_evidence_ids": [
                item["stable_id"] for item in selected if item.get("mandatory")
            ],
            "budget_exhausted": (
                not under_budget
                and (packed_tokens >= self.config.token_budget or packed_chars >= self.config.char_budget)
            ),
            "packing_debug_trace": debug_trace if self.config.debug else [],
        }
        metadata.update(temporal_metadata)
        return {
            "packed_evidence": packed_evidence,
            "dropped_evidence": dropped,
            "metadata": metadata,
        }

    def _disabled_metadata(self, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        estimated_tokens = sum(self._estimate_tokens(item) for item in evidence)
        estimated_chars = sum(self._item_chars(item) for item in evidence)
        return {
            "evidence_packer_enabled": False,
            "packing_strategy": "disabled_v5_passthrough",
            "candidate_evidence_count": len(evidence),
            "packed_evidence_count": len(evidence),
            "dropped_evidence_count": 0,
            "estimated_packed_tokens": estimated_tokens,
            "estimated_candidate_tokens": estimated_tokens,
            "estimated_packed_chars": estimated_chars,
            "estimated_candidate_chars": estimated_chars,
            "evidence_token_budget": self.config.token_budget,
            "evidence_char_budget": self.config.char_budget,
            "evidence_core_ratio": self.config.core_ratio,
            "evidence_support_ratio": self.config.support_ratio,
            "evidence_context_ratio": self.config.context_ratio,
            "evidence_min_core_items": self.config.min_core_items,
            "evidence_min_packed_items": self.config.min_packed_items,
            "evidence_max_items": self.config.max_items,
            "evidence_dedup_threshold": self.config.dedup_threshold,
            "evidence_spatial_relation_min_items": self.config.spatial_relation_min_items,
            "evidence_temporal_event_min_items": self.config.temporal_event_min_items,
            "evidence_temporal_aware_packing": self.config.temporal_aware_packing,
            "evidence_temporal_window_segments": self.config.temporal_window_segments,
            "evidence_temporal_min_before": self.config.temporal_min_before,
            "evidence_temporal_min_focal": self.config.temporal_min_focal,
            "evidence_temporal_min_after": self.config.temporal_min_after,
            "evidence_temporal_max_supplement": self.config.temporal_max_supplement,
            "packing_view_counts": dict(Counter(str(item.get("view") or "unknown") for item in evidence)),
            "dropped_view_counts": {},
            "video_filter_source": "disabled",
            "strict_video_filter_enabled": False,
            "packed_video_count": self._packed_video_count(evidence),
            "dropped_cross_video_count": 0,
            "spatial_relation_supplement_used": False,
            "spatial_relation_supplement_count": 0,
            "temporal_event_supplement_used": False,
            "temporal_event_supplement_count": 0,
            "relation_supplement_used": False,
            "relation_supplement_count": 0,
            "event_supplement_used": False,
            "event_supplement_count": 0,
            "temporal_aware_packing_used": False,
            "temporal_anchor_segment": None,
            "temporal_before_count": 0,
            "temporal_focal_count": 0,
            "temporal_after_count": 0,
            "temporal_supplement_count": 0,
            "pedestrian_crosswalk_expansion_used": False,
            "pedestrian_evidence_count": self._semantic_count(evidence, PEDESTRIAN_TERMS),
            "crosswalk_evidence_count": self._semantic_count(evidence, CROSSWALK_TERMS),
            "vehicle_near_pedestrian_evidence_count": self._vehicle_near_pedestrian_count(evidence),
            "yielding_supported_by_visual_relation": self._yielding_supported_by_visual_relation(evidence),
            "caption_fallback_used": False,
            "min_packed_items_target": min(len(evidence), self.config.min_packed_items),
            "budget_fill_ratio": round(estimated_tokens / max(float(self.config.token_budget), 1.0), 6),
            "mandatory_evidence_ids": [],
            "budget_exhausted": False,
            "packing_debug_trace": [],
        }

    def _normalize_item(
        self,
        raw: dict[str, Any],
        *,
        index: int,
        query_tokens: set[str],
        cost_plan: dict[str, Any],
        query_plan: dict[str, Any],
        anchor_video_values: set[str],
    ) -> dict[str, Any]:
        item = dict(raw) if isinstance(raw, dict) else {"view": "text", "text": str(raw)}
        view = self._view(item)
        text = self._text(item)
        stable_id = self._stable_id(item, index)
        token_cost = self._estimate_tokens_from_chars(len(text))
        video_values = self._item_video_values(item)
        cross_video_mismatch = bool(anchor_video_values and video_values and not (anchor_video_values & video_values))
        score_parts = self._score_parts(
            item,
            view=view,
            text=text,
            query_tokens=query_tokens,
            token_cost=token_cost,
            cost_plan=cost_plan,
            query_plan=query_plan,
            anchor_video_values=anchor_video_values,
            item_video_values=video_values,
            cross_video_mismatch=cross_video_mismatch,
        )
        importance_score = (
            score_parts["relevance_score"]
            + score_parts["planner_priority_score"]
            + score_parts["evidence_confidence_score"]
            + score_parts["query_intent_match_score"]
            + score_parts["temporal_alignment_score"]
            + score_parts["video_alignment_score"]
            + score_parts["coverage_gain_score"]
            - score_parts["redundancy_penalty"]
            - score_parts["token_cost_penalty"]
            - score_parts["noise_risk_penalty"]
        )
        return {
            "raw": item,
            "index": index,
            "view": view,
            "text": text,
            "text_tokens": set(tokenize(text)),
            "stable_id": stable_id,
            "token_cost": token_cost,
            "char_cost": len(text),
            "video_values": video_values,
            "cross_video_mismatch": cross_video_mismatch,
            "score_parts": score_parts,
            "importance_score": round(importance_score, 6),
            "mandatory": False,
            "mandatory_reason": "",
            "layer": "supporting",
        }

    def _score_parts(
        self,
        item: dict[str, Any],
        *,
        view: str,
        text: str,
        query_tokens: set[str],
        token_cost: int,
        cost_plan: dict[str, Any],
        query_plan: dict[str, Any],
        anchor_video_values: set[str],
        item_video_values: set[str],
        cross_video_mismatch: bool,
    ) -> dict[str, float]:
        relevance = 3.0 * overlap_score(query_tokens, text)
        planner_priority = self._planner_priority(view, cost_plan)
        confidence = self._confidence_score(item)
        intent_match = self._intent_match_score(view, text, query_tokens, query_plan)
        temporal_alignment = self._temporal_alignment_score(item, view, query_tokens)
        if cross_video_mismatch:
            video_alignment = -6.0
        elif anchor_video_values and item_video_values:
            video_alignment = 0.45
        else:
            video_alignment = 0.0
        coverage_gain = self._static_coverage_score(item, view)
        token_cost_penalty = min(1.2, token_cost / max(float(self.config.token_budget), 1.0) * 2.0)
        noise_risk = self._noise_risk_score(item, view, text, confidence)
        return {
            "relevance_score": round(relevance, 6),
            "planner_priority_score": round(planner_priority, 6),
            "evidence_confidence_score": round(confidence, 6),
            "query_intent_match_score": round(intent_match, 6),
            "temporal_alignment_score": round(temporal_alignment, 6),
            "video_alignment_score": round(video_alignment, 6),
            "coverage_gain_score": round(coverage_gain, 6),
            "redundancy_penalty": 0.0,
            "token_cost_penalty": round(token_cost_penalty, 6),
            "noise_risk_penalty": round(noise_risk, 6),
        }

    def _planner_priority(self, view: str, cost_plan: dict[str, Any]) -> float:
        if not cost_plan:
            return 0.0
        score = 0.0
        anchor_view = str(cost_plan.get("anchor_view") or "")
        if view == anchor_view:
            score += 1.25
        view_order = [str(value) for value in cost_plan.get("view_order") or []]
        if view in view_order:
            idx = view_order.index(view)
            score += 0.85 * (len(view_order) - idx) / max(len(view_order), 1)
        if view in {str(value) for value in cost_plan.get("views_skipped") or []}:
            score -= 0.35
        return score

    def _confidence_score(self, item: dict[str, Any]) -> float:
        values: list[float] = []
        for source in [item, item.get("record") if isinstance(item.get("record"), dict) else {}]:
            if not isinstance(source, dict):
                continue
            for key in (
                "score",
                "weighted_score",
                "confidence",
                "relation_confidence",
                "detection_confidence",
                "change_score",
            ):
                value = source.get(key)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 1.0:
                    numeric = numeric / (numeric + 1.0)
                values.append(max(0.0, min(1.0, numeric)))
        if not values:
            return 0.0
        return min(1.4, 0.35 + max(values))

    def _intent_match_score(
        self,
        view: str,
        text: str,
        query_tokens: set[str],
        query_plan: dict[str, Any],
    ) -> float:
        intents = query_plan.get("query_intents") or {}
        query_type = str(query_plan.get("query_type") or "")
        text_tokens = set(tokenize(text))
        score = 0.0
        spatial = bool(intents.get("spatial_relation")) or query_type in {"spatial_relation", "instance_spatial_temporal"} or bool(query_tokens & SPATIAL_TERMS)
        temporal = (
            bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or query_type in {"before_after", "temporal", "state_change", "event", "event_localization"}
            or bool(query_tokens & TEMPORAL_TERMS)
        )
        motion = bool(intents.get("temporal_trajectory")) or query_type in {"trajectory", "interaction"} or bool(query_tokens & MOTION_TERMS)
        if spatial:
            score += {
                "visual_relation": 1.8,
                "nearby_object_context": 1.3,
                "visual_object": 0.8,
                "visual_track": 0.55,
                "target": 0.45,
            }.get(view, 0.0)
            score += 0.12 * len(SPATIAL_TERMS & text_tokens)
        if temporal:
            score += {
                "adaptive_event": 1.7,
                "visual_event": 1.45,
                "event": 1.15,
                "temporal_sequence": 1.4,
                "track": 0.55,
                "visual_track": 0.65,
            }.get(view, 0.0)
            score += 0.12 * len(TEMPORAL_TERMS & text_tokens)
        if motion:
            score += {
                "visual_track": 1.65,
                "track": 1.25,
                "adaptive_event": 0.75,
                "visual_event": 0.65,
            }.get(view, 0.0)
            score += 0.12 * len(MOTION_TERMS & text_tokens)
        if query_type in {"scene", "general_description", "speech"} and view in {"scope", "segment", "caption_context"}:
            score += 1.0
        if query_type in {"object_list", "object_grounding"} and view in {"target", "visual_object"}:
            score += 1.0
        return score

    def _temporal_alignment_score(self, item: dict[str, Any], view: str, query_tokens: set[str]) -> float:
        score = 0.0
        if item.get("temporal_alignment") == "aligned":
            score += 0.55
        elif item.get("temporal_alignment"):
            score += 0.20
        if item.get("timestamp") is not None or item.get("start_time") is not None or item.get("end_time") is not None:
            score += 0.25
        if item.get("segment_id") or item.get("segment_ids"):
            score += 0.15
        if query_tokens & (TEMPORAL_TERMS | MOTION_TERMS) and view in {"adaptive_event", "event", "visual_event", "visual_track", "track"}:
            score += 0.20
        return score

    def _static_coverage_score(self, item: dict[str, Any], view: str) -> float:
        fields = {
            "object_id",
            "target_object_id",
            "related_object_id",
            "track_id",
            "target_track_id",
            "related_track_id",
            "relation_type",
            "event_id",
            "event_type",
            "segment_id",
            "timestamp",
            "start_time",
            "end_time",
        }
        score = 0.08 * sum(1 for field in fields if item.get(field) not in (None, "", []))
        if view in {"visual_relation", "visual_track", "adaptive_event", "visual_event"}:
            score += 0.25
        return min(1.1, score)

    def _noise_risk_score(self, item: dict[str, Any], view: str, text: str, confidence: float) -> float:
        risk = 0.0
        text_len = len(text)
        if view in {"scope", "segment", "caption_context"} and text_len > 800:
            risk += 0.45
        if view in {"caption_context", "scope"} and confidence < 0.45:
            risk += 0.20
        if view in {"visual_object", "target"} and confidence and confidence < 0.55:
            risk += 0.35
        if view == "visual_relation" and confidence and confidence < 0.45:
            risk += 0.25
        if text_len > 1200:
            risk += 0.35
        if not item.get("provenance") and not item.get("segment_id") and not item.get("timestamp"):
            risk += 0.10
        return risk

    def _dedupe(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for item in sorted(items, key=lambda row: row["importance_score"], reverse=True):
            key = self._dedupe_key(item)
            if key in seen_keys:
                dropped.append(self._drop_record(item, "duplicate_key"))
                continue
            too_similar = False
            for selected in kept:
                if self._similarity_scope(item, selected) and self._jaccard(
                    item["text_tokens"], selected["text_tokens"]
                ) >= self.config.dedup_threshold:
                    too_similar = True
                    break
            if too_similar:
                dropped.append(self._drop_record(item, "duplicate_text"))
                continue
            seen_keys.add(key)
            kept.append(item)
        return kept, dropped

    def _mark_mandatory(
        self,
        items: list[dict[str, Any]],
        query_tokens: set[str],
        cost_plan: dict[str, Any],
        query_plan: dict[str, Any],
    ) -> None:
        anchor_view = str(cost_plan.get("anchor_view") or "")
        self._mark_top_by_view(items, anchor_view, max(1, self.config.min_core_items), "anchor_view_core")

        query_type = str(query_plan.get("query_type") or "")
        intents = query_plan.get("query_intents") or {}
        spatial = bool(intents.get("spatial_relation")) or query_type in {"spatial_relation", "instance_spatial_temporal"} or bool(query_tokens & SPATIAL_TERMS)
        temporal = (
            bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or query_type in {"before_after", "temporal", "state_change", "event", "event_localization"}
            or bool(query_tokens & TEMPORAL_TERMS)
        )
        motion = bool(intents.get("temporal_trajectory")) or query_type in {"trajectory", "interaction"} or bool(query_tokens & MOTION_TERMS)

        if spatial:
            self._mark_top_by_view(
                items,
                "visual_relation",
                self.config.spatial_relation_min_items,
                "spatial_relation_floor",
            )
            self._mark_top_by_view(items, "nearby_object_context", 1, "spatial_context_floor")
        if temporal:
            remaining = self.config.temporal_event_min_items
            for view in ("adaptive_event", "visual_event", "event"):
                marked = self._mark_top_by_view(items, view, remaining, "temporal_event_floor")
                remaining = max(0, remaining - marked)
                if remaining <= 0:
                    break
        if motion:
            remaining = max(1, self.config.temporal_event_min_items)
            for view in ("visual_track", "track"):
                marked = self._mark_top_by_view(items, view, remaining, "motion_track_floor")
                remaining = max(0, remaining - marked)
                if remaining <= 0:
                    break

        if self.config.min_core_items > 0:
            core_count = sum(1 for item in items if item.get("mandatory"))
            if core_count < self.config.min_core_items:
                query_type = str(query_plan.get("query_type") or "")
                context_views = {"scope", "segment", "caption_context"}
                scene_query = query_type in {"scene", "general_description", "speech"}
                for item in sorted(items, key=lambda row: row["importance_score"], reverse=True):
                    if core_count >= self.config.min_core_items:
                        break
                    if item.get("mandatory"):
                        continue
                    if float(item.get("importance_score") or 0.0) <= 0.20:
                        continue
                    if item["view"] in context_views and not scene_query and item["view"] != str(cost_plan.get("anchor_view") or ""):
                        continue
                    item["mandatory"] = True
                    item["mandatory_reason"] = "min_core_items"
                    core_count += 1

    def _mark_top_by_view(self, items: list[dict[str, Any]], view: str, count: int, reason: str) -> int:
        if not view or count <= 0:
            return 0
        marked = 0
        for item in sorted(
            [row for row in items if row["view"] == view],
            key=lambda row: row["importance_score"],
            reverse=True,
        )[:count]:
            item["mandatory"] = True
            item["mandatory_reason"] = reason
            marked += 1
        return marked

    def _classify_layer(
        self,
        item: dict[str, Any],
        query_tokens: set[str],
        cost_plan: dict[str, Any],
        query_plan: dict[str, Any],
    ) -> str:
        if item.get("mandatory"):
            return "core"
        view = item["view"]
        score = float(item["importance_score"])
        if score < 0.05:
            return "low_value"
        query_type = str(query_plan.get("query_type") or "")
        intents = query_plan.get("query_intents") or {}
        if view == str(cost_plan.get("anchor_view") or "") and score >= 0.35:
            return "core"
        if (bool(intents.get("spatial_relation")) or query_type in {"spatial_relation", "instance_spatial_temporal"} or query_tokens & SPATIAL_TERMS) and view in {
            "visual_relation",
            "nearby_object_context",
        }:
            return "core"
        if (bool(intents.get("temporal_trajectory")) or query_type in {"trajectory", "interaction"} or query_tokens & MOTION_TERMS) and view in {
            "visual_track",
            "track",
        }:
            return "core"
        if (
            bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or query_type in {"before_after", "temporal", "state_change", "event", "event_localization"}
            or query_tokens & TEMPORAL_TERMS
        ) and view in {"adaptive_event", "visual_event", "event"}:
            return "core"
        if view in {"scope", "segment", "caption_context"}:
            return "context" if item["char_cost"] > 500 else "supporting"
        if score < 0.25:
            return "low_value"
        return "supporting"

    def _temporal_pack_state(
        self,
        query: str,
        items: list[dict[str, Any]],
        *,
        query_tokens: set[str],
        query_plan: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.temporal_aware_packing or not self._temporal_aware_query(query, query_tokens, query_plan):
            return {"used": False}
        anchor = self._temporal_anchor_item(items)
        if anchor is None:
            return {"used": False, "reason": "no_temporal_anchor"}
        anchor_segment = self._primary_segment_id(anchor["raw"])
        anchor_index = self._primary_segment_index(anchor["raw"])
        anchor_time = self._item_start_time(anchor["raw"])
        anchor_end = self._item_end_time(anchor["raw"])
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            bucket = self._temporal_bucket(
                item,
                anchor_segment=anchor_segment,
                anchor_index=anchor_index,
                anchor_time=anchor_time,
                anchor_end=anchor_end,
            )
            if not bucket:
                continue
            item["temporal_bucket"] = bucket
            buckets[bucket].append(item)
        for rows in buckets.values():
            rows.sort(key=lambda row: row["importance_score"], reverse=True)
        return {
            "used": True,
            "anchor_id": anchor["stable_id"],
            "anchor_segment": anchor_segment,
            "anchor_index": anchor_index,
            "anchor_time": anchor_time,
            "buckets": dict(buckets),
            "required": {
                "before": self.config.temporal_min_before,
                "focal": self.config.temporal_min_focal,
                "after": self.config.temporal_min_after,
            },
        }

    def _temporal_aware_query(
        self,
        query: str,
        query_tokens: set[str],
        query_plan: dict[str, Any],
    ) -> bool:
        intents = query_plan.get("query_intents") or {}
        query_type = str(query_plan.get("query_type") or "")
        if query_type in TEMPORAL_AWARE_QUERY_TYPES:
            return True
        if any(bool(intents.get(key)) for key in TEMPORAL_AWARE_INTENTS):
            return True
        lowered = query.lower()
        return bool(query_tokens & (TEMPORAL_TERMS | MOTION_TERMS | PEDESTRIAN_TERMS)) or any(
            phrase in lowered for phrase in TEMPORAL_TRIGGER_PHRASES
        )

    def _temporal_anchor_item(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [
            item
            for item in items
            if self._primary_segment_id(item["raw"])
            or self._primary_segment_index(item["raw"]) is not None
            or self._item_start_time(item["raw"]) is not None
        ]
        if not candidates:
            return None
        view_priority = {
            "adaptive_event": 9,
            "event": 8,
            "visual_event": 8,
            "temporal_sequence": 8,
            "segment": 7,
            "caption_context": 7,
            "visual_track": 6,
            "track": 6,
            "target": 4,
            "visual_object": 4,
            "scope": 3,
        }
        return max(
            candidates,
            key=lambda item: (
                view_priority.get(item["view"], 1),
                float(item.get("importance_score") or 0.0),
                -int(item.get("index") or 0),
            ),
        )

    def _temporal_bucket(
        self,
        item: dict[str, Any],
        *,
        anchor_segment: str | None,
        anchor_index: tuple[str, int] | None,
        anchor_time: float | None,
        anchor_end: float | None,
    ) -> str | None:
        raw = item["raw"]
        item_index = self._primary_segment_index(raw)
        window = max(0, int(self.config.temporal_window_segments))
        if anchor_index is not None and item_index is not None and anchor_index[0] == item_index[0]:
            delta = item_index[1] - anchor_index[1]
            if delta == 0:
                return "focal"
            if -window <= delta < 0:
                return "before"
            if 0 < delta <= window:
                return "after"
        if anchor_segment and anchor_segment in self._segment_ids(raw):
            return "focal"

        start = self._item_start_time(raw)
        end = self._item_end_time(raw)
        if anchor_time is None or start is None:
            return None
        anchor_stop = anchor_end if anchor_end is not None else anchor_time
        item_stop = end if end is not None else start
        if item_stop < anchor_time:
            return "before"
        if start > anchor_stop:
            return "after"
        return "focal"

    def _select_temporal_floor_items(self, temporal_state: dict[str, Any], *, max_items: int) -> set[str]:
        buckets = temporal_state.get("buckets") or {}
        required = temporal_state.get("required") or {}
        selected: set[str] = set()
        for bucket in ("before", "focal", "after"):
            limit = max(0, int(required.get(bucket) or 0))
            for item in list(buckets.get(bucket) or [])[:limit]:
                if len(selected) >= max_items:
                    return selected
                selected.add(item["stable_id"])
        return selected

    def _ordered_temporal_floor_items(
        self,
        items: list[dict[str, Any]],
        selected_ids: set[str],
    ) -> list[dict[str, Any]]:
        by_id = {item["stable_id"]: item for item in items}
        ordered: list[dict[str, Any]] = []
        for bucket in ("before", "focal", "after"):
            bucket_items = [
                by_id[stable_id]
                for stable_id in selected_ids
                if stable_id in by_id and by_id[stable_id].get("temporal_bucket") == bucket
            ]
            bucket_items.sort(key=lambda row: row["importance_score"], reverse=True)
            ordered.extend(bucket_items)
        return ordered

    def _temporal_metadata(self, temporal_state: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(item.get("temporal_bucket") or "") for item in selected)
        supplement_count = sum(
            1
            for item in selected
            if str(item.get("packing_reason") or "").startswith("temporal_")
        )
        return {
            "temporal_aware_packing_used": bool(temporal_state.get("used")),
            "temporal_anchor_segment": temporal_state.get("anchor_segment"),
            "temporal_before_count": int(counts.get("before", 0)),
            "temporal_focal_count": int(counts.get("focal", 0)),
            "temporal_after_count": int(counts.get("after", 0)),
            "temporal_supplement_count": int(supplement_count),
        }

    def _primary_segment_id(self, item: dict[str, Any]) -> str | None:
        segment_ids = self._segment_ids(item)
        return segment_ids[0] if segment_ids else None

    def _segment_ids(self, item: dict[str, Any]) -> list[str]:
        values: list[str] = []

        def add(raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, str):
                candidates = [raw]
            elif isinstance(raw, (list, tuple, set)):
                candidates = [str(value) for value in raw if value]
            else:
                candidates = [str(raw)]
            for value in candidates:
                if value and value not in values:
                    values.append(value)

        for source in (item, item.get("record") or {}, item.get("provenance") or {}):
            if not isinstance(source, dict):
                continue
            add(source.get("segment_id"))
            add(source.get("segment_ids"))
            add(source.get("related_segment_ids"))
        return values

    def _primary_segment_index(self, item: dict[str, Any]) -> tuple[str, int] | None:
        for source in (item, item.get("record") or {}, item.get("provenance") or {}):
            if not isinstance(source, dict):
                continue
            raw = source.get("segment_index")
            if raw is not None:
                try:
                    return self._segment_index_prefix(source), int(raw)
                except (TypeError, ValueError):
                    pass
        segment_id = self._primary_segment_id(item)
        if not segment_id:
            return None
        prefix, _, suffix = str(segment_id).rpartition("_")
        try:
            return prefix, int(suffix)
        except ValueError:
            return None

    def _segment_index_prefix(self, source: dict[str, Any]) -> str:
        video = source.get("video_id") or source.get("source_vid") or source.get("video_name")
        if video:
            return str(video)
        segment_id = source.get("segment_id")
        if segment_id:
            return str(segment_id).rpartition("_")[0]
        return ""

    def _item_start_time(self, item: dict[str, Any]) -> float | None:
        return self._first_float(item, ("timestamp", "time", "frame_timestamp", "start_time"))

    def _item_end_time(self, item: dict[str, Any]) -> float | None:
        return self._first_float(item, ("end_time",))

    def _first_float(self, item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for source in (item, item.get("record") or {}, item.get("provenance") or {}):
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            provenance = source.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in keys:
                    value = provenance.get(key)
                    if value is None:
                        continue
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        return None

    def _select_items(
        self,
        items: list[dict[str, Any]],
        cost_plan: dict[str, Any],
        *,
        temporal_state: dict[str, Any],
        min_packed_items_target: int,
        soft_token_target: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        selected: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        debug_trace: list[dict[str, Any]] = []
        max_items = max(1, self.config.max_items)
        used_tokens = 0
        used_chars = 0

        def add_item(item: dict[str, Any], reason: str, *, force: bool = False) -> bool:
            nonlocal used_tokens, used_chars
            if len(selected) >= max_items:
                dropped.append(self._drop_record(item, "max_items_reached"))
                return False
            packed = self._packed_copy(
                item,
                remaining_chars=max(0, self.config.char_budget - used_chars),
                force=force,
            )
            packed_chars = self._item_chars(packed)
            packed_tokens = self._estimate_tokens_from_chars(packed_chars)
            if not force and (
                used_tokens + packed_tokens > self.config.token_budget
                or used_chars + packed_chars > self.config.char_budget
            ):
                dropped.append(self._drop_record(item, "budget_exceeded"))
                return False
            item["packed_item"] = packed
            item["packed_token_cost"] = packed_tokens
            item["packed_char_cost"] = packed_chars
            item["packing_rank"] = len(selected) + 1
            item["packing_reason"] = reason
            selected.append(item)
            used_tokens += packed_tokens
            used_chars += packed_chars
            if self.config.debug and len(debug_trace) < 80:
                debug_trace.append(
                    {
                        "action": "select",
                        "id": item["stable_id"],
                        "view": item["view"],
                        "layer": item["layer"],
                        "reason": reason,
                        "importance_score": item["importance_score"],
                        "tokens": packed_tokens,
                        "used_tokens": used_tokens,
                        "used_chars": used_chars,
                    }
            )
            return True

        temporal_floor_ids: set[str] = set()
        if temporal_state.get("used"):
            temporal_floor_ids = self._select_temporal_floor_items(
                temporal_state,
                max_items=max_items,
            )
            for item in self._ordered_temporal_floor_items(items, temporal_floor_ids):
                add_item(item, f"temporal_{item.get('temporal_bucket')}_floor")

        selected_ids = {item["stable_id"] for item in selected}
        mandatory_core = [
            item
            for item in items
            if item["stable_id"] not in selected_ids
            and (item.get("mandatory") or item.get("layer") == "core")
        ]
        for item in sorted(mandatory_core, key=lambda row: (bool(row.get("mandatory")), row["importance_score"]), reverse=True):
            add_item(item, "mandatory_core" if item.get("mandatory") else "core", force=True)

        selected_ids = {item["stable_id"] for item in selected}
        remaining = [item for item in items if item["stable_id"] not in selected_ids]
        high_confidence_stop = "high_confidence" in str(cost_plan.get("stop_reason") or "")
        context_added = 0

        for layer in ("supporting", "context", "low_value"):
            layer_items = [item for item in remaining if item.get("layer") == layer]
            while layer_items and len(selected) < max_items:
                scored = [
                    (self._marginal_value(item, selected), item)
                    for item in layer_items
                ]
                scored.sort(key=lambda row: row[0], reverse=True)
                value, item = scored[0]
                should_fill = len(selected) < min_packed_items_target or (
                    soft_token_target > 0 and used_tokens < soft_token_target
                )
                if value <= -0.75 and not should_fill:
                    for _, rejected in scored:
                        dropped.append(self._drop_record(rejected, "low_marginal_value"))
                    break
                if value <= -1.50:
                    for _, rejected in scored:
                        dropped.append(self._drop_record(rejected, "very_low_marginal_value"))
                    break
                if high_confidence_stop and layer == "context" and context_added >= 1 and not should_fill:
                    for _, rejected in scored:
                        dropped.append(self._drop_record(rejected, "high_confidence_context_cap"))
                    break
                before_count = len(selected)
                if not add_item(item, f"{layer}_mmr"):
                    layer_items = [row for row in layer_items if row["stable_id"] != item["stable_id"]]
                    continue
                if layer == "context" and len(selected) > before_count:
                    context_added += 1
                layer_items = [row for row in layer_items if row["stable_id"] != item["stable_id"]]

        return selected, dropped, debug_trace

    def _select_all_under_budget(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        selected: list[dict[str, Any]] = []
        debug_trace: list[dict[str, Any]] = []
        ranked = sorted(
            items,
            key=lambda row: (
                not bool(row.get("mandatory")),
                {"core": 0, "supporting": 1, "context": 2, "low_value": 3}.get(str(row.get("layer")), 2),
                -float(row.get("importance_score") or 0.0),
                int(row.get("index") or 0),
            ),
        )
        for item in ranked[: self.config.max_items]:
            item["packed_item"] = self._packed_copy(item, remaining_chars=self.config.char_budget, force=True)
            item["packed_token_cost"] = self._estimate_tokens(item["packed_item"])
            item["packed_char_cost"] = self._item_chars(item["packed_item"])
            item["packing_rank"] = len(selected) + 1
            item["packing_reason"] = "under_budget_keep_valid"
            selected.append(item)
            if self.config.debug and len(debug_trace) < 80:
                debug_trace.append(
                    {
                        "action": "select",
                        "id": item["stable_id"],
                        "view": item["view"],
                        "layer": item["layer"],
                        "reason": "under_budget_keep_valid",
                        "importance_score": item["importance_score"],
                        "tokens": item["packed_token_cost"],
                    }
                )
        return selected, [], debug_trace

    def _marginal_value(self, item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
        redundancy = self._redundancy_penalty(item, selected)
        coverage = self._dynamic_coverage_gain(item, selected)
        return (
            float(item["importance_score"])
            + coverage
            - redundancy
            - min(0.6, float(item["token_cost"]) / max(float(self.config.token_budget), 1.0))
        )

    def _min_packed_items_target(
        self,
        query_plan: dict[str, Any],
        query_tokens: set[str],
        candidate_count: int,
    ) -> int:
        query_type = str(query_plan.get("query_type") or "")
        intents = query_plan.get("query_intents") or {}
        target = self.config.min_packed_items
        if query_type in {"object_list", "general_description", "scene", "speech"}:
            target = max(target, 6)
        if (
            bool(intents.get("spatial_relation"))
            or query_type in {"spatial_relation", "instance_spatial_temporal"}
            or bool(query_tokens & SPATIAL_TERMS)
        ):
            target = max(target, 5)
        if (
            bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or query_type in {"before_after", "temporal", "state_change", "event", "event_localization"}
            or bool(query_tokens & TEMPORAL_TERMS)
        ):
            target = max(target, 5)
        return max(0, min(candidate_count, self.config.max_items, target))

    def _soft_token_target(
        self,
        query_plan: dict[str, Any],
        query_tokens: set[str],
        estimated_candidate_tokens: int,
        candidate_count: int,
    ) -> int:
        if candidate_count < 5:
            return 0
        query_type = str(query_plan.get("query_type") or "")
        intents = query_plan.get("query_intents") or {}
        target = 0
        if query_type in {"object_list", "general_description", "scene", "speech"}:
            target = int(min(1500, self.config.token_budget * 0.45))
        if (
            bool(intents.get("spatial_relation"))
            or query_type in {"spatial_relation", "instance_spatial_temporal"}
            or bool(query_tokens & SPATIAL_TERMS)
            or bool(intents.get("temporal_ordering"))
            or bool(intents.get("temporal_interaction"))
            or bool(intents.get("transition"))
            or query_type in {"before_after", "temporal", "state_change", "event", "event_localization"}
            or bool(query_tokens & TEMPORAL_TERMS)
        ):
            target = max(target, int(min(1800, self.config.token_budget * 0.55)))
        if target <= 0:
            return 0
        return min(target, max(0, int(estimated_candidate_tokens * 0.90)))

    def _dynamic_coverage_gain(self, item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
        if not selected:
            return 0.35
        selected_keys = set()
        for row in selected:
            selected_keys.update(self._coverage_keys(row["raw"]))
        new_keys = [key for key in self._coverage_keys(item["raw"]) if key not in selected_keys]
        return min(0.8, 0.18 * len(new_keys))

    def _redundancy_penalty(self, item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
        if not selected:
            return 0.0
        max_similarity = max(self._jaccard(item["text_tokens"], row["text_tokens"]) for row in selected)
        same_view_count = sum(1 for row in selected if row["view"] == item["view"])
        return min(1.6, max_similarity * 1.15 + 0.10 * same_view_count)

    def _packed_copy(self, item: dict[str, Any], *, remaining_chars: int, force: bool) -> dict[str, Any]:
        raw = dict(item["raw"])
        layer = item["layer"]
        text = self._text(raw)
        target_chars = len(text)
        if layer == "supporting":
            target_chars = min(target_chars, 720)
        elif layer == "context":
            target_chars = min(target_chars, 520)
        elif layer == "low_value":
            target_chars = min(target_chars, 360)
        if not force:
            target_chars = min(target_chars, max(0, remaining_chars))
        if layer != "core" and target_chars < len(text):
            prefix = self._template_prefix(raw)
            body_limit = max(60, target_chars - len(prefix) - 1)
            raw["text"] = shorten(f"{prefix} {shorten(text, body_limit)}".strip(), max(80, target_chars))
            raw["short_text"] = shorten(str(raw.get("text") or ""), 260)
            raw["packing_compressed"] = True
        raw["packing_layer"] = layer
        raw["importance_score"] = item["importance_score"]
        raw["packing_rank"] = item.get("packing_rank")
        raw["packing_reason"] = item.get("packing_reason")
        if item.get("mandatory"):
            raw["packing_mandatory"] = True
            raw["packing_mandatory_reason"] = item.get("mandatory_reason")
        return raw

    def _public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        packed = dict(item.get("packed_item") or item["raw"])
        packed["packing_layer"] = item["layer"]
        packed["importance_score"] = item["importance_score"]
        packed["packing_rank"] = item.get("packing_rank")
        packed["packing_reason"] = item.get("packing_reason")
        if item.get("mandatory"):
            packed["packing_mandatory"] = True
            packed["packing_mandatory_reason"] = item.get("mandatory_reason")
        return packed

    def _drop_record(self, item: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "stable_id": item["stable_id"],
            "id": item["raw"].get("id") or item["raw"].get("node_id"),
            "view": item["view"],
            "drop_reason": reason,
            "packing_layer": item.get("layer"),
            "importance_score": item.get("importance_score"),
            "estimated_tokens": item.get("token_cost"),
            "estimated_chars": item.get("char_cost"),
            "text_preview": shorten(item.get("text") or "", 180),
        }

    def _as_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            result = to_dict()
            return result if isinstance(result, dict) else {}
        return {}

    def _view(self, item: dict[str, Any]) -> str:
        return str(item.get("view") or item.get("type") or (item.get("record") or {}).get("view") or "unknown")

    def _text(self, item: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in (
            "text",
            "short_text",
            "evidence_text",
            "summary",
            "motion_summary",
            "relation_type",
            "event_type",
            "label",
            "role",
        ):
            value = item.get(key)
            if value:
                parts.append(str(value))
        if parts:
            return " ".join(parts)
        record = item.get("record")
        if isinstance(record, dict):
            for key in (
                "text",
                "caption",
                "transcript",
                "evidence_text",
                "summary",
                "motion_summary",
                "relation_type",
                "event_type",
                "label",
                "color",
            ):
                value = record.get(key)
                if value:
                    parts.append(str(value))
        return " ".join(parts) if parts else str(item)

    def _stable_id(self, item: dict[str, Any], index: int) -> str:
        view = self._view(item)
        for key in ("id", "node_id", "relation_id", "event_id", "track_id", "object_id", "segment_id"):
            value = item.get(key)
            if value:
                return f"{view}:{value}"
        return f"{view}:{index}:{shorten(self._text(item), 80)}"

    def _item_chars(self, item: dict[str, Any]) -> int:
        return len(self._text(item))

    def _estimate_tokens(self, item: dict[str, Any]) -> int:
        return self._estimate_tokens_from_chars(self._item_chars(item))

    def _estimate_tokens_from_chars(self, chars: int) -> int:
        return max(1, int(math.ceil(max(0, chars) / 4.0)))

    def _item_video_values(self, item: dict[str, Any]) -> set[str]:
        values = video_identity_values(item)
        provenance = item.get("provenance")
        if isinstance(provenance, dict):
            values.update(video_identity_values(provenance))
        record = item.get("record")
        if isinstance(record, dict):
            values.update(video_identity_values(record))
            record_provenance = record.get("provenance")
            if isinstance(record_provenance, dict):
                values.update(video_identity_values(record_provenance))
        return values

    def _metadata_video_values(self, metadata: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for key in ("query_video_filter", "video_filter", "video_id", "source_vid", "video_path"):
            raw = metadata.get(key)
            if isinstance(raw, (list, tuple, set)):
                for value in raw:
                    values.update(video_identity_values({"video_id": value, "source_vid": value, "video_name": value}))
            elif raw:
                values.update(video_identity_values({"video_id": raw, "source_vid": raw, "video_name": raw}))
        return values

    def _video_filter_info(self, metadata: dict[str, Any]) -> tuple[set[str], str, bool]:
        explicit_source = str(metadata.get("video_filter_source") or "").lower() == "explicit"
        values = self._metadata_video_values(metadata)
        if values and (explicit_source or any(metadata.get(key) for key in ("query_video_filter", "video_filter", "video_id", "source_vid", "video_path"))):
            return values, "explicit", True
        if str(metadata.get("video_filter_source") or "").lower() == "inferred" and values:
            return values, "inferred", False
        return set(), "disabled", False

    def _infer_anchor_video_values(self, evidence: list[dict[str, Any]]) -> set[str]:
        counts: Counter[tuple[str, ...]] = Counter()
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            values = self._item_video_values(item)
            if not values:
                continue
            key = tuple(sorted(values))
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            counts[key] += max(1, int((score + 1.0) * 1000)) + max(0, len(evidence) - index)
        if not counts:
            return set()
        return set(counts.most_common(1)[0][0])

    def _item_video_key(self, item: dict[str, Any]) -> str:
        for source in [item, item.get("provenance"), item.get("record")]:
            if not isinstance(source, dict):
                continue
            for key in ("video_id", "source_vid", "video_name", "video_path"):
                value = source.get(key)
                if value:
                    return str(value)
            provenance = source.get("provenance")
            if isinstance(provenance, dict):
                for key in ("video_id", "source_vid", "video_name", "video_path"):
                    value = provenance.get(key)
                    if value:
                        return str(value)
        values = sorted(self._item_video_values(item))
        return values[0] if values else ""

    def _packed_video_count(self, evidence: list[dict[str, Any]]) -> int:
        return len({video for item in evidence if (video := self._item_video_key(item))})

    def _dedupe_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        raw = item["raw"]
        view = item["view"]
        if view == "visual_relation":
            pair = sorted(
                [
                    str(raw.get("target_object_id") or raw.get("target_track_id") or raw.get("object_id") or ""),
                    str(raw.get("related_object_id") or raw.get("related_track_id") or ""),
                ]
            )
            return (
                view,
                tuple(sorted(item["video_values"]))[:3],
                pair[0],
                pair[1],
                raw.get("relation_type"),
                self._time_bucket(raw.get("timestamp")),
            )
        return (
            view,
            raw.get("id") or raw.get("node_id") or raw.get("segment_id") or item["stable_id"],
        )

    def _similarity_scope(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        context_views = {"scope", "segment", "caption_context"}
        if left["view"] == right["view"]:
            return left["view"] in context_views
        if left["video_values"] and right["video_values"] and not (left["video_values"] & right["video_values"]):
            return False
        return left["view"] in context_views and right["view"] in context_views

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, len(left | right))

    def _time_bucket(self, value: Any, *, bucket_seconds: float = 10.0) -> int | None:
        try:
            return int(float(value) // bucket_seconds)
        except (TypeError, ValueError):
            return None

    def _coverage_keys(self, item: dict[str, Any]) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for field in (
            "object_id",
            "target_object_id",
            "related_object_id",
            "track_id",
            "target_track_id",
            "related_track_id",
            "relation_type",
            "event_type",
            "segment_id",
        ):
            value = item.get(field)
            if value not in (None, "", []):
                keys.add((field, str(value)))
        timestamp = item.get("timestamp") or item.get("start_time")
        bucket = self._time_bucket(timestamp, bucket_seconds=15.0)
        if bucket is not None:
            keys.add(("time_bucket", str(bucket)))
        return keys

    def _semantic_count(self, evidence: list[dict[str, Any]], terms: set[str]) -> int:
        count = 0
        for item in evidence:
            text = self._text(item).lower().replace("_", " ")
            tokens = set(tokenize(text))
            if tokens & terms or any(term in text for term in terms if " " in term):
                count += 1
        return count

    def _vehicle_near_pedestrian_count(self, evidence: list[dict[str, Any]]) -> int:
        count = 0
        for item in evidence:
            text = self._text(item).lower().replace("_", " ")
            tokens = set(tokenize(text))
            has_vehicle = bool(tokens & VEHICLE_TERMS)
            has_pedestrian_context = bool(tokens & (PEDESTRIAN_TERMS | CROSSWALK_TERMS))
            has_near_relation = bool(tokens & NEAR_RELATION_TERMS)
            relation_like = self._view(item) in {"visual_relation", "nearby_object_context", "temporal_sequence"}
            if relation_like and has_vehicle and has_pedestrian_context and has_near_relation:
                count += 1
        return count

    def _yielding_supported_by_visual_relation(self, evidence: list[dict[str, Any]]) -> bool:
        for item in evidence:
            if self._view(item) not in {"visual_relation", "nearby_object_context", "visual_event", "temporal_sequence"}:
                continue
            text = self._text(item).lower().replace("_", " ")
            tokens = set(tokenize(text))
            if not (tokens & VEHICLE_TERMS and tokens & (PEDESTRIAN_TERMS | CROSSWALK_TERMS)):
                continue
            if tokens & {"yield", "yielding", "wait", "waiting", "stop", "stopped", "slowing", "slow"}:
                return True
        return False

    def _template_prefix(self, item: dict[str, Any]) -> str:
        view = self._view(item)
        video = item.get("video_id") or item.get("source_vid") or (item.get("provenance") or {}).get("video_id") or ""
        start = item.get("start_time") or item.get("timestamp")
        end = item.get("end_time")
        if start is not None and end is not None:
            time_text = f"{start}-{end}"
        elif start is not None:
            time_text = str(start)
        else:
            time_text = "unknown_time"
        video_text = f"video={video}" if video else "video=unknown"
        return f"[{view} | {time_text} | {video_text}]"
