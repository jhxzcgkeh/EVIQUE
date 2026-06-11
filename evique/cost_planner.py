from __future__ import annotations

import math
import os
from typing import Any

from .utils import tokenize, visual_relations_enabled


PLANNER_VERSION = "v5_cost_based"
PLANNABLE_VIEWS = [
    "scope",
    "target",
    "track",
    "event",
    "adaptive_event",
    "visual_object",
    "visual_track",
    "visual_event",
    "visual_relation",
]


def get_cost_planner_config() -> dict[str, Any]:
    return {
        "enabled": _env_bool("EVIQUE_COST_PLANNER", True),
        "debug": _env_bool("EVIQUE_COST_PLANNER_DEBUG", False),
        "max_views": _env_int("EVIQUE_COST_PLANNER_MAX_VIEWS", 4),
        "min_confidence": _env_float("EVIQUE_COST_PLANNER_MIN_CONFIDENCE", 0.65),
        "max_rows_total": _env_int("EVIQUE_COST_PLANNER_MAX_ROWS_TOTAL", 16),
    }


class CostBasedViewPlanner:
    def __init__(self, view_stats: dict[str, Any] | None = None, *, config: dict[str, Any] | None = None):
        self.view_stats = view_stats or {}
        self.config = dict(config or get_cost_planner_config())

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    def plan(
        self,
        query: str,
        base_plan: Any,
        *,
        video_filter: set[str] | None = None,
    ) -> dict[str, Any]:
        query_terms = set(getattr(base_plan, "query_terms", []) or tokenize(query))
        active_intents = [
            key for key, value in (getattr(base_plan, "query_intents", {}) or {}).items() if bool(value)
        ]
        estimates: dict[str, dict[str, Any]] = {}
        available: list[tuple[float, str]] = []
        plannable_views = [
            view for view in PLANNABLE_VIEWS
            if view != "visual_relation" or visual_relations_enabled()
        ]
        for view in plannable_views:
            stats = self._stats(view)
            row_count = int(stats.get("row_count") or 0)
            cost = self.estimated_cost(view, stats)
            benefit = self.expected_benefit(
                view,
                stats,
                query_terms=query_terms,
                query_type=str(getattr(base_plan, "query_type", "default")),
                query_intents=getattr(base_plan, "query_intents", {}) or {},
                video_filter=video_filter,
            )
            score = benefit / (cost + 0.05)
            estimate = {
                "row_count": row_count,
                "estimated_cost": round(cost, 6),
                "expected_benefit": round(benefit, 6),
                "planner_score": round(score, 6),
                "available": row_count > 0,
            }
            estimates[view] = estimate
            if row_count > 0:
                available.append((score, view))

        available.sort(key=lambda item: item[0], reverse=True)
        max_views = max(1, int(self.config.get("max_views") or 4))
        view_order = [view for _, view in available[:max_views]]
        if not view_order:
            view_order = [view for view in getattr(base_plan, "selected_views", []) or ["scope"] if view in plannable_views]
        anchor_view = view_order[0] if view_order else "scope"
        max_rows_per_view = self._row_budgets(view_order)
        reason = self._reason(
            query_type=str(getattr(base_plan, "query_type", "default")),
            active_intents=active_intents,
            anchor_view=anchor_view,
            view_order=view_order,
        )
        return {
            "planner_version": PLANNER_VERSION,
            "enabled": self.enabled,
            "query_intents": active_intents,
            "query_type": str(getattr(base_plan, "query_type", "default")),
            "anchor_view": anchor_view,
            "view_order": view_order,
            "max_rows_per_view": max_rows_per_view,
            "max_rows_total": int(self.config.get("max_rows_total") or 16),
            "min_confidence": float(self.config.get("min_confidence") or 0.65),
            "stop_condition": "stop_when_high_confidence_evidence_found",
            "estimated_costs": estimates,
            "reason": reason,
            "debug_trace": self._debug_trace(estimates, view_order) if self.config.get("debug") else [],
        }

    def estimated_cost(self, view: str, stats: dict[str, Any]) -> float:
        row_count = float(stats.get("row_count") or 0.0)
        avg_tokens = float(stats.get("avg_text_tokens") or 0.0)
        scan_cost = min(3.0, math.log10(row_count + 1.0))
        token_cost = min(2.0, avg_tokens / 180.0)
        expansion_cost = {
            "scope": 0.15,
            "target": 0.20,
            "track": 0.35,
            "event": 0.30,
            "adaptive_event": 0.25,
            "visual_object": 0.25,
            "visual_track": 0.40,
            "visual_event": 0.35,
            "visual_relation": 0.55,
        }.get(view, 0.30)
        noise_risk = {
            "scope": 0.55,
            "target": 0.20,
            "track": 0.30,
            "event": 0.35,
            "adaptive_event": 0.22,
            "visual_object": 0.18,
            "visual_track": 0.25,
            "visual_event": 0.25,
            "visual_relation": 0.45,
        }.get(view, 0.30)
        return scan_cost + token_cost + expansion_cost + noise_risk

    def expected_benefit(
        self,
        view: str,
        stats: dict[str, Any],
        *,
        query_terms: set[str],
        query_type: str,
        query_intents: dict[str, bool],
        video_filter: set[str] | None = None,
    ) -> float:
        return (
            self._intent_match_score(view, query_type, query_intents)
            + self._keyword_selectivity_score(view, stats, query_terms)
            + self._modality_match_score(view, query_terms, query_intents)
            + (0.25 if video_filter else 0.0)
        )

    def _stats(self, view: str) -> dict[str, Any]:
        stats = self.view_stats.get(view)
        return stats if isinstance(stats, dict) else {}

    def _row_budgets(self, view_order: list[str]) -> dict[str, int]:
        total = max(1, int(self.config.get("max_rows_total") or 16))
        defaults = {
            "scope": 4,
            "target": 6,
            "track": 4,
            "event": 4,
            "adaptive_event": 5,
            "visual_object": 6,
            "visual_track": 4,
            "visual_event": 4,
            "visual_relation": 6,
        }
        budgets: dict[str, int] = {}
        remaining = total
        for idx, view in enumerate(view_order):
            if remaining <= 0:
                budgets[view] = 0
                continue
            preferred = defaults.get(view, 3)
            if idx == 0:
                preferred = max(preferred, min(6, total // 2 + 1))
            budget = min(preferred, remaining)
            budgets[view] = budget
            remaining -= budget
        return budgets

    def _intent_match_score(self, view: str, query_type: str, query_intents: dict[str, bool]) -> float:
        base_by_type = {
            "general_description": {"scope": 1.2, "event": 0.45, "adaptive_event": 0.55, "target": 0.35},
            "scene": {"scope": 1.2, "event": 0.45, "adaptive_event": 0.55, "target": 0.35},
            "speech": {"scope": 1.3},
            "object_list": {"target": 1.15, "visual_object": 1.0, "scope": 0.55},
            "object_grounding": {"target": 1.15, "visual_object": 1.0, "scope": 0.55},
            "trajectory": {"visual_track": 1.15, "track": 1.0, "adaptive_event": 0.75},
            "interaction": {"visual_relation": 1.1, "visual_track": 0.9, "track": 0.75, "adaptive_event": 0.65},
            "spatial_relation": {"visual_relation": 1.25, "visual_object": 0.75, "visual_track": 0.65, "target": 0.65},
            "before_after": {"adaptive_event": 1.2, "event": 0.85, "track": 0.65, "scope": 0.65},
            "temporal": {"adaptive_event": 1.25, "event": 0.85, "visual_event": 0.75, "track": 0.65},
            "state_change": {"adaptive_event": 1.15, "event": 1.0, "visual_event": 0.75, "track": 0.55},
            "event": {"adaptive_event": 1.2, "event": 1.0, "scope": 0.55},
            "event_localization": {"adaptive_event": 1.2, "event": 0.9, "visual_event": 0.7, "scope": 0.5},
            "instance_spatial_temporal": {
                "visual_relation": 1.2,
                "visual_track": 1.05,
                "adaptive_event": 0.95,
                "visual_object": 0.85,
                "visual_event": 0.8,
            },
        }
        score = base_by_type.get(query_type, {"scope": 0.75, "adaptive_event": 0.65, "target": 0.55}).get(view, 0.25)
        if query_intents.get("spatial_relation") and view == "visual_relation":
            score += 0.45
        if (query_intents.get("temporal_ordering") or query_intents.get("temporal_interaction")) and view == "adaptive_event":
            score += 0.45
        if query_intents.get("temporal_trajectory") and view in {"visual_track", "track"}:
            score += 0.35
        if query_intents.get("transition") and view in {"adaptive_event", "event", "visual_event"}:
            score += 0.30
        return score

    def _keyword_selectivity_score(self, view: str, stats: dict[str, Any], query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        candidate_counts: dict[str, int] = {}
        for key in (
            "label_counts",
            "event_type_counts",
            "relation_type_counts",
            "dominant_signal_counts",
            "segmentation_mode_counts",
        ):
            values = stats.get(key)
            if isinstance(values, dict):
                candidate_counts.update({str(label).lower(): int(count) for label, count in values.items() if count})
        if not candidate_counts:
            return 0.05 if view in {"scope", "event", "adaptive_event"} else 0.0
        row_count = max(1.0, float(stats.get("row_count") or 1.0))
        score = 0.0
        for term in query_terms:
            for label, count in candidate_counts.items():
                label_terms = set(tokenize(label)) | {label}
                if term in label_terms:
                    rarity = 1.0 - min(0.95, float(count) / row_count)
                    score += 0.25 + rarity
        return min(1.2, score)

    def _modality_match_score(self, view: str, query_terms: set[str], query_intents: dict[str, bool]) -> float:
        visual_terms = {"visible", "look", "color", "left", "right", "near", "closest", "move", "moving", "object"}
        speech_terms = {"say", "said", "speech", "talk", "dialogue", "transcript"}
        if query_terms & speech_terms:
            return 0.65 if view == "scope" else 0.05
        if query_intents.get("spatial_relation"):
            return 0.55 if view in {"visual_relation", "visual_object", "visual_track"} else 0.15
        if query_intents.get("temporal_trajectory") or query_intents.get("temporal_ordering"):
            return 0.55 if view in {"adaptive_event", "visual_track", "track", "event"} else 0.15
        if query_terms & visual_terms:
            return 0.45 if view.startswith("visual_") or view in {"target", "track", "adaptive_event"} else 0.15
        return 0.25 if view in {"scope", "event", "adaptive_event"} else 0.15

    def _reason(self, *, query_type: str, active_intents: list[str], anchor_view: str, view_order: list[str]) -> str:
        intent_text = ", ".join(active_intents) if active_intents else query_type
        order_text = " -> ".join(view_order)
        return f"{intent_text} query; {anchor_view} has the best benefit/cost anchor score; expansion order is {order_text}"

    def _debug_trace(self, estimates: dict[str, dict[str, Any]], view_order: list[str]) -> list[dict[str, Any]]:
        rows = []
        for view, estimate in estimates.items():
            row = dict(estimate)
            row["view"] = view
            row["selected"] = view in view_order
            rows.append(row)
        rows.sort(key=lambda row: float(row.get("planner_score", 0.0)), reverse=True)
        return rows


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


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
