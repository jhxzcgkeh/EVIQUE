from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .utils import tokenize, visual_relations_enabled


OBJECT_QUERY_TERMS = {
    "appear",
    "appears",
    "object",
    "objects",
    "person",
    "people",
    "vehicle",
    "vehicles",
    "visible",
    "what",
    "which",
}
ATTRIBUTE_TERMS = {
    "black",
    "blue",
    "brown",
    "dark",
    "gray",
    "green",
    "grey",
    "large",
    "light",
    "purple",
    "red",
    "small",
    "striped",
    "white",
    "yellow",
    "wearing",
}
INSTANCE_OBJECT_TERMS = {
    "animal",
    "bag",
    "ball",
    "bicycle",
    "bus",
    "car",
    "cat",
    "chair",
    "defender",
    "dog",
    "item",
    "motorcycle",
    "object",
    "performer",
    "person",
    "people",
    "player",
    "table",
    "truck",
    "vehicle",
}
INSTANCE_SPATIAL_TERMS = {
    "above",
    "around",
    "behind",
    "below",
    "beside",
    "close",
    "closest",
    "front",
    "near",
    "nearby",
    "nearest",
    "next",
}
INSTANCE_EVENT_TERMS = {
    "appear",
    "approach",
    "before",
    "cross",
    "during",
    "enter",
    "entered",
    "enters",
    "go",
    "happen",
    "happened",
    "happens",
    "leave",
    "move",
    "moved",
    "moves",
    "moving",
    "pass",
    "run",
    "start",
    "stop",
    "turn",
    "walk",
    "walks",
}
INSTANCE_REGION_TERMS = {
    "area",
    "center",
    "entrance",
    "exit",
    "field",
    "middle",
    "penalty",
    "region",
    "road",
    "room",
    "side",
    "stage",
    "table",
    "yard",
}
TRAJECTORY_TERMS = {
    "across",
    "approach",
    "approaching",
    "direction",
    "enter",
    "entered",
    "entering",
    "exit",
    "follow",
    "leave",
    "leaving",
    "left",
    "move",
    "moving",
    "path",
    "route",
    "track",
    "trajectory",
    "travel",
    "traveling",
    "travelling",
    "turn",
    "turning",
}
TEMPORAL_TERMS = {"after", "before", "during", "first", "last", "later", "next", "then", "when", "while"}
INTERACTION_TERMS = {"beside", "between", "cross", "interact", "near", "nearby", "together", "with"}
STATE_TERMS = {
    "change",
    "congested",
    "congestion",
    "increase",
    "queue",
    "queued",
    "queueing",
    "queuing",
    "slow",
    "slowing",
    "start",
    "state",
    "stop",
    "stopped",
    "wait",
    "waiting",
}
SPEECH_TERMS = {"say", "said", "speak", "speech", "talk", "transcript"}
SCENE_TERMS = {"background", "describe", "description", "general", "happen", "happening", "overview", "place", "scene", "setting", "where"}
EVENT_TERMS = {"activity", "activities", "event", "events"}
BEFORE_AFTER_TERMS = {"after", "before", "earlier", "later", "next", "then"}
DIRECTION_TERMS = {"left", "right", "north", "south", "east", "west", "straight", "forward", "backward", "toward", "away", "through"}
QUERY_INTENT_KEYS = [
    "spatial_relation",
    "temporal_trajectory",
    "temporal_ordering",
    "temporal_interaction",
    "transition",
    "multi_object_interaction",
    "event_localization",
    "evidence_grounded",
]
SPATIAL_RELATION_TERMS = {
    "adjacent", "around", "behind", "beside", "close", "closest", "front", "left",
    "near", "nearby", "nearest", "neighboring", "right", "surrounding",
}
SPATIAL_RELATION_PHRASES = {
    "close to", "in front of", "left of", "next to", "relative position", "right of", "side by side",
}
TEMPORAL_TRAJECTORY_TERMS = {
    "after", "before", "cross", "crosses", "direction", "enter", "entered", "enters",
    "exit", "exits", "leave", "leaves", "move", "moves", "moving", "next",
    "subsequent", "then", "trajectory", "turn", "turns",
}
TEMPORAL_TRAJECTORY_PHRASES = {"goes next", "later position", "pass through"}
TEMPORAL_ORDERING_TERMS = {
    "after",
    "afterward",
    "afterwards",
    "before",
    "earlier",
    "first",
    "following",
    "immediate",
    "immediately",
    "later",
    "next",
    "order",
    "preceding",
    "previous",
    "prior",
    "sequence",
    "subsequent",
    "then",
}
TEMPORAL_ORDERING_PHRASES = {
    "before and after",
    "happen next",
    "happens next",
    "immediately after",
    "immediately afterward",
    "immediately afterwards",
    "in sequence",
    "one after another",
    "right after",
    "short period",
    "what happened next",
    "what happens next",
}
TEMPORAL_STATE_ACTION_TERMS = {
    "arrive",
    "arrived",
    "arrives",
    "begin",
    "began",
    "begins",
    "depart",
    "departed",
    "departs",
    "enter",
    "entered",
    "entering",
    "enters",
    "exit",
    "exited",
    "exiting",
    "exits",
    "go",
    "goes",
    "leave",
    "leaves",
    "leaving",
    "move",
    "moved",
    "moves",
    "moving",
    "proceed",
    "proceeded",
    "proceeding",
    "proceeds",
    "remain",
    "remained",
    "remaining",
    "remains",
    "start",
    "started",
    "starting",
    "starts",
    "stay",
    "stayed",
    "stays",
    "stop",
    "stopped",
    "stopping",
    "stops",
    "wait",
    "waited",
    "waiting",
    "waits",
}
TEMPORAL_STATE_ACTION_PHRASES = {
    "begin moving",
    "begins moving",
    "remain stopped",
    "remains stopped",
    "start moving",
    "starts moving",
    "waiting before",
}
TRANSITION_TERMS = {
    "change",
    "changed",
    "changes",
    "changing",
    "shift",
    "shifted",
    "shifting",
    "shifts",
    "switch",
    "switched",
    "switches",
    "switching",
    "transition",
    "transitions",
}
TRANSITION_PHRASES = {
    "changes from",
    "light changes",
    "light turns",
    "red to green",
    "red-to-green",
    "signal changes",
    "state changes",
    "turns from",
}
MULTI_OBJECT_INTERACTION_TERMS = {
    "cluster", "clustered", "constrain", "follow", "following", "follows", "influence",
    "interact", "interaction", "multiple", "waiting",
}
MULTI_OBJECT_INTERACTION_PHRASES = {
    "beside another", "group of", "near another", "multiple vehicles", "pass close", "wait behind",
}
EVENT_LOCALIZATION_TERMS = {
    "area", "center", "clustered", "congestion", "event", "localized", "middle",
    "region", "stopped", "stopping", "where", "when",
}
EVENT_LOCALIZATION_PHRASES = {"center region", "localized event", "middle region", "slow traffic"}
EVIDENCE_GROUNDED_TERMS = {
    "clearly", "concluded", "determine", "evidence", "insufficient", "unsupported", "uncertain", "visible",
}
EVIDENCE_GROUNDED_PHRASES = {
    "based only on visible evidence", "cannot determine", "clearly visible",
    "insufficient evidence", "what can and cannot be concluded",
}
GLOBAL_SUMMARY_PHRASES = {
    "general scene", "overall description", "summarize whole video", "summarize the whole video",
}
STOPWORDS = {
    "and",
    "are",
    "did",
    "does",
    "has",
    "have",
    "how",
    "that",
    "the",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
}


@dataclass
class QueryPlan:
    query: str
    query_type: str
    selected_views: list[str]
    dependency_order: list[str]
    required_views: list[str]
    optional_views: list[str]
    evidence_roles: list[str]
    query_terms: list[str]
    view_weights: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float | int | str] = field(default_factory=dict)
    guidance: str = ""
    query_intents: dict[str, bool] = field(default_factory=dict)
    route_reason: str = ""
    visual_trigger_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QueryPlanner:
    """A compact implementation of EVIQUE query-to-evidence planning.

    The planner keeps the paper's planning constraints explicit: view coverage,
    dependency ordering, provenance availability, candidate cardinality, access
    cost, and downstream token budget.
    """

    def __init__(self, *, max_evidence: int = 18, token_budget: int = 12000):
        self.max_evidence = max_evidence
        self.token_budget = token_budget

    def plan(self, query: str) -> QueryPlan:
        raw_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query.lower()))
        keep_terms = {
            term
            for term in raw_terms
            if len(term) > 2 and term not in STOPWORDS
        }
        singular_terms = {term[:-1] for term in keep_terms if term.endswith("s") and len(term) > 4}
        terms = set(tokenize(query)) | keep_terms | singular_terms
        query_intents = self._detect_query_intents(terms, query)
        visual_chain_priority = self._visual_chain_priority(query_intents, terms, query)
        query_type = self._classify(terms, query, query_intents)
        required, optional, roles = self._views_and_roles(query_type, terms, visual_chain_priority)
        visual_relation_view_enabled = visual_relations_enabled()
        if not visual_relation_view_enabled:
            required = self._without_visual_relation(required)
            optional = self._without_visual_relation(optional)
        selected = self._dedupe(required + optional)
        dependency_order = self._dependency_order(selected)
        view_weights = self._view_weights(query_type)
        if visual_chain_priority:
            visual_chain_views = ["visual_object", "visual_track", "visual_event"]
            if visual_relation_view_enabled:
                visual_chain_views.insert(2, "visual_relation")
            for view in visual_chain_views:
                view_weights[view] = max(1.0, view_weights.get(view, 0.0))
        visual_trigger_reason = self._visual_trigger_reason(query_intents, terms, query, visual_chain_priority)
        route_reason = (
            f"query_type={query_type}; "
            f"visual_chain_priority={'on' if visual_chain_priority else 'off'}; "
            f"{visual_trigger_reason}"
        )
        return QueryPlan(
            query=query,
            query_type=query_type,
            selected_views=selected,
            dependency_order=dependency_order,
            required_views=required,
            optional_views=optional,
            evidence_roles=roles,
            query_terms=sorted(terms),
            view_weights=view_weights,
            constraints={
                "coverage": 1,
                "dependency": 1,
                "provenance": 1,
                "max_evidence": self.max_evidence,
                "token_budget": self.token_budget,
                "optional_score_threshold": 0.15,
                "max_evidence_chars": min(self.token_budget, 8000),
                "max_item_chars": 900,
                "max_segment_items": 6,
                "max_track_items": 3,
                "max_event_items": 3,
                "visual_chain_priority": int(visual_chain_priority),
                "access_cost_policy": "query-dependent required views with optional evidence expansion and minimal evidence packaging",
            },
            guidance=self._guidance(query_type),
            query_intents=query_intents,
            route_reason=route_reason,
            visual_trigger_reason=visual_trigger_reason,
        )

    def _classify(self, terms: set[str], query: str, query_intents: dict[str, bool]) -> str:
        lowered = query.lower()
        visual_context = bool(terms & (INSTANCE_OBJECT_TERMS | INSTANCE_EVENT_TERMS | TRAJECTORY_TERMS | DIRECTION_TERMS))
        if self._is_global_summary(terms, lowered) and not any(query_intents.values()):
            return "general_description"
        if self._is_instance_spatial_temporal(terms, lowered):
            return "instance_spatial_temporal"
        if (
            query_intents.get("temporal_interaction")
            or query_intents.get("temporal_ordering")
            or query_intents.get("transition")
        ) and (
            visual_context
            or query_intents.get("temporal_trajectory")
            or query_intents.get("event_localization")
            or bool(terms & (EVENT_TERMS | STATE_TERMS))
        ):
            return "temporal"
        if query_intents.get("spatial_relation") and (
            query_intents.get("temporal_trajectory")
            or query_intents.get("multi_object_interaction")
            or visual_context
        ):
            return "instance_spatial_temporal"
        if query_intents.get("multi_object_interaction"):
            return "interaction"
        if query_intents.get("temporal_trajectory"):
            return "trajectory"
        if query_intents.get("spatial_relation"):
            return "spatial_relation"
        if query_intents.get("event_localization"):
            return "event_localization"
        if terms & SPEECH_TERMS:
            return "speech"
        if terms & STATE_TERMS:
            return "state_change"
        if terms & BEFORE_AFTER_TERMS or terms & TEMPORAL_TERMS:
            return "before_after"
        if terms & TRAJECTORY_TERMS or terms & DIRECTION_TERMS:
            return "trajectory"
        if terms & INTERACTION_TERMS:
            return "interaction"
        if terms & EVENT_TERMS or {"happen", "happening"} & terms or "event" in lowered or "events" in lowered or "activity" in lowered:
            return "event"
        if terms & SCENE_TERMS:
            return "general_description"
        if ("object" in terms or "objects" in terms or "visible" in terms) and (
            "which" in lowered or "what" in lowered or "list" in lowered
        ):
            return "object_list"
        if terms & OBJECT_QUERY_TERMS:
            return "object_grounding"
        return "default"

    def _contains_phrase(self, lowered_query: str, phrases: set[str]) -> bool:
        return any(phrase in lowered_query for phrase in phrases)

    def _detect_query_intents(self, terms: set[str], query: str) -> dict[str, bool]:
        lowered = query.lower()
        context_terms = INSTANCE_OBJECT_TERMS | INSTANCE_EVENT_TERMS | TRAJECTORY_TERMS | STATE_TERMS | EVENT_TERMS | INSTANCE_REGION_TERMS
        object_or_motion_context = bool(terms & context_terms)
        spatial = bool(terms & SPATIAL_RELATION_TERMS) or self._contains_phrase(lowered, SPATIAL_RELATION_PHRASES)
        temporal = bool(terms & (TEMPORAL_TRAJECTORY_TERMS | TRAJECTORY_TERMS)) or self._contains_phrase(lowered, TEMPORAL_TRAJECTORY_PHRASES)
        temporal_ordering = bool(terms & TEMPORAL_ORDERING_TERMS) or self._contains_phrase(lowered, TEMPORAL_ORDERING_PHRASES)
        temporal_state_action = bool(terms & TEMPORAL_STATE_ACTION_TERMS) or self._contains_phrase(lowered, TEMPORAL_STATE_ACTION_PHRASES)
        transition = bool(terms & TRANSITION_TERMS) or self._contains_phrase(lowered, TRANSITION_PHRASES)
        interaction = (
            bool(terms & MULTI_OBJECT_INTERACTION_TERMS)
            or self._contains_phrase(lowered, MULTI_OBJECT_INTERACTION_PHRASES)
            or (spatial and bool({"another", "other", "multiple"} & terms))
            or (temporal_ordering and bool({"another", "other", "multiple", "several"} & terms))
        )
        temporal_interaction = (
            (temporal_ordering or transition)
            and (interaction or object_or_motion_context or temporal_state_action)
        ) or self._contains_phrase(lowered, {"before and after", "one after another"})
        event_localization = (
            self._contains_phrase(lowered, EVENT_LOCALIZATION_PHRASES)
            or bool(terms & (EVENT_LOCALIZATION_TERMS - {"where", "when"}))
            or (bool({"where", "when"} & terms) and object_or_motion_context)
        )
        evidence_grounded = bool(terms & EVIDENCE_GROUNDED_TERMS) or self._contains_phrase(lowered, EVIDENCE_GROUNDED_PHRASES)
        return {
            "spatial_relation": spatial,
            "temporal_trajectory": temporal,
            "temporal_ordering": temporal_ordering,
            "temporal_interaction": temporal_interaction,
            "transition": transition,
            "multi_object_interaction": interaction,
            "event_localization": event_localization,
            "evidence_grounded": evidence_grounded,
        }

    def _visual_chain_priority(self, query_intents: dict[str, bool], terms: set[str], query: str) -> bool:
        lowered = query.lower()
        if self._is_global_summary(terms, lowered) and not any(
            query_intents.get(key, False) for key in ["spatial_relation", "temporal_trajectory", "multi_object_interaction", "event_localization"]
        ):
            return False
        return any(query_intents.get(key, False) for key in QUERY_INTENT_KEYS)

    def _visual_trigger_reason(
        self,
        query_intents: dict[str, bool],
        terms: set[str],
        query: str,
        visual_chain_priority: bool,
    ) -> str:
        active = [key for key in QUERY_INTENT_KEYS if query_intents.get(key)]
        if visual_chain_priority:
            return "triggered_by_query_intents=" + ",".join(active)
        if self._is_global_summary(terms, query.lower()):
            return "not_triggered_global_summary_query"
        return "not_triggered_no_spatio_temporal_visual_intent"

    def _is_global_summary(self, terms: set[str], lowered_query: str) -> bool:
        return self._contains_phrase(lowered_query, GLOBAL_SUMMARY_PHRASES) or (
            bool({"summarize", "summary", "overview"} & terms) and bool({"whole", "overall", "general"} & terms)
        )
    def _is_instance_spatial_temporal(self, terms: set[str], lowered_query: str) -> bool:
        phrase_spatial = any(
            phrase in lowered_query
            for phrase in ("next to", "close to", "left of", "right of", "in front of")
        )
        passive_event = any(
            phrase in lowered_query
            for phrase in ("is moved", "was moved", "being moved", "gets moved", "got moved")
        )
        has_object_requirement = bool(terms & INSTANCE_OBJECT_TERMS)
        has_spatial_requirement = bool(terms & INSTANCE_SPATIAL_TERMS) or phrase_spatial
        has_temporal_anchor = (
            "when" in lowered_query
            or "after" in lowered_query
            or "before" in lowered_query
            or "during" in lowered_query
            or passive_event
            or bool(terms & INSTANCE_EVENT_TERMS)
        )
        has_motion_or_event_requirement = passive_event or bool(terms & (INSTANCE_EVENT_TERMS | TRAJECTORY_TERMS | DIRECTION_TERMS))
        has_region_or_attribute_context = bool(terms & (ATTRIBUTE_TERMS | INSTANCE_REGION_TERMS)) or "onto" in lowered_query
        return (
            has_object_requirement
            and has_spatial_requirement
            and has_temporal_anchor
            and has_motion_or_event_requirement
            and (has_region_or_attribute_context or len(terms & INSTANCE_OBJECT_TERMS) >= 2)
        )

    def _views_and_roles(self, query_type: str, terms: set[str], visual_chain_priority: bool) -> tuple[list[str], list[str], list[str]]:
        if query_type == "instance_spatial_temporal":
            required = ["visual_object", "visual_track", "visual_relation", "visual_event"]
            optional = ["scope", "target", "track", "event"]
            roles = ["attribute_grounding", "event_binding", "spatial_relation", "motion_evidence"]
        elif query_type in {"scene", "general_description"}:
            required, optional, roles = ["scope"], ["target"], ["scene_summary", "key_segments"]
        elif query_type in {"object_list", "object_grounding"}:
            required, optional, roles = ["target", "scope"], [], ["target_presence", "supporting_segments"]
        elif query_type == "spatial_relation":
            required, optional, roles = ["target", "scope"], ["track", "event"], ["spatial_relation", "target_presence", "supporting_segments"]
        elif query_type in {"trajectory", "interaction"}:
            required, optional, roles = ["target", "scope"], ["track"], ["target_presence", "motion_evidence", "supporting_segments"]
        elif query_type in {"before_after", "temporal"}:
            required, optional, roles = ["scope", "target"], ["track", "event"], ["anchor_segment", "before_context", "after_context"]
        elif query_type == "state_change":
            required, optional, roles = ["event", "scope"], ["track", "target"], ["state_event", "supporting_segments", "target_context"]
        elif query_type == "speech":
            required, optional, roles = ["scope"], [], ["transcript_excerpt"]
        elif query_type in {"event", "event_localization"}:
            required, optional, roles = ["event", "scope"], ["target", "track"], ["event_summary", "supporting_segments"]
        else:
            required, optional, roles = ["scope"], ["target", "event"], ["general_evidence", "supporting_segments"]

        if visual_chain_priority and query_type not in {"general_description", "scene", "speech"}:
            required = self._dedupe(["visual_object", "visual_track", "visual_relation", "visual_event"] + required)
            roles = self._dedupe(["attribute_grounding", "spatial_relation", "motion_evidence", "event_binding"] + roles)
        return required, optional, roles
    def _dependency_order(self, views: list[str]) -> list[str]:
        ordered = []
        for view in ["visual_object", "visual_track", "visual_event", "visual_relation", "scope", "target", "track", "event"]:
            if view in views:
                ordered.append(view)
        return ordered

    def _guidance(self, query_type: str) -> str:
        return {
            "instance_spatial_temporal": "Build a provenance-preserving visual evidence chain: target attributes, anchor event, spatial relation, related object, then compact subsequent motion.",
            "object_list": "Use Target and Scope evidence only; do not expand tracks unless the query asks for motion.",
            "object_grounding": "Ground object presence with target annotations and nearby segment descriptions.",
            "trajectory": "Use target presence and segment captions as anchors; add compact track summaries only when relevant.",
            "interaction": "Use target co-presence and compact motion evidence; avoid full-track expansion.",
            "before_after": "Anchor the answer in key segments, then add only adjacent before/after context and relevant state or motion summaries.",
            "temporal": "Anchor the answer in key segments or events, then add bounded before/focal/after temporal context without inferring causality.",
            "state_change": "Prioritize event/state evidence and segment support; add target or track context only when relevant.",
            "speech": "Use transcript-bearing Scope evidence and keep visual claims grounded.",
            "scene": "Use concise Scope evidence and scene tags.",
            "general_description": "Use concise Scope evidence and only a small amount of target context.",
            "event": "Use event windows plus supporting segments when the query asks about events.",
            "default": "Use only provenance-backed minimal evidence.",
        }.get(query_type, "Use only provenance-backed evidence.")

    def _view_weights(self, query_type: str) -> dict[str, float]:
        weights = {
            "instance_spatial_temporal": {
                "visual_object": 1.0,
                "visual_track": 1.0,
                "visual_relation": 1.0,
                "visual_event": 1.0,
                "scope": 0.55,
                "target": 0.45,
                "track": 0.45,
                "event": 0.45,
                "coverage": 0.2,
            },
            "object_list": {"scope": 0.85, "target": 1.0, "track": 0.25, "event": 0.2, "coverage": 0.9},
            "object_grounding": {"scope": 0.9, "target": 1.0, "track": 0.25, "event": 0.2, "coverage": 0.5},
            "trajectory": {"scope": 0.8, "target": 0.95, "track": 1.0, "event": 0.55, "coverage": 0.3},
            "interaction": {"scope": 0.75, "target": 1.0, "track": 0.9, "event": 0.45, "coverage": 0.3},
            "before_after": {"scope": 1.0, "target": 0.8, "track": 0.75, "event": 0.8, "coverage": 0.3},
            "temporal": {"scope": 1.0, "target": 0.8, "track": 0.7, "event": 0.9, "coverage": 0.3},
            "state_change": {"scope": 0.9, "target": 0.65, "track": 0.75, "event": 1.0, "coverage": 0.3},
            "speech": {"scope": 1.0, "target": 0.25, "track": 0.2, "event": 0.4, "coverage": 0.2},
            "scene": {"scope": 1.0, "target": 0.5, "track": 0.2, "event": 0.35, "coverage": 0.2},
            "general_description": {"scope": 1.0, "target": 0.55, "track": 0.2, "event": 0.35, "coverage": 0.2},
            "event": {"scope": 0.85, "target": 0.45, "track": 0.45, "event": 1.0, "coverage": 0.2},
            "default": {"scope": 1.0, "target": 0.6, "track": 0.3, "event": 0.5, "coverage": 0.2},
        }
        return weights.get(query_type, weights["default"])

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _without_visual_relation(self, values: list[str]) -> list[str]:
        return [value for value in values if value != "visual_relation"]
