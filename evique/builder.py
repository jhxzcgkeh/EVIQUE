from __future__ import annotations

import re
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import (
    EventViewRecord,
    EvidenceNode,
    EvidenceRelation,
    Provenance,
    ScopeViewRecord,
    TargetViewRecord,
    TrackViewRecord,
)
from .utils import (
    directory_size_bytes,
    extract_caption_and_transcript,
    format_seconds,
    parse_time_range,
    read_json,
    read_jsonl,
    read_visual_relations,
    remove_visual_relation_files,
    shorten,
    tokenize,
    unified_segment_text,
    visual_relation_file_metadata,
    visual_relations_enabled,
    write_json,
    write_jsonl,
)
from .event_segmenter import (
    adaptive_event_stats,
    build_adaptive_events,
    get_event_segmentation_config,
)
from .cost_planner import get_cost_planner_config
from .view_stats import VIEW_STATS_FILE, write_view_stats
from .visual_builder import build_visual_evique
from .visual_compactor import merge_visual_compact_stats, visual_compact_metadata
from .video_identity import (
    EVIQUE_VERSION,
    EVIQUE_VERSION_LABEL,
    collect_video_identities,
    make_video_identity,
    provenance_video_fields,
    resolve_video_name_for_path,
)


GENERIC_NON_TARGET_BLACKLIST = {
    "sequence",
    "sequences",
    "capture",
    "captures",
    "frame",
    "frames",
    "footage",
    "video",
    "clip",
    "clips",
    "image",
    "images",
    "view",
    "views",
    "scene",
    "scenes",
    "shot",
    "shots",
    "camera",
    "timestamp",
    "time",
    "moment",
    "moments",
    "series",
    "depicts",
    "depict",
    "displays",
    "display",
    "shows",
    "show",
    "approximately",
    "throughout",
    "still",
    "static",
    "continuous",
    "initially",
    "vantage",
    "angle",
    "high-angle",
    "aerial",
    "daylight",
    "weather",
    "conditions",
}

GENERAL_TARGET_WHITELIST = {
    "person", "people", "man", "woman", "child", "children",
    "pedestrian", "crowd", "group",
    "performer", "dancer", "singer", "speaker", "player", "athlete",
    "driver", "cyclist",
    "car", "vehicle", "truck", "bus", "van", "suv", "taxi",
    "motorcycle", "motorbike", "bike", "bicycle",
    "animal", "dog", "cat", "bird", "horse", "cow", "sheep",
    "ball", "instrument", "guitar", "microphone", "phone", "bag",
    "flag", "sign", "banner", "table", "chair", "screen",
    "road", "lane", "intersection", "crosswalk", "sidewalk",
    "traffic light", "traffic signal",
    "building", "tree", "river", "mountain", "stage", "field",
    "court", "room", "door", "window",
}

MOVABLE_OR_ACTOR_LABELS = {
    "person", "people", "man", "woman", "child", "children",
    "pedestrian", "performer", "dancer", "singer", "speaker",
    "player", "athlete", "driver", "cyclist",
    "car", "vehicle", "truck", "bus", "van", "suv", "taxi",
    "motorcycle", "motorbike", "bike", "bicycle",
    "animal", "dog", "cat", "bird", "horse", "cow", "sheep",
    "ball",
}

MOTION_KEYWORDS = {
    "enter", "entering", "entered",
    "leave", "leaving", "left",
    "move", "moving", "moves",
    "turn", "turning", "turned",
    "cross", "crossing", "crossed",
    "approach", "approaching",
    "pass", "passing",
    "walk", "walking",
    "run", "running",
    "jump", "jumping",
    "dance", "dancing",
    "play", "playing",
    "travel", "travelling", "traveling",
}

STATE_KEYWORDS = {
    "stop", "stopped", "stopping",
    "wait", "waiting",
    "queue", "queued", "queuing", "queueing",
    "slow", "slowing", "slowed",
    "congestion", "congested",
    "park", "parked",
    "stand", "standing",
    "sit", "sitting",
    "hold", "holding",
}

DIRECTION_KEYWORDS = {
    "left", "right", "north", "south", "east", "west",
    "straight", "forward", "backward",
    "toward", "away", "across", "through", "around", "near",
}

POSITION_TERMS = {"above", "behind", "beside", "between", "inside", "near", "nearby", "outside", "under"}
QUANTITY_TERMS = {"one", "two", "three", "four", "few", "several", "many", "multiple", "pair"}

COLOR_TERMS = {
    "black",
    "blue",
    "bright",
    "dark",
    "gray",
    "green",
    "grey",
    "red",
    "silver",
    "white",
    "yellow",
}

QUANTITY_OR_DETERMINER_TERMS = {
    "all",
    "any",
    "each",
    "every",
    "few",
    "many",
    "multiple",
    "other",
    "several",
    "some",
}

NON_ENTITY_DESCRIPTOR_TERMS = {
    "active",
    "afternoon",
    "black",
    "calm",
    "cast",
    "color",
    "different",
    "halt",
    "late",
    "likely",
    "navigate",
    "noticeable",
    "other",
    "position",
    "progress",
    "progresse",
    "relatively",
    "shadow",
    "significant",
    "slowly",
    "some",
    "steadily",
    "taken",
    "under",
    "vehicular",
    "visible",
    "white",
}

FUNCTION_NON_TARGET_TERMS = {
    "and",
    "between",
    "both",
    "either",
    "for",
    "from",
    "into",
    "or",
    "over",
    "that",
    "the",
    "these",
    "this",
    "those",
    "there",
    "while",
    "who",
    "which",
    "with",
}

TIME_NON_TARGET_TERMS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

ABSTRACT_NON_TARGET_TERMS = {
    "activity",
    "area",
    "busy",
    "clear",
    "direction",
    "event",
    "events",
    "indicate",
    "interaction",
    "movement",
    "multiple",
    "perspective",
    "presence",
    "remain",
    "seen",
    "stationary",
    "suburban",
    "sunny",
    "surveillance",
    "variou",
    "various",
}

DOMAIN_TRIGGERS: dict[str, set[str]] = {
    "traffic": {"traffic", "road", "street", "intersection", "lane", "vehicle", "car", "bus", "truck"},
    "performance": {"performer", "stage", "dancer", "singer", "audience", "crowd", "banner", "music"},
    "sports": {"player", "athlete", "ball", "court", "field", "game"},
    "nature": {"animal", "tree", "river", "mountain", "forest", "sky", "water"},
    "indoor": {"room", "table", "chair", "screen", "door", "window"},
}

DOMAIN_TARGETS: dict[str, set[str]] = {
    "traffic": {"car", "vehicle", "truck", "bus", "road", "lane", "intersection", "crosswalk", "sidewalk", "traffic light", "traffic signal", "pedestrian", "cyclist", "sign"},
    "performance": {"person", "performer", "dancer", "singer", "speaker", "stage", "banner", "crowd", "instrument", "guitar", "microphone"},
    "sports": {"player", "athlete", "ball", "court", "field"},
    "nature": {"animal", "dog", "cat", "bird", "horse", "cow", "sheep", "tree", "river", "mountain"},
    "indoor": {"person", "table", "chair", "screen", "door", "window", "room"},
}

OBJECT_ALIASES: dict[str, list[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "child", "children", "host", "participant", "student"],
    "crowd": ["crowd", "crowds", "audience", "group", "groups"],
    "performer": ["performer", "performers"],
    "dancer": ["dancer", "dancers"],
    "singer": ["singer", "singers"],
    "speaker": ["speaker", "speakers"],
    "player": ["player", "players"],
    "athlete": ["athlete", "athletes"],
    "driver": ["driver", "drivers"],
    "face": ["face", "faces"],
    "hand": ["hand", "hands"],
    "car": ["car", "cars", "sedan", "sedans"],
    "vehicle": ["vehicle", "vehicles"],
    "bus": ["bus", "buses"],
    "truck": ["truck", "trucks"],
    "van": ["van", "vans"],
    "suv": ["suv", "suvs"],
    "taxi": ["taxi", "taxis"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "cyclist": ["cyclist", "cyclists"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "road": ["road", "street", "lane", "highway"],
    "intersection": ["intersection", "crosswalk", "junction"],
    "traffic light": ["traffic light", "signal light", "red light", "green light"],
    "sign": ["sign", "signage", "billboard"],
    "building": ["building", "buildings", "house", "storefront"],
    "tree": ["tree", "trees", "vegetation"],
    "river": ["river", "rivers"],
    "mountain": ["mountain", "mountains"],
    "animal": ["animal", "animals"],
    "dog": ["dog", "dogs"],
    "cat": ["cat", "cats"],
    "bird": ["bird", "birds"],
    "horse": ["horse", "horses"],
    "cow": ["cow", "cows"],
    "sheep": ["sheep"],
    "ball": ["ball", "balls"],
    "instrument": ["instrument", "instruments"],
    "guitar": ["guitar", "guitars"],
    "microphone": ["microphone", "microphones", "mic", "mics"],
    "bag": ["bag", "bags"],
    "flag": ["flag", "flags"],
    "banner": ["banner", "banners"],
    "stage": ["stage", "stages"],
    "field": ["field", "fields"],
    "court": ["court", "courts"],
    "room": ["room", "rooms"],
    "table": ["table", "desk"],
    "chair": ["chair", "chairs", "seat", "seats"],
    "laptop": ["laptop", "computer", "macbook"],
    "screen": ["screen", "monitor", "display"],
    "phone": ["phone", "smartphone", "mobile"],
    "document": ["document", "paper", "slide", "text"],
    "door": ["door", "entrance"],
    "window": ["window", "windows"],
    "board": ["board", "whiteboard", "blackboard"],
}

SCENE_TAGS: dict[str, list[str]] = {
    "traffic": ["traffic", "road", "street", "intersection", "lane", "vehicle", "car", "bus", "truck"],
    "indoor": ["indoor", "room", "office", "studio", "classroom", "table", "desk", "wall"],
    "conversation": ["talk", "talking", "discuss", "discussion", "speak", "speaker", "conversation"],
    "presentation": ["slide", "presentation", "screen", "whiteboard", "lecture", "demonstration"],
    "outdoor": ["outdoor", "sky", "tree", "building", "road", "street"],
    "crowd": ["crowd", "people", "audience", "group"],
}

ACTION_TERMS = {
    "appear",
    "approach",
    "arrive",
    "change",
    "continue",
    "cross",
    "depart",
    "discuss",
    "enter",
    "exit",
    "follow",
    "hold",
    "leave",
    "look",
    "move",
    "park",
    "pass",
    "point",
    "present",
    "say",
    "sit",
    "slow",
    "speak",
    "stand",
    "start",
    "stop",
    "talk",
    "turn",
    "wait",
    "walk",
}

ATTRIBUTE_TERMS = {
    "black",
    "blue",
    "bright",
    "dark",
    "green",
    "large",
    "red",
    "small",
    "white",
    "yellow",
    "young",
}
SIGNAL_STATE_COLOR_TERMS = {"green", "red", "yellow"}
SIGNAL_TARGET_LABELS = {"traffic light", "traffic signal", "signal light"}

NON_OBJECT_TERMS = (
    ACTION_TERMS
    | MOTION_KEYWORDS
    | STATE_KEYWORDS
    | DIRECTION_KEYWORDS
    | FUNCTION_NON_TARGET_TERMS
    | TIME_NON_TARGET_TERMS
    | ABSTRACT_NON_TARGET_TERMS
    | COLOR_TERMS
    | QUANTITY_OR_DETERMINER_TERMS
    | NON_ENTITY_DESCRIPTOR_TERMS
    | GENERIC_NON_TARGET_BLACKLIST
    | {
    "behind",
    "beside",
    "caption",
    "current",
    "light",
    "near",
    "nearby",
    "through",
    "traffic",
    "transcript",
    }
)

CONTEXT_ANCHOR_TERMS = (
    ACTION_TERMS
    | MOTION_KEYWORDS
    | STATE_KEYWORDS
    | DIRECTION_KEYWORDS
    | POSITION_TERMS
    | QUANTITY_TERMS
    | ATTRIBUTE_TERMS
)


def _sorted_segment_items(segments: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        idx = item[0]
        try:
            return int(idx), idx
        except ValueError:
            return 10**9, idx

    return sorted(segments.items(), key=key)


def _match_terms(text: str, aliases: list[str]) -> int:
    lowered = text.lower()
    count = 0
    for alias in aliases:
        pattern = rf"\b{re.escape(alias.lower())}\b"
        count += len(re.findall(pattern, lowered))
    return count


def _extract_scene_tags(text: str) -> list[str]:
    tags = [tag for tag, terms in SCENE_TAGS.items() if _match_terms(text, terms)]
    return sorted(tags)


def _normalize_label(label: str) -> str:
    value = re.sub(r"[^a-z0-9\s-]+", " ", label.lower())
    value = re.sub(r"\s+", " ", value).strip()
    singulars = {
        "buses": "bus",
        "children": "child",
        "men": "man",
        "people": "person",
        "vehicles": "vehicle",
        "women": "woman",
    }
    if value in singulars:
        return singulars[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 4:
        return value[:-1]
    return value


def _active_domains(text: str, scene_tags: list[str] | None = None) -> set[str]:
    text_tokens = set(tokenize(text))
    domains = set(scene_tags or [])
    for domain, triggers in DOMAIN_TRIGGERS.items():
        if text_tokens & triggers or _match_terms(text, list(triggers)):
            domains.add(domain)
    return domains


def _has_nearby_anchor(text: str, label: str, window: int = 4) -> bool:
    tokens = [_normalize_label(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())]
    anchors = {_normalize_label(term) for term in CONTEXT_ANCHOR_TERMS}
    label_tokens = label.split()
    if not tokens or not label_tokens:
        return False
    for idx in range(0, len(tokens) - len(label_tokens) + 1):
        if tokens[idx : idx + len(label_tokens)] != label_tokens:
            continue
        left = max(0, idx - window)
        right = min(len(tokens), idx + len(label_tokens) + window)
        if set(tokens[left:right]) & anchors:
            return True
    return False


def _has_domain_target_context(text: str, label: str, domains: set[str], window: int = 6) -> bool:
    label_tokens = label.split()
    if not label_tokens:
        return False
    for domain in domains:
        domain_targets = {_normalize_label(target) for target in DOMAIN_TARGETS.get(domain, set())}
        if label in domain_targets:
            return True
    tokens = [_normalize_label(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())]
    anchors = {_normalize_label(term) for term in CONTEXT_ANCHOR_TERMS}
    for idx in range(0, len(tokens) - len(label_tokens) + 1):
        if tokens[idx : idx + len(label_tokens)] != label_tokens:
            continue
        left = max(0, idx - window)
        right = min(len(tokens), idx + len(label_tokens) + window)
        nearby = set(tokens[left:right])
        for domain in domains:
            domain_targets = {_normalize_label(target) for target in DOMAIN_TARGETS.get(domain, set())}
            if nearby & (domain_targets | anchors):
                return True
    return False


def _extract_keywords(text: str, keywords: set[str]) -> list[str]:
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower()))
    return sorted(keyword for keyword in keywords if keyword in tokens)


def _extract_target_labels(text: str, max_labels: int = 8) -> list[tuple[str, float]]:
    counts = Counter()
    domains = _active_domains(text, _extract_scene_tags(text))
    blocked_terms = {_normalize_label(term) for term in (GENERIC_NON_TARGET_BLACKLIST | NON_OBJECT_TERMS)}
    for label, aliases in OBJECT_ALIASES.items():
        label = _normalize_label(label)
        if label in blocked_terms:
            continue
        count = _match_terms(text, aliases)
        if not count:
            continue
        if label in GENERAL_TARGET_WHITELIST:
            counts[label] = count + 0.5
        elif count >= 2 or _has_nearby_anchor(text, label) or _has_domain_target_context(text, label, domains):
            counts[label] = count

    # Keep a small open-vocabulary tail for domain words not in the seed list.
    normalized_tokens = [_normalize_label(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())]
    token_counts = Counter(token for token in normalized_tokens if token)
    known_aliases = {_normalize_label(alias) for aliases in OBJECT_ALIASES.values() for alias in aliases}
    known_aliases.update(_normalize_label(label) for label in OBJECT_ALIASES)
    domain_targets = {
        _normalize_label(target)
        for targets in DOMAIN_TARGETS.values()
        for target in targets
    }
    for label, occurrence_count in token_counts.items():
        if len(label) < 4 or label in blocked_terms:
            continue
        if label in known_aliases:
            continue
        if label.endswith("ing") or label.endswith("ed"):
            continue
        if label in GENERAL_TARGET_WHITELIST:
            counts[label] += max(1.0, float(occurrence_count)) + 0.5
        elif label in domain_targets and _has_domain_target_context(text, label, domains):
            counts[label] += max(0.6, float(occurrence_count) * 0.5)
        elif (
            occurrence_count >= 3
            and _has_nearby_anchor(text, label)
            and not label.endswith("ly")
            and label not in blocked_terms
        ):
            counts[label] += max(0.4, float(occurrence_count) * 0.35)

    return [(label, max(1.0, float(count))) for label, count in counts.most_common(max_labels)]


def _extract_action_tags(text: str) -> list[str]:
    tokens = set(tokenize(text))
    actions = sorted(action for action in ACTION_TERMS if action in tokens or f"{action}s" in tokens)
    return actions


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _extract_attributes(text: str, label: str = "") -> list[str]:
    tokens = set(tokenize(text))
    if not _env_bool("EVIQUE_STRICT_COLOR_BINDING", True):
        return sorted(attr for attr in ATTRIBUTE_TERMS if attr in tokens)

    signal_state_colors = _signal_state_colors(text)
    attributes: list[str] = []
    for attr in sorted(ATTRIBUTE_TERMS):
        if attr not in tokens:
            continue
        if attr in COLOR_TERMS:
            if _is_signal_target(label) and attr in signal_state_colors:
                attributes.append(f"signal_state_{attr}")
                continue
            if _attribute_bound_to_label(text, label, attr):
                attributes.append(attr)
            continue
        if _attribute_bound_to_label(text, label, attr):
            attributes.append(attr)
    return sorted(dict.fromkeys(attributes))


def _is_signal_target(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return normalized in SIGNAL_TARGET_LABELS


def _signal_state_colors(text: str) -> set[str]:
    lowered = re.sub(r"\s+", " ", text.lower())
    colors: set[str] = set()
    patterns = [
        r"\b(?:traffic\s+)?(?:light|signal)s?\s+(?:turns?|turned|changes?|changed|becomes?|became|is|are)\s+(green|red|yellow)\b",
        r"\b(green|red|yellow)\s+(?:traffic\s+)?(?:light|signal)s?\b",
        r"\b(green|red|yellow)[-\s]+light\b",
    ]
    for pattern in patterns:
        colors.update(match.group(1) for match in re.finditer(pattern, lowered))
    return colors


def _attribute_bound_to_label(text: str, label: str, attribute: str) -> bool:
    if not label:
        return False
    lowered = re.sub(r"\s+", " ", text.lower())
    aliases = [label] + OBJECT_ALIASES.get(label, [])
    modifier = r"(?:(?:small|large|bright|dark|light|young)\s+)?"
    for alias in dict.fromkeys(alias for alias in aliases if alias):
        escaped_alias = re.escape(alias.lower())
        escaped_attr = re.escape(attribute.lower())
        patterns = [
            rf"\b{escaped_attr}(?:[-\s]+(?:colored|coloured))?\s+{modifier}{escaped_alias}\b",
            rf"\b{escaped_alias}\s+(?:is|are|appears|appear|looks|look|seems|seem|with|has|have|having)\s+(?:a\s+)?{escaped_attr}\b",
            rf"\b{escaped_alias}\s+(?:colored|coloured)\s+{escaped_attr}\b",
        ]
        if any(re.search(pattern, lowered) for pattern in patterns):
            # Avoid binding signal-state colors to ordinary object color attributes.
            if _is_signal_target(label) and attribute in _signal_state_colors(text):
                return False
            return True
    return False


def _target_text(label: str, scope: ScopeViewRecord, attributes: list[str]) -> str:
    attrs = f" attributes={', '.join(attributes)}." if attributes else ""
    return (
        f"Object evidence: {label} appears in segment {scope.segment_id} "
        f"({format_seconds(scope.start_time)}-{format_seconds(scope.end_time)}).{attrs} "
        f"Caption: {shorten(scope.caption, 360)}"
    )


def _track_summary(label: str, scopes: list[ScopeViewRecord], action_tags: list[str]) -> str:
    if not scopes:
        return f"Pseudo-track for {label}."
    start = format_seconds(scopes[0].start_time)
    end = format_seconds(scopes[-1].end_time)
    action_text = f" Actions observed: {', '.join(action_tags)}." if action_tags else ""
    segment_text = ", ".join(scope.segment_id for scope in scopes[:8])
    if len(scopes) > 8:
        segment_text += ", ..."
    return (
        f"Pseudo-track for {label} from {start} to {end}, observed across {len(scopes)} "
        f"segments ({segment_text}).{action_text}"
    )


def _event_type(scene_tags: list[str], action_tags: list[str], object_counts: Counter[str]) -> str:
    if "traffic" in scene_tags:
        if {"stop", "slow", "wait", "change"} & set(action_tags):
            return "traffic_state_change"
        return "traffic_activity"
    if "conversation" in scene_tags or {"talk", "speak", "discuss", "present"} & set(action_tags):
        return "conversation_or_presentation"
    if object_counts:
        return "object_activity"
    return "fixed_window"


def _event_summary(
    video_name: str,
    start_time: float | None,
    end_time: float | None,
    scene_tags: list[str],
    action_tags: list[str],
    object_counts: Counter[str],
    scopes: list[ScopeViewRecord],
) -> str:
    objects = ", ".join(f"{label}={count}" for label, count in object_counts.most_common(8)) or "no explicit targets"
    scenes = ", ".join(scene_tags) or "unspecified scene"
    actions = ", ".join(action_tags) or "no explicit action tags"
    caption_sample = " ".join(shorten(scope.caption, 180) for scope in scopes[:2])
    return (
        f"Event window in {video_name} from {format_seconds(start_time)} to {format_seconds(end_time)}. "
        f"Scene tags: {scenes}. Object evidence: {objects}. Action tags: {actions}. "
        f"Representative visual evidence: {caption_sample}"
    ).strip()


def _node(record_id: str, node_type: str, view_id: str, text: str, provenance: dict[str, Any]) -> EvidenceNode:
    return EvidenceNode(id=record_id, type=node_type, view_id=view_id, text=text, provenance=provenance)


def build_evique_from_segments(
    video_segments: dict[str, dict[str, dict[str, Any]]] | None = None,
    *,
    video_segments_path: Path | None = None,
    video_path: Path | None = None,
    video_paths: list[Path] | None = None,
    video_path_map: dict[str, str] | None = None,
    question_records: list[dict[str, Any]] | None = None,
    output_dir: Path,
    event_window_seconds: int = 120,
    track_gap_seconds: int = 120,
    visual_field: str | None = None,
) -> dict[str, Any]:
    """Build EVIQUE views from shared or native base segment records.

    Shared mode can still consume the comparison base store. Standalone mode
    consumes EVIQUE-native base records generated from raw video.
    """

    start_clock = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not visual_relations_enabled():
        remove_visual_relation_files(output_dir)
    if video_segments is None:
        if video_segments_path is None:
            raise ValueError("video_segments or video_segments_path is required")
        video_segments = read_json(video_segments_path)
    collected_video_paths = list(video_paths or [])
    if video_path is not None and not collected_video_paths:
        collected_video_paths.append(video_path)
    video_identities = collect_video_identities(
        video_segments=video_segments,
        video_paths=collected_video_paths,
        video_path_map=video_path_map,
        question_records=question_records,
    )
    identity_by_video_name = {str(identity.get("video_name")): identity for identity in video_identities}

    scopes: list[ScopeViewRecord] = []
    targets: list[TargetViewRecord] = []
    tracks: list[TrackViewRecord] = []
    events: list[EventViewRecord] = []
    nodes: list[EvidenceNode] = []
    relations: list[EvidenceRelation] = []

    scopes_by_id: dict[str, ScopeViewRecord] = {}
    targets_by_segment: dict[str, list[TargetViewRecord]] = defaultdict(list)

    for video_name, segments in video_segments.items():
        identity = identity_by_video_name.get(str(video_name)) or make_video_identity(video_name=str(video_name))
        for index, segment in _sorted_segment_items(segments):
            segment_id = f"{video_name}_{index}"
            start_time, end_time = parse_time_range(segment.get("time"))
            frame_times = [float(t) for t in segment.get("frame_times") or [] if isinstance(t, (int, float))]
            caption, transcript = extract_caption_and_transcript(segment, visual_field=visual_field)
            text = unified_segment_text(segment, visual_field=visual_field)
            provenance = Provenance(
                video_name=video_name,
                video_id=identity.get("video_id"),
                source_vid=identity.get("source_vid"),
                video_path=identity.get("video_path"),
                segment_id=segment_id,
                segment_index=str(index),
                start_time=start_time,
                end_time=end_time,
                frame_times=frame_times,
            )
            scope = ScopeViewRecord(
                id=f"scope:{segment_id}",
                node_id=f"segment:{segment_id}",
                video_name=video_name,
                segment_id=segment_id,
                segment_index=str(index),
                start_time=start_time,
                end_time=end_time,
                scene_tags=_extract_scene_tags(text),
                caption=caption,
                transcript=transcript,
                text=text,
                provenance=provenance,
                video_id=identity.get("video_id"),
                source_vid=identity.get("source_vid"),
                video_path=identity.get("video_path"),
            )
            scopes.append(scope)
            scopes_by_id[segment_id] = scope
            nodes.append(
                _node(
                    scope.node_id,
                    "segment",
                    scope.id,
                    f"Segment {segment_id}: {shorten(scope.text, 620)}",
                    asdict(provenance),
                )
            )

            for ordinal, (label, count) in enumerate(_extract_target_labels(text), start=1):
                attributes = _extract_attributes(text, label)
                aliases = OBJECT_ALIASES.get(label, [label])
                target = TargetViewRecord(
                    id=f"target:{segment_id}:{label.replace(' ', '_')}:{ordinal}",
                    node_id=f"object:{segment_id}:{label.replace(' ', '_')}:{ordinal}",
                    video_name=video_name,
                    segment_id=segment_id,
                    label=label,
                    aliases=aliases,
                    attributes=attributes,
                    confidence=min(0.95, 0.55 + 0.1 * count),
                    bbox=None,
                    text=_target_text(label, scope, attributes),
                    provenance=provenance,
                    video_id=identity.get("video_id"),
                    source_vid=identity.get("source_vid"),
                    video_path=identity.get("video_path"),
                )
                targets.append(target)
                targets_by_segment[segment_id].append(target)
                nodes.append(_node(target.node_id, "object", target.id, target.text, asdict(provenance)))
                relations.append(EvidenceRelation(scope.node_id, target.node_id, "contains", asdict(provenance)))

    tracks_by_id: dict[str, TrackViewRecord] = {}
    targets_by_video_label: dict[tuple[str, str], list[TargetViewRecord]] = defaultdict(list)
    for target in targets:
        targets_by_video_label[(target.video_name, target.label)].append(target)

    for (video_name, label), label_targets in targets_by_video_label.items():
        if _normalize_label(label) not in MOVABLE_OR_ACTOR_LABELS:
            continue
        label_targets.sort(
            key=lambda target: (
                scopes_by_id[target.segment_id].start_time
                if scopes_by_id[target.segment_id].start_time is not None
                else 10**12
            )
        )
        current: list[TargetViewRecord] = []
        last_end: float | None = None
        track_ordinal = 0

        def flush_track(items: list[TargetViewRecord]) -> None:
            nonlocal track_ordinal
            if not items:
                return
            identity = identity_by_video_name.get(str(video_name)) or make_video_identity(video_name=str(video_name))
            track_ordinal += 1
            segment_ids = []
            for item in items:
                if item.segment_id not in segment_ids:
                    segment_ids.append(item.segment_id)
            track_scopes = [scopes_by_id[sid] for sid in segment_ids if sid in scopes_by_id]
            evidence_text = " ".join(scope.text for scope in track_scopes)
            action_tags = _extract_action_tags(evidence_text)
            action_keywords = _extract_keywords(evidence_text, MOTION_KEYWORDS)
            direction_keywords = _extract_keywords(evidence_text, DIRECTION_KEYWORDS)
            state_keywords = _extract_keywords(evidence_text, STATE_KEYWORDS)
            start_value = track_scopes[0].start_time if track_scopes else None
            end_value = track_scopes[-1].end_time if track_scopes else None
            track = TrackViewRecord(
                id=f"track:{video_name}:{label.replace(' ', '_')}:{track_ordinal}",
                node_id=f"track:{video_name}:{label.replace(' ', '_')}:{track_ordinal}",
                video_name=video_name,
                label=label,
                object_ids=[item.node_id for item in items],
                segment_ids=segment_ids,
                start_time=start_value,
                end_time=end_value,
                motion_summary=_track_summary(label, track_scopes, action_tags),
                action_tags=action_tags,
                evidence_text=shorten(evidence_text, 900),
                action_keywords=action_keywords,
                direction_keywords=direction_keywords,
                state_keywords=state_keywords,
                provenance={**provenance_video_fields(identity), "source": "pseudo_track_by_label_and_time_gap"},
                video_id=identity.get("video_id"),
                source_vid=identity.get("source_vid"),
                video_path=identity.get("video_path"),
            )
            tracks.append(track)
            tracks_by_id[track.id] = track
            nodes.append(_node(track.node_id, "track", track.id, track.motion_summary, track.provenance))
            for item in items:
                relations.append(EvidenceRelation(item.node_id, track.node_id, "belongs_to_track", asdict(item.provenance)))

        for target in label_targets:
            scope = scopes_by_id[target.segment_id]
            target_start = scope.start_time
            if current and last_end is not None and target_start is not None and target_start - last_end > track_gap_seconds:
                flush_track(current)
                current = []
            current.append(target)
            last_end = scope.end_time if scope.end_time is not None else target_start
        flush_track(current)

    scopes_by_video: dict[str, list[ScopeViewRecord]] = defaultdict(list)
    for scope in scopes:
        scopes_by_video[scope.video_name].append(scope)

    for video_name, video_scopes in scopes_by_video.items():
        identity = identity_by_video_name.get(str(video_name)) or make_video_identity(video_name=str(video_name))
        video_scopes.sort(key=lambda scope: scope.start_time if scope.start_time is not None else 10**12)
        windows: dict[int, list[ScopeViewRecord]] = defaultdict(list)
        for fallback_idx, scope in enumerate(video_scopes):
            if scope.start_time is None:
                window_idx = fallback_idx // max(1, event_window_seconds // 30)
            else:
                window_idx = int(scope.start_time // event_window_seconds)
            windows[window_idx].append(scope)

        previous_event: EventViewRecord | None = None
        for window_idx in sorted(windows):
            window_scopes = windows[window_idx]
            segment_ids = [scope.segment_id for scope in window_scopes]
            related_targets = [target for sid in segment_ids for target in targets_by_segment.get(sid, [])]
            related_tracks = [
                track
                for track in tracks
                if track.video_name == video_name and any(sid in set(track.segment_ids) for sid in segment_ids)
            ]
            all_text = " ".join(scope.text for scope in window_scopes)
            scene_tags = sorted({tag for scope in window_scopes for tag in scope.scene_tags})
            action_tags = _extract_action_tags(all_text)
            object_counts = Counter(target.label for target in related_targets)
            start_value = window_scopes[0].start_time
            end_value = window_scopes[-1].end_time
            state_signature = {
                "scene_tags": scene_tags,
                "object_counts": dict(object_counts.most_common()),
                "action_tags": action_tags,
                "segment_count": len(segment_ids),
                "transcript_present": any(scope.transcript.strip() for scope in window_scopes),
            }
            event = EventViewRecord(
                id=f"event:{video_name}:{window_idx}",
                node_id=f"event:{video_name}:{window_idx}",
                video_name=video_name,
                start_time=start_value,
                end_time=end_value,
                event_type=_event_type(scene_tags, action_tags, object_counts),
                summary=_event_summary(video_name, start_value, end_value, scene_tags, action_tags, object_counts, window_scopes),
                state_signature=state_signature,
                action_tags=action_tags,
                related_segment_ids=segment_ids,
                related_object_ids=[target.node_id for target in related_targets],
                related_track_ids=[track.node_id for track in related_tracks],
                provenance={**provenance_video_fields(identity), "window_index": window_idx, "source": "fixed_window_state_signature"},
                video_id=identity.get("video_id"),
                source_vid=identity.get("source_vid"),
                video_path=identity.get("video_path"),
            )
            if previous_event is not None:
                event.previous_event_ids.append(previous_event.node_id)
                previous_event.next_event_ids.append(event.node_id)
                relations.append(EvidenceRelation(previous_event.node_id, event.node_id, "followed_by", event.provenance))
            previous_event = event
            events.append(event)
            nodes.append(_node(event.node_id, "event", event.id, event.summary, event.provenance))
            for scope in window_scopes:
                relations.append(EvidenceRelation(event.node_id, scope.node_id, "located_in", asdict(scope.provenance)))
            for target in related_targets:
                relations.append(EvidenceRelation(event.node_id, target.node_id, "supports_object", asdict(target.provenance)))
            for track in related_tracks:
                relations.append(EvidenceRelation(track.node_id, event.node_id, "participates_in", event.provenance))
                track.related_event_ids.append(event.node_id)

    track_segments = {track.node_id: set(track.segment_ids) for track in tracks}
    for i, left in enumerate(tracks):
        for right in tracks[i + 1 :]:
            if left.video_name != right.video_name or left.label == right.label:
                continue
            if track_segments[left.node_id] & track_segments[right.node_id]:
                left.neighbor_track_ids.append(right.node_id)
                right.neighbor_track_ids.append(left.node_id)
                identity = identity_by_video_name.get(str(left.video_name)) or make_video_identity(video_name=str(left.video_name))
                relations.append(
                    EvidenceRelation(
                        left.node_id,
                        right.node_id,
                        "neighbor_of",
                        {**provenance_video_fields(identity), "source": "shared_segment"},
                    )
                )

    event_segmentation_config = get_event_segmentation_config()
    event_segmentation_mode = str(event_segmentation_config.get("mode") or "adaptive")
    cost_planner_config = get_cost_planner_config()
    graph_stats = {
        "evique_version": EVIQUE_VERSION,
        "model_version": EVIQUE_VERSION_LABEL,
        "video_count": len(video_identities),
        "segment": len(scopes),
        "object": len(targets),
        "track": len(tracks),
        "event": len(events),
        "event_segmentation_mode": event_segmentation_mode,
        "adaptive_event_segmentation": event_segmentation_mode in {"adaptive", "hybrid"},
        "adaptive_event_config": {
            "min_seconds": event_segmentation_config.get("min_seconds"),
            "max_seconds": event_segmentation_config.get("max_seconds"),
            "change_threshold": event_segmentation_config.get("change_threshold"),
            "merge_gap_seconds": event_segmentation_config.get("merge_gap_seconds"),
        },
        "cost_based_view_planner": cost_planner_config,
        "relations": len(relations),
        "event_window_seconds": event_window_seconds,
        "track_gap_seconds": track_gap_seconds,
        "object_label_counts": dict(Counter(target.label for target in targets).most_common()),
        "scene_tag_counts": dict(Counter(tag for scope in scopes for tag in scope.scene_tags).most_common()),
    }
    visual_mode = os.getenv("EVIQUE_VISUAL_MODE", "caption").strip().lower()
    visual_manifest: dict[str, Any] | None = None
    graph_stats.update(visual_relation_file_metadata())
    if visual_mode in {"visual", "hybrid"}:
        visual_manifest = _build_v4_visual_manifest(
            video_segments=video_segments,
            video_identities=video_identities,
            output_dir=output_dir,
            visual_mode=visual_mode,
        )
        graph_stats.update(visual_manifest.get("graph_stats", {}))
    else:
        graph_stats.update(visual_compact_metadata({}))
    compact_relation_stats = graph_stats.get("visual_compact_stats") or {}
    graph_stats.setdefault("visual_relation_before", int(compact_relation_stats.get("relations_before") or 0))
    graph_stats.setdefault("visual_relation_after", int(compact_relation_stats.get("relations_after") or 0))

    if visual_manifest:
        visual_objects = read_jsonl(output_dir / "visual_object_view.jsonl")
        visual_tracks = read_jsonl(output_dir / "visual_track_view.jsonl")
        visual_events = read_jsonl(output_dir / "visual_event_view.jsonl")
        visual_relations = read_visual_relations(output_dir)
        keyframes = read_jsonl(output_dir / "keyframe_view.jsonl")
    else:
        visual_objects = []
        visual_tracks = []
        visual_events = []
        visual_relations = []
        keyframes = []
    adaptive_events = build_adaptive_events(
        scopes=scopes,
        targets=targets,
        tracks=tracks,
        fixed_events=events,
        keyframes=keyframes,
        visual_objects=visual_objects,
        visual_tracks=visual_tracks,
        visual_events=visual_events,
        visual_relations=visual_relations,
        config=event_segmentation_config,
    )
    adaptive_stats = adaptive_event_stats(adaptive_events, event_segmentation_config)
    graph_stats["adaptive_event"] = len(adaptive_events)
    graph_stats["adaptive_event_stats"] = adaptive_stats
    for adaptive_event in adaptive_events:
        provenance = adaptive_event.get("provenance") or {}
        nodes.append(_node(adaptive_event["node_id"], "adaptive_event", adaptive_event["id"], adaptive_event.get("summary", ""), provenance))
        for segment_id in adaptive_event.get("related_segment_ids") or []:
            scope = scopes_by_id.get(str(segment_id))
            if scope:
                relations.append(EvidenceRelation(adaptive_event["node_id"], scope.node_id, "adaptive_located_in", provenance))
    graph_stats["relations"] = len(relations)

    write_jsonl([asdict(record) for record in scopes], output_dir / "scope_view.jsonl")
    write_jsonl([asdict(record) for record in targets], output_dir / "target_view.jsonl")
    write_jsonl([asdict(record) for record in tracks], output_dir / "track_view.jsonl")
    write_jsonl([asdict(record) for record in events], output_dir / "event_view.jsonl")
    write_jsonl(adaptive_events, output_dir / "adaptive_event_view.jsonl")
    write_jsonl([asdict(record) for record in nodes], output_dir / "evidence_nodes.jsonl")
    write_jsonl([asdict(record) for record in relations], output_dir / "evidence_relations.jsonl")
    view_stats = write_view_stats(output_dir)
    graph_stats["view_stats_path"] = VIEW_STATS_FILE
    elapsed = time.perf_counter() - start_clock
    manifest = {
        "model": "EVIQUE",
        "evique_version": EVIQUE_VERSION,
        "model_version": EVIQUE_VERSION_LABEL,
        "index_build_time_seconds": elapsed,
        "index_size_mb": directory_size_bytes(output_dir) / (1024 * 1024),
        "view_files": {
            "scope": "scope_view.jsonl",
            "target": "target_view.jsonl",
            "track": "track_view.jsonl",
            "event": "event_view.jsonl",
            "adaptive_event": "adaptive_event_view.jsonl",
        },
        "adaptive_event_view": {
            "path": "adaptive_event_view.jsonl",
            "row_count": len(adaptive_events),
            "avg_duration": adaptive_stats.get("avg_duration", 0.0),
            "avg_change_score": adaptive_stats.get("avg_change_score", 0.0),
            "dominant_signal_counts": adaptive_stats.get("dominant_signal_counts", {}),
        },
        "event_segmentation": {
            "mode": event_segmentation_mode,
            "fixed_event_view_path": "event_view.jsonl",
            "adaptive_event_view_path": "adaptive_event_view.jsonl",
            "config": adaptive_stats.get("config", {}),
        },
        "cost_based_view_planner": cost_planner_config,
        "view_stats_path": VIEW_STATS_FILE,
        "view_stats": view_stats,
        "visual_mode": visual_mode,
        "multi_video_visual_index": bool(visual_manifest and visual_manifest.get("multi_video_visual_index")),
        "video_count": len(video_identities),
        "video_identity_fields": ["video_id", "source_vid", "video_path"],
        "video_identities": video_identities,
        "graph_stats": graph_stats,
    }
    manifest.update(visual_compact_metadata(graph_stats.get("visual_compact_stats", {})))
    manifest["visual_relation_before"] = int(graph_stats.get("visual_relation_before") or 0)
    manifest["visual_relation_after"] = int(graph_stats.get("visual_relation_after") or 0)
    manifest.update(visual_relation_file_metadata())
    if visual_manifest:
        manifest["visual_view_files"] = visual_manifest.get("view_files", {})
        manifest["visual_relation_files"] = visual_manifest.get("relation_files", {})
        manifest["visual_build_time_seconds"] = visual_manifest.get("visual_build_time_seconds", 0.0)
        manifest["multi_video_visual_index"] = bool(visual_manifest.get("multi_video_visual_index"))
        manifest["visual_video_count"] = int(visual_manifest.get("video_count") or 0)
        manifest["visual_debug_files"] = visual_manifest.get("debug_files", {})
        manifest.update(visual_compact_metadata(visual_manifest.get("visual_compact_stats", {})))
        manifest["visual_relation_before"] = int(visual_manifest.get("visual_relation_before") or 0)
        manifest["visual_relation_after"] = int(visual_manifest.get("visual_relation_after") or 0)
        manifest.update(
            visual_relation_file_metadata(
                visual_manifest.get("write_legacy_visual_relation_view", False),
                file_generated=bool(visual_manifest.get("visual_relations_file_generated", False)),
            )
        )
    print(f"[{EVIQUE_VERSION_LABEL}] adaptive_event_segmentation={event_segmentation_mode}")
    print(
        f"[{EVIQUE_VERSION_LABEL}] event_boundary_signals="
        f"{', '.join((event_segmentation_config.get('signal_weights') or {}).keys())}"
    )
    print(f"[{EVIQUE_VERSION_LABEL}] cost_based_view_planner={cost_planner_config.get('enabled')}")
    print(f"[{EVIQUE_VERSION_LABEL}] adaptive_event_count={len(adaptive_events)}")
    print(f"[{EVIQUE_VERSION_LABEL}] visual_mode={visual_mode}")
    print(f"[{EVIQUE_VERSION_LABEL}] multi_video_visual_index={manifest.get('multi_video_visual_index')}")
    compact_stats = manifest.get("visual_compact_stats") or {}
    print(f"[{EVIQUE_VERSION_LABEL}] compact_visual_index={'on' if compact_stats.get('enabled') else 'off'}")
    print(f"[{EVIQUE_VERSION_LABEL}] visual_compact_level={compact_stats.get('level', 'balanced')}")
    print(
        f"[{EVIQUE_VERSION_LABEL}] visual_object_before/after="
        f"{compact_stats.get('objects_before', 0)}/{compact_stats.get('objects_after', 0)}"
    )
    print(
        f"[{EVIQUE_VERSION_LABEL}] visual_track_before/after="
        f"{compact_stats.get('tracks_before', 0)}/{compact_stats.get('tracks_after', 0)}"
    )
    print(
        f"[{EVIQUE_VERSION_LABEL}] visual_relation_before/after="
        f"{compact_stats.get('relations_before', 0)}/{compact_stats.get('relations_after', 0)}"
    )
    print(
        f"[{EVIQUE_VERSION_LABEL}] visual_index_size_before/after="
        f"{float(compact_stats.get('visual_index_size_bytes_before') or 0) / (1024 * 1024):.2f}MB/"
        f"{float(compact_stats.get('visual_index_size_bytes_after') or 0) / (1024 * 1024):.2f}MB"
    )
    print(f"[{EVIQUE_VERSION_LABEL}] video_count={manifest.get('video_count')}")
    print(f"[{EVIQUE_VERSION_LABEL}] video-aware retrieval enabled")
    write_json(graph_stats, output_dir / "graph_stats.json")
    write_json(manifest, output_dir / "index_manifest.json")
    return manifest


def _safe_identity_dir(identity: dict[str, Any]) -> str:
    raw = str(identity.get("video_id") or identity.get("video_name") or "video")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or "video"


def _visual_identity_subset(
    *,
    video_segments: dict[str, dict[str, dict[str, Any]]],
    video_identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    with_paths = [identity for identity in video_identities if identity.get("video_path")]
    env_video_path = os.getenv("EVIQUE_VIDEO_PATH")
    if with_paths:
        return with_paths
    if env_video_path and len(video_segments) == 1:
        video_name = next(iter(video_segments))
        return [make_video_identity(video_name=str(video_name), video_path=env_video_path)]
    if env_video_path and len(video_segments) > 1:
        raise ValueError(
            "EVIQUE_VISUAL_MODE=visual|hybrid found multiple videos but only a single "
            "EVIQUE_VIDEO_PATH. EVIQUE v4 requires per-video paths for multi-video visual build."
        )
    raise ValueError(
        "EVIQUE_VISUAL_MODE=visual|hybrid requires at least one video path from --video, "
        "kv_store_video_path.json, questions/manifest metadata, or single-video EVIQUE_VIDEO_PATH fallback."
    )


def _segments_for_visual_identity(
    identity: dict[str, Any],
    video_segments: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    video_name = str(identity.get("video_name") or "")
    if video_name in video_segments:
        return {video_name: video_segments[video_name]}
    if identity.get("video_path"):
        resolved_name = resolve_video_name_for_path(
            str(identity.get("video_path")),
            segment_names=video_segments.keys(),
        )
        if resolved_name and resolved_name in video_segments:
            identity["video_name"] = resolved_name
            return {resolved_name: video_segments[resolved_name]}
    if len(video_segments) == 1:
        only_name = next(iter(video_segments))
        identity["video_name"] = only_name
        return {only_name: video_segments[only_name]}
    return {}


def _combine_visual_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key) for row in rows if row.get(key)).most_common())


def _build_v4_visual_manifest(
    *,
    video_segments: dict[str, dict[str, dict[str, Any]]],
    video_identities: list[dict[str, Any]],
    output_dir: Path,
    visual_mode: str,
) -> dict[str, Any]:
    visual_identities = _visual_identity_subset(video_segments=video_segments, video_identities=video_identities)
    multi_video = len(visual_identities) > 1
    keyframes: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    build_seconds = 0.0
    compact_stats_by_video: list[dict[str, Any]] = []
    debug_files: dict[str, Any] = {}
    relations_enabled = visual_relations_enabled()

    if not multi_video:
        identity = dict(visual_identities[0])
        subset = _segments_for_visual_identity(identity, video_segments)
        manifest = build_visual_evique(
            video_path=Path(str(identity["video_path"])),
            video_segments=subset or video_segments,
            output_dir=output_dir,
            video_identity=identity,
        )
        manifest.update(
            {
                "evique_version": EVIQUE_VERSION,
                "visual_mode": visual_mode,
                "multi_video_visual_index": False,
                "video_count": 1,
                "video_identity_fields": ["video_id", "source_vid", "video_path"],
                "video_identities": [identity],
            }
        )
        return manifest

    for identity in visual_identities:
        identity = dict(identity)
        subset = _segments_for_visual_identity(identity, video_segments)
        if not subset:
            raise ValueError(
                f"Cannot match visual video path to segment store: {identity.get('video_path')}"
            )
        safe_identity = _safe_identity_dir(identity)
        per_video_dir = output_dir / "visual_per_video" / safe_identity
        manifest = build_visual_evique(
            video_path=Path(str(identity["video_path"])),
            video_segments=subset,
            output_dir=per_video_dir,
            video_identity=identity,
        )
        build_seconds += float(manifest.get("visual_build_time_seconds", 0.0))
        compact_stats_by_video.append(manifest.get("visual_compact_stats", {}))
        for key, debug_file in (manifest.get("debug_files") or {}).items():
            if not isinstance(debug_file, dict):
                continue
            debug_path = debug_file.get("path")
            if debug_path:
                debug_files[f"{safe_identity}:{key}"] = {
                    "path": f"visual_per_video/{safe_identity}/{debug_path}",
                    "debug_only": True,
                }
        keyframes.extend(read_jsonl(per_video_dir / "keyframe_view.jsonl"))
        objects.extend(read_jsonl(per_video_dir / "visual_object_view.jsonl"))
        tracks.extend(read_jsonl(per_video_dir / "visual_track_view.jsonl"))
        relations.extend(read_visual_relations(per_video_dir))
        events.extend(read_jsonl(per_video_dir / "visual_event_view.jsonl"))

    write_jsonl(keyframes, output_dir / "keyframe_view.jsonl")
    write_jsonl(objects, output_dir / "visual_object_view.jsonl")
    write_jsonl(tracks, output_dir / "visual_track_view.jsonl")
    relation_file_metadata = visual_relation_file_metadata(file_generated=relations_enabled)
    if relations_enabled:
        write_jsonl(relations, output_dir / "visual_relations.jsonl")
    if relation_file_metadata["write_legacy_visual_relation_view"]:
        write_jsonl(relations, output_dir / "visual_relation_view.jsonl")
    write_jsonl(events, output_dir / "visual_event_view.jsonl")

    visual_stats = {
        "visual_keyframe": len(keyframes),
        "visual_object": len(objects),
        "visual_track": len(tracks),
        "visual_relation": len(relations),
        "visual_relations": len(relations),
        "visual_event": len(events),
        "visual_object_label_counts": _combine_visual_counts(objects, "label"),
        "visual_color_counts": _combine_visual_counts(objects, "color"),
        "visual_track_label_counts": _combine_visual_counts(tracks, "label"),
        "visual_event_type_counts": _combine_visual_counts(events, "event_type"),
    }
    visual_compact_stats = merge_visual_compact_stats(compact_stats_by_video)
    visual_stats.update(visual_compact_metadata(visual_compact_stats))
    visual_stats["visual_relation_before"] = int(visual_compact_stats.get("relations_before") or 0)
    visual_stats["visual_relation_after"] = int(visual_compact_stats.get("relations_after") or 0)
    visual_stats.update(relation_file_metadata)
    relation_files = {}
    if relation_file_metadata["visual_relations_enabled"]:
        relation_files["visual_relations"] = "visual_relations.jsonl"
        if relation_file_metadata["legacy_visual_relation_file"]:
            relation_files["legacy_visual_relation"] = relation_file_metadata["legacy_visual_relation_file"]
    return {
        "evique_version": EVIQUE_VERSION,
        "visual_mode": visual_mode,
        "multi_video_visual_index": True,
        "video_count": len(visual_identities),
        "video_identity_fields": ["video_id", "source_vid", "video_path"],
        "video_identities": visual_identities,
        "visual_build_time_seconds": build_seconds,
        "view_files": {
            "keyframe": "keyframe_view.jsonl",
            "visual_object": "visual_object_view.jsonl",
            "visual_track": "visual_track_view.jsonl",
            "visual_event": "visual_event_view.jsonl",
        },
        "relation_files": relation_files,
        **relation_file_metadata,
        "debug_files": debug_files,
        "graph_stats": visual_stats,
        **visual_compact_metadata(visual_compact_stats),
    }
