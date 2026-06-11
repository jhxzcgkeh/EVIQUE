from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Iterable

from evique.video_identity import EVIQUE_VERSION, EVIQUE_VERSION_LABEL


MODULES_REQUIRED = (
    "evique.builder",
    "evique.retriever",
    "evique.standalone_base_builder",
    "evique.solo_video",
    "evique.solo_asr",
    "evique.solo_caption",
    "evique.solo_chunking",
)


def _dependency_patterns() -> list[re.Pattern[str]]:
    raw_patterns = [
        "from " + "video" + "rag",
        "import " + "video" + "rag",
        "nano" + "_graphrag",
        "nano" + "-graphrag",
        "from " + "nano",
        "import " + "nano",
        "Base" + "lines",
        "Naive" + "RAG",
        "Text" + "Video" + "RAG",
        "Video" + "RAG",
    ]
    return [re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns]


def _python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" not in path.parts:
            yield path


def _scan_package(root: Path) -> list[str]:
    issues: list[str] = []
    patterns = _dependency_patterns()
    for path in _python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                rel = path.relative_to(root)
                issues.append(f"{rel}:{line_no}: external model-code dependency marker")
    return issues


def _check_default_visual_relations() -> list[str]:
    issues: list[str] = []
    sentinel = object()
    old_value = os.environ.get("EVIQUE_DISABLE_VISUAL_RELATIONS", sentinel)
    os.environ.pop("EVIQUE_DISABLE_VISUAL_RELATIONS", None)
    try:
        from evique.utils import visual_relation_file_metadata, visual_relations_enabled

        metadata = visual_relation_file_metadata()
        if visual_relations_enabled():
            issues.append("default visual relation switch is enabled")
        if metadata.get("visual_relations_enabled") is not False:
            issues.append("default visual_relations_enabled metadata is not false")
        if metadata.get("visual_relations_file_generated") is not False:
            issues.append("default visual_relations_file_generated metadata is not false")
    finally:
        if old_value is sentinel:
            os.environ.pop("EVIQUE_DISABLE_VISUAL_RELATIONS", None)
        else:
            os.environ["EVIQUE_DISABLE_VISUAL_RELATIONS"] = str(old_value)
    return issues


def _check_imports() -> list[str]:
    issues: list[str] = []
    for module_name in MODULES_REQUIRED:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            issues.append(f"{module_name}: import failed: {exc}")
    return issues


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    print(f"EVIQUE version: {EVIQUE_VERSION_LABEL} ({EVIQUE_VERSION})")
    issues: list[str] = []
    issues.extend(_scan_package(package_root))
    issues.extend(_check_default_visual_relations())
    issues.extend(_check_imports())
    if issues:
        print("[FAIL] EVIQUE standalone package check failed")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("[OK] EVIQUE standalone package check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

