"""Knowledge graph asset discovery and lookup."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_DIR = _WORKSPACE_ROOT / "knowledge_graphs"


def _normalize_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _topic_from_stem(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip()


@lru_cache(maxsize=1)
def load_graph_catalog() -> dict[str, dict]:
    catalog = {}
    for path in sorted(_GRAPH_DIR.glob("*.mermaid")):
        raw_text = path.read_text()
        topic = _topic_from_stem(path)
        image_path = path.with_suffix(".png")
        record = catalog.setdefault(topic, {"topic": topic})
        record["mermaid_text"] = raw_text
        record["mermaid_text_path"] = path
        if image_path.exists():
            record["image_path"] = image_path

    for path in sorted(_GRAPH_DIR.glob("*.dot")):
        raw_text = path.read_text()
        topic = _topic_from_stem(path)
        image_path = path.with_suffix(".png")
        record = catalog.setdefault(topic, {"topic": topic})
        record["dot_text"] = raw_text
        record["dot_text_path"] = path
        if image_path.exists():
            record["image_path"] = image_path

    for topic, record in catalog.items():
        record.setdefault("mermaid_text", None)
        record.setdefault("mermaid_text_path", None)
        record.setdefault("dot_text", None)
        record.setdefault("dot_text_path", None)
        record.setdefault("image_path", None)

    return catalog


def list_graph_topics() -> list[str]:
    return sorted(load_graph_catalog())


def get_graph_record(topic: str) -> dict:
    topic_normalized = _normalize_topic(topic)
    catalog = load_graph_catalog()
    for graph_topic, record in catalog.items():
        if _normalize_topic(graph_topic) == topic_normalized:
            return record
    available = ", ".join(sorted(catalog))
    raise KeyError(f"Knowledge graph topic '{topic}' not found. Available: {available}")


def get_graph_text(topic: str) -> str:
    return get_graph_mermaid_text(topic)


def get_graph_mermaid_text(topic: str) -> str:
    record = get_graph_record(topic)
    if record["mermaid_text"] is None:
        raise KeyError(f"Mermaid graph text not found for topic '{topic}'")
    return record["mermaid_text"]


def get_graph_dot_text(topic: str) -> str:
    record = get_graph_record(topic)
    if record["dot_text"] is None:
        raise KeyError(f"DOT graph text not found for topic '{topic}'")
    return record["dot_text"]


def get_graph_image_path(topic: str) -> str | None:
    image_path = get_graph_record(topic)["image_path"]
    return str(image_path) if image_path else None


def get_graph_text_path(topic: str) -> str:
    return get_graph_mermaid_text_path(topic)


def get_graph_mermaid_text_path(topic: str) -> str:
    record = get_graph_record(topic)
    path = record["mermaid_text_path"]
    if path is None:
        raise KeyError(f"Mermaid graph path not found for topic '{topic}'")
    return str(path)


def get_graph_dot_text_path(topic: str) -> str:
    record = get_graph_record(topic)
    path = record["dot_text_path"]
    if path is None:
        raise KeyError(f"DOT graph path not found for topic '{topic}'")
    return str(path)