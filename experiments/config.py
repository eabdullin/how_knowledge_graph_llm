"""Experiment configuration, model registry, and bundled benchmark loading."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .knowledge_graph_registry import (
    get_graph_image_path,
    get_graph_mermaid_text_path,
    load_graph_catalog,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark.jsonl"
load_dotenv(REPO_ROOT / ".env")


class Modality(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    TOOLS = "tools"
    NO_GRAPH = "no_graph"


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class ModelConfig:
    provider: ModelProvider
    model_id: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    reasoning_effort: Optional[str] = None
    enable_thinking: Optional[bool] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    supports_images: bool = True

    @property
    def name(self) -> str:
        return f"{self.provider.value}/{self.model_id}"


_GPT_54_MINI_LOW = ModelConfig(
    ModelProvider.OPENAI,
    "gpt-5.4-mini",
    temperature=1.0,
    max_tokens=4096,
    reasoning_effort="low",
)


MODELS: dict[str, ModelConfig] = {
    # Canonical name for the configuration actually used by the historical
    # paper run labelled "gpt-5.4-nano-medium".
    "gpt-5.4-mini-low": _GPT_54_MINI_LOW,
    "gpt-5.4-mini-medium": ModelConfig(
        ModelProvider.OPENAI,
        "gpt-5.4-mini",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="medium",
    ),
    "gpt-5.4-low": ModelConfig(
        ModelProvider.OPENAI,
        "gpt-5.4",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="low",
    ),
    "gpt-5.4-medium": ModelConfig(
        ModelProvider.OPENAI,
        "gpt-5.4",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="medium",
    ),
    "gpt-5.4-high": ModelConfig(
        ModelProvider.OPENAI,
        "gpt-5.4",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="high",
    ),
    "gpt-5.4-none": ModelConfig(
        ModelProvider.OPENAI,
        "gpt-5.4",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="none",
    ),
    "gemma-4-31b-it": ModelConfig(
        ModelProvider.OPENAI_COMPATIBLE,
        os.environ.get("GEMMA_31B_MODEL_ID", "google/gemma-4-31b-it"),
        base_url=(
            os.environ.get("GEMMA_31B_BASE_URL")
            or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        ),
        api_key_env="GEMMA_31B_API_KEY",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=4096,
        enable_thinking=True,
        supports_images=True,
    ),
    "gemma-4-E4B-it": ModelConfig(
        ModelProvider.OPENAI_COMPATIBLE,
        os.environ.get("GEMMA_E4B_MODEL_ID", "google/gemma-4-E4B-it"),
        base_url=(
            os.environ.get("GEMMA_E4B_BASE_URL")
            or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        ),
        api_key_env="GEMMA_E4B_API_KEY",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=4096,
        enable_thinking=True,
        supports_images=True,
    ),
}


HISTORICAL_MODEL_ALIASES = {
    "gpt-5.4-nano-medium": "gpt-5.4-mini-low",
}
for alias, canonical_name in HISTORICAL_MODEL_ALIASES.items():
    MODELS[alias] = MODELS[canonical_name]


def resolve_model_config(name: str, *, warn_alias: bool = True) -> ModelConfig:
    """Resolve a public model name and disclose historical aliases."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    if warn_alias and name in HISTORICAL_MODEL_ALIASES:
        canonical = HISTORICAL_MODEL_ALIASES[name]
        warnings.warn(
            f"'{name}' is a historical paper-run label. It actually resolves "
            f"to '{canonical}' (backend gpt-5.4-mini, low reasoning).",
            UserWarning,
            stacklevel=2,
        )
    return MODELS[name]


@dataclass
class ConversationTurn:
    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    kg_items_used: list[str] = field(default_factory=list)
    thinking: Optional[str] = None
    usage: dict = field(default_factory=dict)


@dataclass
class ConversationLog:
    experiment_id: str
    modality: Modality
    model: str
    problem_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len([turn for turn in self.turns if turn.role == "modeller"])

    @property
    def modeller_text(self) -> str:
        return "\n".join(
            turn.content
            for turn in self.turns
            if turn.role == "modeller" and turn.content
        )


@dataclass
class ExperimentConfig:
    modalities: list[Modality] = field(
        default_factory=lambda: [
            Modality.IMAGE,
            Modality.TEXT,
            Modality.TOOLS,
            Modality.NO_GRAPH,
        ]
    )
    models: list[str] = field(default_factory=lambda: ["gpt-5.4-mini-medium"])
    max_turns: int = 40
    num_runs: int = 5
    output_dir: Path = Path("results")
    user_model: str = "gpt-5.4-none"
    judge_model: str = "gpt-5.4-medium"


KG_IMAGE_PATHS: dict[str, str | None] = {
    topic: get_graph_image_path(topic) for topic in load_graph_catalog()
}
KG_TEXT_PATHS: dict[str, str] = {
    topic: get_graph_mermaid_text_path(topic) for topic in load_graph_catalog()
}


def load_benchmark_problems() -> dict[str, dict]:
    """Load the exact 30-problem benchmark reported in the paper."""
    problems: dict[str, dict] = {}
    with BENCHMARK_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row["id"]
            if problem_id in problems:
                raise ValueError(
                    f"Duplicate benchmark id '{problem_id}' at line {line_number}"
                )
            problems[problem_id] = {
                "id": problem_id,
                "domain": row["kg_topic"],
                "description": row["description"],
                "kg_topic": row["kg_topic"],
                "source": "mamo_two_kg_subset",
                "mamo_difficulty": row["difficulty"],
                "mamo_row_index": row["source_row_index"],
                "selection_reason": row["selection_reason"],
            }
    validate_benchmark(problems)
    return problems


def validate_benchmark(problems: dict[str, dict]) -> None:
    """Enforce the benchmark invariants stated in the paper."""
    if len(problems) != 30:
        raise ValueError(f"Expected 30 benchmark problems, found {len(problems)}")

    topic_counts: dict[str, int] = {}
    for problem in problems.values():
        topic = problem["kg_topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    expected = {
        "Production Planning": 15,
        "Travel Salesman Problem": 15,
    }
    if topic_counts != expected:
        raise ValueError(
            f"Expected benchmark topic counts {expected}, found {topic_counts}"
        )


def load_mamo_two_kg_subset(
    max_problems: Optional[int] = None,
) -> list[dict]:
    """Compatibility wrapper for older analysis code."""
    problems = list(load_benchmark_problems().values())
    return problems[:max_problems] if max_problems else problems
