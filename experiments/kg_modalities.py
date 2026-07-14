"""Three KG modality implementations: image, text, and tools.

Each modality class provides:
- A system prompt (with KG info embedded for text/image modalities)
- Optional tool definitions (for tools modality)
- Optional image paths (for image modality)
- A method to handle tool calls (for tools modality)
"""

import json
import os
import subprocess
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge_graph import KnowledgeGraph



# Base modeller prompt (adapted from GRPO paper Appendix D.2)
_BASE_MODELLER_PROMPT = """\
YOU ARE "Optimouse" - An AI Operations Researcher (modeller) helping \
clients to formulate optimization problem statements.

Your goal is to gather necessary details that could help to build a \
linear programming model (or mixed integer linear programming model). \
You don't have to build the model.

Engage clients by asking clear, concise, and sequential questions to \
obtain components of the problem (objective function, decision \
variables, constraints).
Ask no more than one question per turn.
Do not ask multiple questions, combined questions, or multi-part sub-questions in the same turn.

Your job is to elicit information from the client, not to invent or \
confirm assumptions. Do not ask leading questions such as "Can I assume \
X is zero?" or "Is it okay if I assume X?" Ask directly for the real \
information instead, for example "What X do you have?" \
If the client does not know or the information is unavailable, record it \
as unknown or not specified and move on. Do not fill missing values with \
defaults unless the client explicitly provides that default.

If the client says they do not know, do not have the information, are not \
sure, or want to skip a field, do not ask for that same field again in \
different words. Ask each missing field at most once. Mark it as unknown \
or unavailable, continue to the next distinct business fact, and include \
the unresolved item in the final summary if it matters.

Avoid yes/no questions that contain a proposed modelling value. Prefer \
open, neutral questions about the missing business fact. If an \
assumption is still needed for a future mathematical model, list it in \
the final summary as an unresolved assumption to confirm later, not as a \
fact gathered from the client.

The client is not a math expert and has no experience with math and \
optimizations, so keep conversation simple and avoid technical terms.

Provide a summary in bullet points at the end of the conversation once \
you gathered all the information enough to build a mathematical model. \
The summary should be wrapped into markdown block. E.g. \
Here is your summary: ```markdown
  summary text
```

Your final summary must list only user-provided facts. \
Prefer explicit numbers and exact constraints over abstract category labels. \
Do not compute derived quantities, infer missing coefficients, or summarize optional branches unless the user explicitly provided them.

Start conversation with a friendly greeting and ask the client about \
their business.\
"""


_KG_USAGE_GUIDANCE = textwrap.dedent("""\
Interpret the knowledge graph as a structural map of model components, not as a script to follow from top to bottom.

How to read the graph:
- Top-level branches are component families of a model, such as entities, parameters, decisions, objective terms, and constraints.
- Child branches refine one component into variants, subtypes, or optional modules.
- Formula leaves are examples of how a component may be written mathematically. Use them to understand what information is needed, not as a checklist of questions.
- Colors or tags usually mark example domains, modelling modules, or variants. They do not mean every tagged branch is active in this user's problem.

How to use the graph in dialogue:
- After the first few user answers, identify the active component set for this problem: objective, main decision, core data, and core constraints.
- Stay mostly within that active component set.
- When the user gives a fact, map it to its component branch and continue extracting the next missing fact from that same branch before opening a sibling branch.
- Do not follow the visual order of branches. Choose the next question by importance for the minimal formulation, not by position in the graph.
- If the user gives an aggregate or nonstandard constraint, stay with that constraint and capture it exactly instead of switching to a more standard sibling branch.
- If a branch is unknown or unsupported, mark it as unknown and return to the active components. Do not walk nearby optional branches just because they are visible.

Stopping rule:
- Stop asking and summarize once you have enough information to describe a valid optimization model for the active components, even if optional modules remain unknown.
- In the final summary, include only user-provided facts and unresolved unknowns that still matter for those active components.
""")


class KGModality(ABC):
    """Base class for KG modality implementations."""

    def __init__(self, kg_text: str, kg_topic: str):
        self.kg_text = kg_text
        self.kg_topic = kg_topic
        self.kg = KnowledgeGraph(kg_text)

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def get_tools(self) -> Optional[list[dict]]:
        return None

    def get_tool_choice(
        self,
        *,
        is_first: bool,
        user_message: str | None,
    ) -> Optional[str]:
        return None

    def get_images(self) -> Optional[list[str]]:
        return None

    def handle_tool_call(self, name: str, arguments: dict) -> str:
        raise NotImplementedError("This modality does not support tool calls")

    @property
    def modality_name(self) -> str:
        return self.__class__.__name__.replace("Modality", "").lower()


class TextModality(KGModality):
    """KG provided as Mermaid-like text in the system prompt."""

    def get_system_prompt(self) -> str:
        return (
            _BASE_MODELLER_PROMPT
            + "\n\n"
            + textwrap.dedent("""\
            You have access to the following knowledge graph about optimization \
            modelling problems. Use it to guide your questions.

            """)
            + _KG_USAGE_GUIDANCE
            + textwrap.dedent("""\

            Do not mention the knowledge graph to the client.

            === KNOWLEDGE GRAPH ===
            """)
            + self.kg_text.strip()
            + "\n=== END KNOWLEDGE GRAPH ==="
        )


class NoGraphModality(KGModality):
    """No KG provided: base modeller prompt only."""

    def get_system_prompt(self) -> str:
        return _BASE_MODELLER_PROMPT

    @property
    def modality_name(self) -> str:
        return "no_graph"


class ImageModality(KGModality):
    """KG rendered as an image and provided via vision."""

    def __init__(
        self,
        kg_text: str,
        kg_topic: str,
        output_dir: Path = Path("results"),
        image_path: Optional[str] = None,
    ):
        super().__init__(kg_text, kg_topic)
        self.output_dir = output_dir
        self._image_path: Optional[str] = image_path

    def get_system_prompt(self) -> str:
        return (
            _BASE_MODELLER_PROMPT
            + "\n\n"
            + textwrap.dedent("""\
            You have access to the following knowledge graph (image) about optimization \
            modelling problems. Use it to guide your questions.
            Do not mention the image or knowledge graph to the client.

            """)
            + _KG_USAGE_GUIDANCE
        )

    def get_images(self) -> Optional[list[str]]:
        if self._image_path is None:
            raise RuntimeError("Image path not set. Either provide an image_path or call _render_kg_image() to auto-render.")
        return [self._image_path]


class ToolsModality(KGModality):
    """KG accessed via function-calling tools (current approach)."""

    def __init__(self, kg_text: str, kg_topic: str):
        super().__init__(kg_text, kg_topic)
        self._loaded_topic: str | None = None
        self._loaded_graph_text: str | None = None
        self._loaded_kg: KnowledgeGraph | None = None

    @property
    def loaded_topic(self) -> str | None:
        return self._loaded_topic

    def get_system_prompt(self) -> str:
        return (
            _BASE_MODELLER_PROMPT
            + "\n\n"
            + textwrap.dedent("""\
            You have access to a knowledge graph about optimization modelling \
            problems.

            How to use the graph:
            - Top-level branches are component families of a model, such as entities, parameters, decisions, objective terms, and constraints.
            - Child branches refine one component into variants, subtypes, or optional modules.
            - Formula leaves are examples of how a component may be written mathematically. Use them to understand what information is needed, not as a checklist of questions.
            - Colors or tags usually mark example domains, modelling modules, or variants. They do not mean every tagged branch is active in this user's problem.

            Tool policy:
            - Call load_knowledge_graph exactly once in the entire dialogue to access the knowledge graph for this client before asking graph-guided follow-up questions.
            - After loading the graph, use the navigation tools to inspect only the relevant branches.
            - You MUST use tools to access the knowledge graph and guide your questions.

            Do not mention the knowledge graph or tools to the client.
            """)
        )

    def get_tool_choice(
        self,
        *,
        is_first: bool,
        user_message: str | None,
    ) -> Optional[str]:
        if self._loaded_topic is None and not is_first and user_message:
            return "required"
        if self._loaded_topic is not None:
            return "auto"
        return None

    def get_tools(self) -> list[dict]:
        if self._loaded_topic is None:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "load_knowledge_graph",
                        "description": (
                            "Load the knowledge graph for this client. "
                            "This tool can be used only once in the dialogue."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        return [
            {
                "type": "function",
                "function": {
                    "name": "get_children",
                    "description": (
                        "Get the immediate children of a node in the loaded knowledge graph. "
                        "Use a slash-separated path relative to the loaded topic, for example "
                        "'Parameters/Cost' or 'Decision Variables'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Slash-separated label path",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_subtree",
                    "description": (
                        "Get a subtree of the knowledge graph rooted at the given "
                        "path, expanded to the requested depth."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Slash-separated label path",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Levels deep to expand (default 2)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_nodes",
                    "description": (
                        "Search the knowledge graph for nodes whose labels contain "
                        "the query string. Returns matching nodes with breadcrumb paths."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search term, e.g. 'penalty' or 'cost'",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_node_path",
                    "description": (
                        "Find nodes matching a query and return their full "
                        "root-to-node breadcrumb paths."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search term to locate in node labels",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, arguments: dict) -> str:
        required_args = {
            "load_knowledge_graph": [],
            "get_children": ["path"],
            "get_subtree": ["path"],
            "search_nodes": ["query"],
            "get_node_path": ["query"],
        }

        if not isinstance(arguments, dict):
            return json.dumps({
                "error": f"Invalid arguments for {name}",
                "received": arguments,
            }, ensure_ascii=False)

        missing_args = [arg for arg in required_args.get(name, []) if arg not in arguments]
        if missing_args:
            return json.dumps({
                "error": f"Missing required arguments for {name}: {', '.join(missing_args)}",
                "received": arguments,
            }, ensure_ascii=False)

        if name == "load_knowledge_graph":
            return self._load_knowledge_graph(arguments)

        dispatch = {
            "get_children": lambda args: self._active_kg().get_children(
                self._qualify_path(args["path"])
            ),
            "get_subtree": lambda args: self._active_kg().get_subtree(
                self._qualify_path(args["path"]), args.get("depth", 2)
            ),
            "search_nodes": lambda args: self._active_kg().search_nodes(args["query"]),
            "get_node_path": lambda args: self._active_kg().get_node_path(args["query"]),
        }
        if name not in dispatch:
            return json.dumps({"error": f"Unknown tool: {name}"})
        result = dispatch[name](arguments)
        return json.dumps(result, ensure_ascii=False)

    def _load_knowledge_graph(self, arguments: dict) -> str:
        if self._loaded_topic is not None:
            return json.dumps({
                "error": "load_knowledge_graph can only be called once per dialogue",
                "loaded_topic": self._loaded_topic,
            }, ensure_ascii=False)

        topic = self.kg_topic
        self._loaded_topic = topic
        self._loaded_graph_text = self.kg_text
        self._loaded_kg = KnowledgeGraph(self._loaded_graph_text)
        overview = self._loaded_kg.get_children(topic)
        return json.dumps({
            "loaded_topic": topic,
            "top_level_categories": overview.get("children", []),
            "symbol_legend": self._extract_symbol_legend(self._loaded_graph_text),
        }, ensure_ascii=False)

    def _active_kg(self) -> KnowledgeGraph:
        if self._loaded_kg is None:
            raise RuntimeError("Knowledge graph not loaded yet")
        return self._loaded_kg

    def _qualify_path(self, path: str) -> str:
        if not self._loaded_topic:
            return path
        normalized_path = re.sub(r"\s+", " ", path.strip())
        if not normalized_path:
            return self._loaded_topic
        if normalized_path.lower().startswith(self._loaded_topic.lower()):
            return normalized_path
        return f"{self._loaded_topic}/{normalized_path}"

    def _extract_symbol_legend(self, graph_text: str) -> list[dict[str, str]]:
        legend_header = re.compile(r"^\s*(?://|%%)\s*Symbol legend:\s*$", re.IGNORECASE)
        legend_entry = re.compile(r"^\s*(?://|%%)\s*\[([^\]]+)\]\s*=\s*(.+?)\s*$")

        legend_items: list[dict[str, str]] = []
        in_legend = False
        for line in graph_text.splitlines():
            if not in_legend:
                if legend_header.match(line):
                    in_legend = True
                continue

            if not line.strip():
                continue

            match = legend_entry.match(line)
            if match:
                legend_items.append({
                    "symbol": match.group(1).strip(),
                    "meaning": match.group(2).strip(),
                })
                continue

            break

        return legend_items


def create_modality(modality_name: str, kg_text: str, kg_topic: str, **kwargs) -> KGModality:
    """Factory for creating modality instances."""
    modalities = {
        "text": TextModality,
        "image": ImageModality,
        "tools": ToolsModality,
        "no_graph": NoGraphModality,
    }
    cls = modalities.get(modality_name)
    if cls is None:
        raise ValueError(f"Unknown modality: {modality_name}. Choose from {list(modalities)}")
    return cls(kg_text, kg_topic, **kwargs)
