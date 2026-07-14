#!/usr/bin/env python3
"""Knowledge graph parser and navigation tools.

Parses a Mermaid-like graph definition into an in-memory tree structure
and exposes navigation methods suitable for LLM tool calling.
"""

import re
from typing import Optional

_GRAPH_HEADER = re.compile(r"^\s*graph\s+(?!\[)(.+?)\s*:?\s*$")
_DOT_GRAPH_HEADER = re.compile(r"^\s*digraph(?:\s+([^\s{]+))?\s*\{\s*$")
_EDGE_LINE = re.compile(
    r"([A-Za-z0-9_]+)(?:\[([^\]]*)\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[([^\]]*)\])?"
)
_NODE_DEF = re.compile(r"([A-Za-z0-9_]+)\[([^\]]*)\]")
_DOT_EDGE_LINE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"(?:\s*\[[^\]]*\])?\s*;?\s*$')
_DOT_NODE_DEF = re.compile(r'^\s*"([^"]+)"\s*\[label="((?:[^"\\]|\\.)*)"\]\s*;?\s*$')

MAX_SEARCH_RESULTS = 20


class KnowledgeGraph:
    """Tree-structured knowledge graph with label-path navigation."""

    def __init__(self, raw_text: str):
        self._graphs: dict[str, dict] = {}
        self._parse(raw_text)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, text: str):
        current_graph: Optional[str] = None
        graph_name_explicit = False
        nodes: dict[str, str] = {}
        edges: list[tuple[str, str]] = []
        children_set: set[str] = set()

        def finalize_graph():
            nonlocal current_graph, graph_name_explicit, nodes, edges, children_set
            if current_graph is None:
                return
            graph_name = self._resolve_graph_name(
                current_graph,
                nodes,
                children_set,
                explicit=graph_name_explicit,
            )
            self._build_graph(graph_name, nodes, edges, children_set)

        for line in text.split("\n"):
            header = _GRAPH_HEADER.match(line)
            if header:
                finalize_graph()
                current_graph = header.group(1).strip()
                graph_name_explicit = True
                nodes, edges, children_set = {}, [], set()
                continue

            dot_header = _DOT_GRAPH_HEADER.match(line)
            if dot_header:
                finalize_graph()
                current_graph = dot_header.group(1) or ""
                graph_name_explicit = False
                nodes, edges, children_set = {}, [], set()
                continue

            if current_graph is None:
                continue

            edge_match = _EDGE_LINE.search(line)
            if edge_match:
                src_id, src_label, dst_id, dst_label = edge_match.groups()
                if src_label and src_id not in nodes:
                    nodes[src_id] = src_label
                if dst_label:
                    nodes[dst_id] = dst_label
                elif dst_id not in nodes:
                    nodes[dst_id] = dst_id
                edges.append((src_id, dst_id))
                children_set.add(dst_id)
                continue

            dot_edge_match = _DOT_EDGE_LINE.match(line)
            if dot_edge_match:
                src_id, dst_id = dot_edge_match.groups()
                if src_id not in nodes:
                    nodes[src_id] = src_id
                if dst_id not in nodes:
                    nodes[dst_id] = dst_id
                edges.append((src_id, dst_id))
                children_set.add(dst_id)
                continue

            dot_node_match = _DOT_NODE_DEF.match(line)
            if dot_node_match:
                node_id, label = dot_node_match.groups()
                nodes[node_id] = self._unescape_dot_label(label)
                continue

            for nid, nlabel in _NODE_DEF.findall(line):
                if nid not in nodes:
                    nodes[nid] = nlabel

        finalize_graph()

    def _resolve_graph_name(
        self,
        current_graph: str,
        nodes: dict[str, str],
        children_set: set[str],
        *,
        explicit: bool,
    ) -> str:
        if explicit and current_graph:
            return current_graph

        roots = [nid for nid in nodes if nid not in children_set]
        if len(roots) == 1:
            return nodes[roots[0]]
        if current_graph:
            return current_graph
        raise ValueError("Could not infer graph name from DOT content")

    def _unescape_dot_label(self, label: str) -> str:
        return label.replace(r'\"', '"')

    def _build_graph(self, name: str, nodes: dict, edges: list, children_set: set):
        adj: dict[str, list[str]] = {nid: [] for nid in nodes}
        for src, dst in edges:
            adj.setdefault(src, []).append(dst)

        roots = [nid for nid in nodes if nid not in children_set]
        self._graphs[name] = {"nodes": nodes, "adj": adj, "roots": roots}

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str):
        """Resolve a '/'-separated label path to (graph_name, node_id).

        First segment matches a graph name (case-insensitive).
        Remaining segments match node labels walking the tree.
        If a root node's label equals the graph name, its children are
        promoted to the top-level candidate set for convenience.
        """
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts:
            return None, None

        graph_name, graph = self._find_graph(parts[0])
        if graph is None:
            return None, None

        if len(parts) == 1:
            return graph_name, None

        candidates = self._top_level_candidates(graph_name, graph)
        resolved_id = None

        for part in parts[1:]:
            resolved_id = self._match_candidate(graph, candidates, part)
            if resolved_id is None:
                return graph_name, None
            candidates = graph["adj"].get(resolved_id, [])

        return graph_name, resolved_id

    def _find_graph(self, name: str):
        name_lower = name.lower()
        for gname, g in self._graphs.items():
            if gname.lower() == name_lower:
                return gname, g
        return None, None

    def _top_level_candidates(self, graph_name: str, graph: dict) -> list[str]:
        """Root node IDs plus children of any root whose label matches the graph name."""
        candidates: list[str] = []
        gname_lower = graph_name.lower()
        for rid in graph["roots"]:
            candidates.append(rid)
            if graph["nodes"][rid].lower() == gname_lower:
                candidates.extend(graph["adj"].get(rid, []))
        return candidates

    def _match_candidate(self, graph: dict, candidates: list[str], label: str):
        label_lower = label.lower()
        for nid in candidates:
            if graph["nodes"].get(nid, "").lower() == label_lower:
                return nid
        for nid in candidates:
            if label_lower in graph["nodes"].get(nid, "").lower():
                return nid
        return None

    def _build_breadcrumb(self, graph_name: str, graph: dict, target_id: str) -> str:
        parent_map: dict[str, str] = {}
        for nid, children in graph["adj"].items():
            for cid in children:
                parent_map[cid] = nid

        parts = [graph["nodes"][target_id]]
        current = target_id
        while current in parent_map:
            current = parent_map[current]
            parts.append(graph["nodes"][current])
        parts.append(graph_name)
        parts.reverse()
        return " > ".join(parts)

    # ------------------------------------------------------------------
    # Public navigation API (called by LLM tools)
    # ------------------------------------------------------------------

    def list_topics(self) -> list[dict]:
        """List all root problem types with their top-level categories."""
        result = []
        for graph_name, g in self._graphs.items():
            categories = []
            gname_lower = graph_name.lower()
            for rid in g["roots"]:
                if g["nodes"][rid].lower() == gname_lower:
                    for cid in g["adj"].get(rid, []):
                        categories.append(g["nodes"][cid])
                else:
                    categories.append(g["nodes"][rid])
            result.append({"topic": graph_name, "categories": categories})
        return result

    def get_children(self, path: str) -> dict:
        """Get immediate children of a node identified by label path."""
        graph_name, node_id = self._resolve_path(path)
        if graph_name is None:
            return {"error": f"Topic not found: {path}"}

        graph = self._graphs[graph_name]

        if node_id is None:
            if "/" in path:
                return {"error": f"Node not found at path: {path}"}
            children = self._top_level_children(graph_name, graph)
            return {"path": path, "children": children}

        children = [
            {
                "label": graph["nodes"][cid],
                "has_children": bool(graph["adj"].get(cid)),
            }
            for cid in graph["adj"].get(node_id, [])
        ]
        return {
            "path": path,
            "node": graph["nodes"][node_id],
            "children": children,
        }

    def get_subtree(self, path: str, depth: int = 2) -> dict:
        """Get a subtree rooted at the given path, up to *depth* levels."""
        graph_name, node_id = self._resolve_path(path)
        if graph_name is None:
            return {"error": f"Topic not found: {path}"}

        graph = self._graphs[graph_name]

        if node_id is None:
            if "/" in path:
                return {"error": f"Node not found at path: {path}"}
            subtree = {}
            for rid in graph["roots"]:
                subtree[graph["nodes"][rid]] = self._subtree_dict(graph, rid, depth)
            return {"path": path, "subtree": subtree}

        return {
            "path": path,
            "node": graph["nodes"][node_id],
            "subtree": self._subtree_dict(graph, node_id, depth),
        }

    def search_nodes(self, query: str) -> list[dict]:
        """Search for nodes whose labels contain the query string."""
        query_lower = query.lower()
        results = []
        for graph_name, graph in self._graphs.items():
            for nid, label in graph["nodes"].items():
                if query_lower in label.lower():
                    results.append({
                        "label": label,
                        "path": self._build_breadcrumb(graph_name, graph, nid),
                        "has_children": bool(graph["adj"].get(nid)),
                    })
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
        return results

    def get_node_path(self, query: str) -> list[dict]:
        """Find nodes matching query and return their full breadcrumb paths."""
        query_lower = query.lower()
        results = []
        for graph_name, graph in self._graphs.items():
            for nid, label in graph["nodes"].items():
                if query_lower in label.lower():
                    results.append({
                        "label": label,
                        "breadcrumb": self._build_breadcrumb(graph_name, graph, nid),
                    })
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _top_level_children(self, graph_name: str, graph: dict) -> list[dict]:
        children = []
        gname_lower = graph_name.lower()
        for rid in graph["roots"]:
            if graph["nodes"][rid].lower() == gname_lower:
                for cid in graph["adj"].get(rid, []):
                    children.append({
                        "label": graph["nodes"][cid],
                        "has_children": bool(graph["adj"].get(cid)),
                    })
            else:
                children.append({
                    "label": graph["nodes"][rid],
                    "has_children": bool(graph["adj"].get(rid)),
                })
        return children

    def _subtree_dict(self, graph: dict, node_id: str, depth: int) -> dict:
        if depth <= 0:
            child_ids = graph["adj"].get(node_id, [])
            if child_ids:
                return {"_truncated": f"{len(child_ids)} children (increase depth)"}
            return {}

        result = {}
        for cid in graph["adj"].get(node_id, []):
            label = graph["nodes"][cid]
            child_subtree = self._subtree_dict(graph, cid, depth - 1)
            result[label] = child_subtree if child_subtree else None
        return result
