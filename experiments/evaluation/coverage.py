"""KG coverage metrics: measure how well the modeller explored the knowledge graph."""

import re
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from knowledge_graph import KnowledgeGraph

from ..config import ConversationLog


def compute_coverage(
    conversation: ConversationLog,
    kg_text: str,
    kg_topic: str,
) -> dict:
    """Compute KG coverage metrics for a conversation.

    Returns dict with:
        - node_coverage: fraction of KG nodes mentioned in conversation
        - depth_coverage: max depth reached per branch
        - breadth_coverage: fraction of top-level branches touched
        - nodes_mentioned: list of matched node labels
        - nodes_missed: list of unmatched node labels
        - exploration_pattern: "dfs", "bfs", or "mixed"
    """
    kg = KnowledgeGraph(kg_text)
    graph_name, graph = _find_graph(kg, kg_topic)
    if graph is None:
        return {"error": f"Graph '{kg_topic}' not found"}

    # Collect all node labels and their depths
    nodes = graph["nodes"]
    adj = graph["adj"]
    roots = graph["roots"]

    # Build parent map and compute depths
    parent_map = {}
    for nid, children in adj.items():
        for cid in children:
            parent_map[cid] = nid

    depths = {}
    for nid in nodes:
        depth = 0
        current = nid
        while current in parent_map:
            current = parent_map[current]
            depth += 1
        depths[nid] = depth

    max_depth = max(depths.values()) if depths else 0

    explicit_mentions = _collect_explicit_mentions(conversation, graph_name, nodes, parent_map)

    # Get all conversation text (modeller + user)
    all_text = " ".join(
        t.content.lower() for t in conversation.turns if t.content
    )

    # Also include tool call results if available
    tool_texts = []
    for tc_detail in conversation.metadata.get("tool_calls_detail", []):
        if "result" in tc_detail:
            tool_texts.append(tc_detail["result"].lower())

    # Match nodes against conversation text
    mentioned = set(explicit_mentions)
    mention_order = []
    for nid in explicit_mentions:
        mention_order.append(nid)
    for nid, label in nodes.items():
        if nid in mentioned:
            continue
        # Normalize label for matching
        label_lower = label.lower().strip()
        # Skip very short or purely symbolic labels
        if len(label_lower) < 3 or label_lower.startswith("\\") or label_lower.startswith("{"):
            # Check for exact substring match for short labels
            if label_lower in all_text:
                mentioned.add(nid)
                mention_order.append(nid)
            continue

        # Match meaningful words from the label
        words = re.findall(r'[a-z]+', label_lower)
        meaningful_words = [w for w in words if len(w) > 2]
        if meaningful_words and all(w in all_text for w in meaningful_words):
            mentioned.add(nid)
            mention_order.append(nid)

    # Node coverage
    total_nodes = len(nodes)
    node_coverage = len(mentioned) / total_nodes if total_nodes > 0 else 0

    # Breadth coverage: fraction of top-level branches explored
    # Get top-level children (direct children of roots)
    top_level = set()
    gname_lower = graph_name.lower()
    for rid in roots:
        if nodes[rid].lower() == gname_lower:
            top_level.update(adj.get(rid, []))
        else:
            top_level.add(rid)

    top_touched = sum(
        1 for tlid in top_level
        if tlid in mentioned or _any_descendant_mentioned(adj, tlid, mentioned)
    )
    breadth_coverage = top_touched / len(top_level) if top_level else 0

    # Depth coverage per branch
    branch_max_depths = {}
    for tlid in top_level:
        branch_label = nodes[tlid]
        branch_mentioned = {
            nid for nid in mentioned
            if _is_descendant(adj, parent_map, tlid, nid) or nid == tlid
        }
        if branch_mentioned:
            branch_max_depths[branch_label] = max(
                depths[nid] for nid in branch_mentioned
            )
        else:
            branch_max_depths[branch_label] = 0

    avg_depth = (
        sum(branch_max_depths.values()) / len(branch_max_depths)
        if branch_max_depths else 0
    )

    # Exploration pattern analysis
    pattern = _classify_exploration_pattern(mention_order, depths, adj, parent_map)

    # Compile missed nodes (only meaningful ones)
    missed = []
    for nid in nodes:
        if nid not in mentioned:
            label = nodes[nid]
            words = re.findall(r'[a-z]+', label.lower())
            if any(len(w) > 2 for w in words):
                missed.append(label)

    return {
        "node_coverage": round(node_coverage, 4),
        "coverage_source": "explicit_refs" if explicit_mentions else "text_fallback",
        "breadth_coverage": round(breadth_coverage, 4),
        "avg_depth_reached": round(avg_depth, 2),
        "max_possible_depth": max_depth,
        "branch_depths": branch_max_depths,
        "explicit_nodes_mentioned": [nodes[nid] for nid in explicit_mentions],
        "nodes_mentioned_count": len(mentioned),
        "total_nodes": total_nodes,
        "nodes_mentioned": [nodes[nid] for nid in mentioned],
        "nodes_missed": missed[:30],  # Cap for readability
        "exploration_pattern": pattern,
    }


def _find_graph(kg: KnowledgeGraph, topic: str):
    topic_lower = topic.lower()
    for gname, g in kg._graphs.items():
        if gname.lower() == topic_lower:
            return gname, g
    return None, None


def _any_descendant_mentioned(adj, root_id, mentioned):
    """Check if any descendant of root_id is in mentioned."""
    stack = list(adj.get(root_id, []))
    while stack:
        nid = stack.pop()
        if nid in mentioned:
            return True
        stack.extend(adj.get(nid, []))
    return False


def _is_descendant(adj, parent_map, ancestor_id, node_id):
    """Check if node_id is a descendant of ancestor_id."""
    current = node_id
    while current in parent_map:
        current = parent_map[current]
        if current == ancestor_id:
            return True
    return False


def _classify_exploration_pattern(mention_order, depths, adj, parent_map):
    """Classify exploration as DFS, BFS, or mixed based on mention order."""
    if len(mention_order) < 3:
        return "insufficient_data"

    # Compare consecutive mention depths
    depth_changes = []
    for i in range(1, len(mention_order)):
        d_prev = depths.get(mention_order[i - 1], 0)
        d_curr = depths.get(mention_order[i], 0)
        depth_changes.append(d_curr - d_prev)

    # DFS: tends to go deeper before backtracking (positive changes)
    # BFS: tends to stay at same depth (zero changes)
    deeper = sum(1 for d in depth_changes if d > 0)
    same = sum(1 for d in depth_changes if d == 0)
    shallower = sum(1 for d in depth_changes if d < 0)
    total = len(depth_changes)

    if total == 0:
        return "insufficient_data"

    if deeper / total > 0.4:
        return "dfs"
    elif same / total > 0.5:
        return "bfs"
    else:
        return "mixed"


def _collect_explicit_mentions(conversation, graph_name, nodes, parent_map):
    mentioned = []
    breadcrumbs = {
        nid: _node_breadcrumb(graph_name, nodes, parent_map, nid)
        for nid in nodes
    }

    for turn in conversation.turns:
        for ref in getattr(turn, "kg_items_used", []) or []:
            matched = _match_reference_to_node(ref, nodes, breadcrumbs)
            if matched is not None and matched not in mentioned:
                mentioned.append(matched)
    return mentioned


def _node_breadcrumb(graph_name, nodes, parent_map, node_id):
    parts = [nodes[node_id]]
    current = node_id
    while current in parent_map:
        current = parent_map[current]
        parts.append(nodes[current])
    parts.append(graph_name)
    return " > ".join(reversed(parts))


def _normalize_ref(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip(" />")


def _match_reference_to_node(ref, nodes, breadcrumbs):
    ref_norm = _normalize_ref(ref.replace("/", " > "))
    if not ref_norm:
        return None

    breadcrumb_matches = [
        nid for nid, breadcrumb in breadcrumbs.items()
        if _normalize_ref(breadcrumb).endswith(ref_norm)
    ]
    if len(breadcrumb_matches) == 1:
        return breadcrumb_matches[0]

    label_matches = [
        nid for nid, label in nodes.items()
        if _normalize_ref(label) == ref_norm
    ]
    if len(label_matches) == 1:
        return label_matches[0]

    contains_matches = [
        nid for nid, breadcrumb in breadcrumbs.items()
        if ref_norm in _normalize_ref(breadcrumb)
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]

    return None
