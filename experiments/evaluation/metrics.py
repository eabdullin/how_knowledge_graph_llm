"""Aggregate metrics across experiment runs."""

import json
import statistics
from pathlib import Path

from ..config import ConversationLog


JUDGE_SCORE_KEYS = [
    "information_recall_score",
    "information_precision_score",
    "information_redundancy_score",
    "total_reward",
]


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate evaluation results across multiple runs.

    Each result dict should have:
        - modality, model, problem_id
        - coverage: dict from coverage.py
        - judge: dict from llm_judge.py
        - geval: dict from llm_judge.py (optional)
        - num_turns: int
        - total_tool_calls: int
    """
    if not results:
        return {}

    # Group by (modality, model)
    groups = {}
    for r in results:
        key = (r["modality"], r["model"])
        groups.setdefault(key, []).append(r)

    summary = {}
    for (modality, model), runs in groups.items():
        label = f"{modality}/{model}"

        # Coverage metrics
        coverages = [r["coverage"] for r in runs if r.get("coverage") and "error" not in r.get("coverage", {})]
        geval_scores = [r["geval"] for r in runs if r.get("geval") and "error" not in r.get("geval", {})]

        entry = {
            "modality": modality,
            "model": model,
            "n_runs": len(runs),
        }

        # Efficiency
        successful = [r for r in runs if "error" not in r]
        turn_vals = [r.get("num_turns", 0) for r in successful]
        tool_call_vals = [r.get("total_tool_calls", 0) for r in successful]
        conversation_token_vals = [
            r.get("usage_totals", {}).get("total_tokens", 0) for r in successful
        ]
        reasoning_token_vals = [
            r.get("usage_totals", {}).get("reasoning_tokens", 0) for r in successful
        ]
        evaluation_token_vals = [
            r.get("evaluation_usage_totals", {}).get("total_tokens", 0) for r in successful
        ]
        entry["avg_turns"] = _safe_mean(turn_vals)
        entry["std_turns"] = _safe_std(turn_vals)
        entry["avg_tool_calls"] = _safe_mean(tool_call_vals)
        entry["std_tool_calls"] = _safe_std(tool_call_vals)
        entry["avg_conversation_tokens"] = _safe_mean(conversation_token_vals)
        entry["std_conversation_tokens"] = _safe_std(conversation_token_vals)
        entry["avg_reasoning_tokens"] = _safe_mean(reasoning_token_vals)
        entry["std_reasoning_tokens"] = _safe_std(reasoning_token_vals)
        entry["avg_evaluation_tokens"] = _safe_mean(evaluation_token_vals)
        entry["std_evaluation_tokens"] = _safe_std(evaluation_token_vals)
        entry["completion_rate"] = _safe_mean(
            [1 if r.get("completed", False) else 0 for r in runs]
        )

        # Coverage
        if coverages:
            node_coverage_vals = [c["node_coverage"] for c in coverages]
            breadth_coverage_vals = [c["breadth_coverage"] for c in coverages]
            depth_reached_vals = [c["avg_depth_reached"] for c in coverages]
            entry["avg_node_coverage"] = _safe_mean(node_coverage_vals)
            entry["std_node_coverage"] = _safe_std(node_coverage_vals)
            entry["avg_breadth_coverage"] = _safe_mean(breadth_coverage_vals)
            entry["std_breadth_coverage"] = _safe_std(breadth_coverage_vals)
            entry["avg_depth_reached"] = _safe_mean(depth_reached_vals)
            entry["std_depth_reached"] = _safe_std(depth_reached_vals)
            # Exploration pattern distribution
            patterns = [c.get("exploration_pattern", "unknown") for c in coverages]
            entry["exploration_patterns"] = {
                p: patterns.count(p) / len(patterns)
                for p in set(patterns)
            }

        # Judge scores
        for key in JUDGE_SCORE_KEYS:
            vals = []
            for run in runs:
                value = mean_judge_metric(run, key)
                if value is not None:
                    vals.append(value)
            if vals:
                entry[f"avg_{key}"] = _safe_mean(vals)
                entry[f"std_{key}"] = _safe_std(vals)

        # G-Eval scores
        if geval_scores:
            for criterion in ["coherence", "consistency", "fluency", "relevance", "engagingness"]:
                vals = []
                for g in geval_scores:
                    if criterion in g and isinstance(g[criterion], dict):
                        vals.append(g[criterion].get("score", 0))
                if vals:
                    entry[f"avg_geval_{criterion}"] = _safe_mean(vals)
                    entry[f"std_geval_{criterion}"] = _safe_std(vals)

        summary[label] = entry

    return summary


def results_to_table(summary: dict) -> str:
    """Format aggregated results as a readable table."""
    if not summary:
        return "No results to display."

    headers = [
        "Condition", "Runs", "Turns", "Coverage", "Breadth",
        "Recall", "Precision", "Redundancy", "Total", "Tokens", "Reasoning", "Pattern",
    ]
    rows = []
    for label, s in sorted(summary.items()):
        rows.append([
            label,
            str(s.get("n_runs", 0)),
            _format_mean_std(s.get("avg_turns", 0), s.get("std_turns", 0), precision=1),
            _format_mean_std(s.get("avg_node_coverage", 0), s.get("std_node_coverage", 0), percent=True),
            f"{s.get('avg_breadth_coverage', 0):.2%}",
            _format_mean_std(s.get("avg_information_recall_score", 0), s.get("std_information_recall_score", 0), precision=1),
            _format_mean_std(s.get("avg_information_precision_score", 0), s.get("std_information_precision_score", 0), precision=1),
            _format_mean_std(s.get("avg_information_redundancy_score", 0), s.get("std_information_redundancy_score", 0), precision=1),
            _format_mean_std(s.get("avg_total_reward", 0), s.get("std_total_reward", 0), precision=1),
            _format_mean_std(s.get("avg_conversation_tokens", 0), s.get("std_conversation_tokens", 0), precision=0),
            _format_mean_std(s.get("avg_reasoning_tokens", 0), s.get("std_reasoning_tokens", 0), precision=0),
            _top_pattern(s.get("exploration_patterns", {})),
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = " | ".join("-" * w for w in col_widths)
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    data_lines = [
        " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths))
        for row in rows
    ]

    return "\n".join([header_line, sep] + data_lines)


def save_results(results: list[dict], output_path: Path):
    """Save raw results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _safe_mean(vals):
    return statistics.mean(vals) if vals else 0.0


def _safe_std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def mean_judge_metric(result: dict, key: str) -> float | None:
    vals = []
    for field in ["judge_summary"]:
        judge_result = result.get(field, {})
        if not isinstance(judge_result, dict) or judge_result.get("error"):
            continue
        value = judge_metric_value(judge_result, key)
        if value is not None:
            vals.append(value)
    if not vals:
        return None
    return _safe_mean(vals)


def judge_metric_value(judge_result: dict, key: str) -> float | None:
    if key == "total_reward":
        return judge_total_reward(judge_result)
    return _coerce_float(judge_result.get(key))


def judge_total_reward(judge_result: dict | None) -> float | None:
    if not isinstance(judge_result, dict) or judge_result.get("error"):
        return None

    total = 0.0
    for key in [
        "information_recall_score",
        "information_precision_score",
        "information_redundancy_score",
    ]:
        value = _coerce_float(judge_result.get(key))
        if value is None:
            return None
        total += value
    return total


def _coerce_float(value) -> float | None:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_pattern(patterns: dict) -> str:
    if not patterns:
        return "n/a"
    return max(patterns, key=patterns.get)


def _format_mean_std(mean_val, std_val, *, precision: int = 1, percent: bool = False) -> str:
    if percent:
        return f"{mean_val:.{precision}%} +- {std_val:.{precision}%}"
    return f"{mean_val:.{precision}f} +- {std_val:.{precision}f}"
