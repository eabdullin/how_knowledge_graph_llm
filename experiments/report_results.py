"""Merge modality shards and generate a report for a run directory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from .evaluation.metrics import aggregate_results, judge_total_reward, mean_judge_metric


DEFAULT_MODALITIES = ["image", "text", "tools"]
OPTIONAL_MODALITIES = ["no_graph"]
EXCLUDED_REPORT_MODELS = {"gpt-oss-20b"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--runs-per-problem", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    rows, shard_rows = load_shards(run_dir)
    rows = filter_report_rows(rows)
    shard_rows = {
        key: filtered_rows
        for key, shard in shard_rows.items()
        if (filtered_rows := filter_report_rows(shard))
    }
    rows.sort(key=lambda row: (row.get("modality", ""), row.get("problem_id", ""), row.get("experiment_id", "")))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_results.json").write_text(json.dumps(rows, indent=2, default=str))

    summary = aggregate_results(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    baseline_rows = []
    baseline_summary = {}
    if args.baseline_dir:
        baseline_rows = filter_report_rows(load_raw_results(args.baseline_dir))
        baseline_summary = aggregate_results(baseline_rows)

    report = build_report(
        run_dir=run_dir,
        rows=rows,
        shard_rows=shard_rows,
        summary=summary,
        baseline_rows=baseline_rows,
        baseline_summary=baseline_summary,
        expected_total=args.expected_total,
        runs_per_problem=args.runs_per_problem,
        max_turns=args.max_turns,
        model=args.model,
    )
    (run_dir / "report.md").write_text(report)

    if args.require_complete and len(rows) != args.expected_total:
        raise SystemExit(f"Expected {args.expected_total} rows, found {len(rows)}")

    print(f"merged rows: {len(rows)}")
    print(f"wrote: {run_dir / 'raw_results.json'}")
    print(f"wrote: {run_dir / 'summary.json'}")
    print(f"wrote: {run_dir / 'report.md'}")


def load_shards(run_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = []
    shard_rows = {}
    direct_path = run_dir / "raw_results.json"
    if direct_path.exists():
        direct_rows = json.loads(direct_path.read_text())
        return direct_rows, {"direct": direct_rows}

    shards_root = run_dir / "shards"
    if not shards_root.exists():
        return rows, shard_rows

    direct_matches = []
    for modality in DEFAULT_MODALITIES + OPTIONAL_MODALITIES:
        path = shards_root / modality / "raw_results.json"
        if path.exists():
            direct_matches.append((modality, path))

    if direct_matches:
        for modality, path in direct_matches:
            shard_data = json.loads(path.read_text())
            shard_rows[modality] = shard_data
            rows.extend(shard_data)
        return rows, shard_rows

    for path in sorted(shards_root.glob("**/raw_results.json")):
        relative_parent = path.parent.relative_to(shards_root)
        label = str(relative_parent).replace("/", " :: ")
        shard_data = json.loads(path.read_text())
        shard_rows[label] = shard_data
        rows.extend(shard_data)
    return rows, shard_rows


def load_raw_results(run_dir: Path) -> list[dict]:
    raw_path = run_dir / "raw_results.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text())
    rows, _ = load_shards(run_dir)
    return rows


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "experiment_manifest.json"
    if path.exists():
        return json.loads(path.read_text())

    shard_paths = sorted((run_dir / "shards").glob("*/*/experiment_manifest.json"))
    if not shard_paths:
        return {}

    manifests = [json.loads(shard_path.read_text()) for shard_path in shard_paths]
    models = sorted({model for manifest in manifests for model in manifest.get("models", [])})
    modalities = sorted({modality for manifest in manifests for modality in manifest.get("modalities", [])})
    problem_ids = sorted({problem_id for manifest in manifests for problem_id in manifest.get("problem_ids", [])})

    problems_by_source = defaultdict(set)
    problems_by_topic = defaultdict(set)
    for manifest in manifests:
        for source, ids in manifest.get("problems_by_source", {}).items():
            problems_by_source[source].update(ids)
        for topic, ids in manifest.get("problems_by_topic", {}).items():
            problems_by_topic[topic].update(ids)

    user_models = sorted({manifest.get("user_model") for manifest in manifests if manifest.get("user_model")})
    judge_models = sorted({manifest.get("judge_model") for manifest in manifests if manifest.get("judge_model")})

    return {
        "models": models,
        "modalities": modalities,
        "problem_ids": problem_ids,
        "problem_count": len(problem_ids),
        "problems_by_source": {
            source: sorted(ids) for source, ids in sorted(problems_by_source.items())
        },
        "problems_by_topic": {
            topic: sorted(ids) for topic, ids in sorted(problems_by_topic.items())
        },
        "user_model": ", ".join(user_models) if user_models else "unknown",
        "judge_model": ", ".join(judge_models) if judge_models else "unknown",
    }


def build_report(
    *,
    run_dir: Path,
    rows: list[dict],
    shard_rows: dict[str, list[dict]],
    summary: dict,
    baseline_rows: list[dict],
    baseline_summary: dict,
    expected_total: int,
    runs_per_problem: int,
    max_turns: int,
    model: str,
) -> str:
    rows = filter_report_rows(rows)
    shard_rows = {
        key: filter_report_rows(shard)
        for key, shard in shard_rows.items()
        if filter_report_rows(shard)
    }
    summary = {
        key: value for key, value in summary.items()
        if value.get("model") not in EXCLUDED_REPORT_MODELS
    }
    baseline_rows = filter_report_rows(baseline_rows)
    baseline_summary = {
        key: value for key, value in baseline_summary.items()
        if value.get("model") not in EXCLUDED_REPORT_MODELS
    }

    manifest = load_manifest(run_dir)
    problem_ids = sorted(manifest.get("problem_ids", unique_problem_ids(rows)))
    topics = manifest.get("problems_by_topic", group_problem_ids(rows, "kg_topic"))
    sources = manifest.get("problems_by_source", group_problem_ids(rows, "problem_source"))
    modalities = manifest.get("modalities", list(shard_rows))
    filtered_models = filter_report_models(manifest.get("models", [model]))
    condition_keys = report_condition_keys(shard_rows, modalities)
    completed = sum(1 for row in rows if row.get("completed"))
    errors = sum(1 for row in rows if row.get("error"))
    parse_failures = judge_parse_failures(rows)
    dialogue_counts = dialogue_file_counts(run_dir)
    modality_summary = aggregate_by_modality(rows)
    topic_summary = aggregate_by_topic(rows, list(topics))
    problem_summary = aggregate_by_problem(rows)
    modality_reward_matrix = judge_reward_matrix(rows, "modality", modalities, filtered_models)
    modality_topic_model_reward_matrix = judge_reward_matrix(
        add_combined_key(rows, "modality_topic", "modality", "kg_topic"),
        "modality_topic",
        combined_label_order(modalities, list(topics)),
        filtered_models,
    )
    topic_reward_matrix = judge_reward_matrix(rows, "kg_topic", list(topics), filtered_models)
    problem_reward_matrix = judge_reward_matrix(rows, "problem_id", problem_ids, filtered_models)
    no_graph_note = (
        " The `no_graph` condition is a control where the modeller receives no knowledge graph text, image, or tools."
        if "no_graph" in modalities else ""
    )
    modeller_models = ", ".join(filtered_models)
    user_model = manifest.get("user_model", "unknown")
    judge_model = manifest.get("judge_model", "unknown")
    evaluation_status = describe_evaluation_status(rows)
    evaluation_phrase = judge_model
    if evaluation_status["active_label"]:
        evaluation_phrase = f"{judge_model}` originally; active re-evaluation uses `{evaluation_status['active_label']}"

    lines = [
        "# Knowledge Graph Modality Experiment Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"This run evaluates knowledge-graph modality effects after tightening the dialogue protocol to emphasize elicitation over assumption-filling. The modeller uses `{modeller_models}`, the simulated user uses `{user_model}`, and evaluation uses `{evaluation_phrase}`. The modeller prompt also forbids leading assumption-confirmation questions and repeated re-asking after unknown answers." + no_graph_note,
        "",
        f"Saved conversations: {len(rows)} / {expected_total}. Completed: {completed}. Errors: {errors}.",
        evaluation_status["summary_line"],
        "This report is an intermediate checkpoint; aggregate metrics will continue to move until all shards finish."
        if len(rows) < expected_total
        else "All target conversations are present in this report.",
        "",
        "## Experimental Design",
        "",
        "| Factor | Value |",
        "|---|---|",
        f"| Model | `{modeller_models}` |",
        f"| Modalities | {', '.join(modalities)} |",
        f"| Problems | {len(problem_ids)} total |",
        f"| Runs per problem/modality | {runs_per_problem} |",
        f"| Max turns | {max_turns} |",
        f"| Target conversations | {expected_total} |",
        "",
        (
            "The matrix was parallelized by modality and modeller model, with each shard writing its own raw results and dialogue files before the final merge."
            if uses_nested_conditions(condition_keys)
            else "The matrix was parallelized by modality, with each shard writing its own raw results and dialogue files before the final merge."
        ),
        "",
        "### Problem Sources",
        "",
        "| Source | Unique problems |",
        "|---|---:|",
    ]
    for source, ids in sorted(sources.items()):
        lines.append(f"| {source} | {len(ids)} |")

    lines.extend([
        "",
        "### KG Topics",
        "",
        "| Topic | Unique problems |",
        "|---|---:|",
    ])
    for topic, ids in sorted(topics.items()):
        lines.append(f"| {topic} | {len(ids)} |")

    lines.extend([
        "",
        "## Progress And Artifacts",
        "",
    ])
    topic_headers = [topic for topic, ids in sorted(topics.items()) if ids]
    lines.append(
        "| Condition | Saved | Completed | Errors | "
        + " | ".join(topic_headers)
        + " | Dialogue Markdown | Dialogue JSON |"
    )
    lines.append(
        "|---|---:|---:|---:|"
        + "".join("---:|" for _ in topic_headers)
        + "---:|---:|"
    )
    expected_per_condition = len(problem_ids) * runs_per_problem
    for condition in condition_keys:
        shard = shard_rows.get(condition, [])
        dialogue = dialogue_counts.get(condition, {"md": 0, "json": 0})
        topic_progress_cells = [
            format_topic_cell(shard, topic, len(topic_problem_ids) * runs_per_problem)
            for topic, topic_problem_ids in sorted(topics.items())
            if topic_problem_ids
        ]
        lines.append(
            f"| {condition_label(condition)} | {len(shard)} / {expected_per_condition} | "
            f"{sum(1 for row in shard if row.get('completed'))} | "
            f"{sum(1 for row in shard if row.get('error'))} | "
            + " | ".join(topic_progress_cells)
            + " | "
            f"{dialogue['md']} | {dialogue['json']} |"
        )

    total_topic_progress_cells = [
        format_topic_cell(rows, topic, len(topic_problem_ids) * runs_per_problem * len(condition_keys))
        for topic, topic_problem_ids in sorted(topics.items())
        if topic_problem_ids
    ]
    lines.extend([
        f"| Total | {len(rows)} / {expected_total} | {completed} | {errors} | "
        + " | ".join(total_topic_progress_cells)
        + " | "
        f"{sum(item['md'] for item in dialogue_counts.values())} | {sum(item['json'] for item in dialogue_counts.values())} |",
    ])

    if evaluation_status["details"]:
        lines.extend([
            "",
            "### Re-Evaluation Checkpoint",
            "",
            evaluation_status["details"],
        ])

    lines.extend([
        "",
        "Artifacts:",
        "",
        "| Artifact | Path |",
        "|---|---|",
        f"| Combined raw results | `{display_path(run_dir / 'raw_results.json')}` |",
        f"| Combined summary | `{display_path(run_dir / 'summary.json')}` |",
        f"| Report | `{display_path(run_dir / 'report.md')}` |",
        "",
        "## Aggregate Results By Modality",
        "",
        "Values below are shown as mean +- std across conversation runs after pooling all modeller models within each modality.",
        "",
        "| Modality | Runs | Turns | Tool calls | Conversation tokens | Reasoning tokens | Evaluation tokens | Node coverage | Mean judge reward | G-Eval coherence | Completion rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, item in sorted(modality_summary.items()):
        lines.append(
            f"| {condition_label(label)} | {item.get('n_runs', 0)} | "
            f"{format_mean_std(item.get('avg_turns', 0), item.get('std_turns', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_tool_calls', 0), item.get('std_tool_calls', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_conversation_tokens', 0), item.get('std_conversation_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_reasoning_tokens', 0), item.get('std_reasoning_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_evaluation_tokens', 0), item.get('std_evaluation_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_node_coverage', 0), item.get('std_node_coverage', 0), precision=1, percent=True)} | "
            f"{format_mean_std(item.get('avg_total_reward', 0), item.get('std_total_reward', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_geval_coherence', 0), item.get('std_geval_coherence', 0), precision=2)} | "
            f"{item.get('completion_rate', 0):.0%} |"
        )

    lines.extend([
        "",
        "## Judge Reward Matrix By Modality",
        "",
        "Cells show mean of dialogue-judge and summary-judge reward for each modality/model slice in the current filtered report view.",
        "",
        build_reward_matrix_table("Modality", modality_reward_matrix, filtered_models, condition_label),
        "",
        "## Judge Reward Matrix By Modality Split With KG Topic",
        "",
        "Cells show mean of dialogue-judge and summary-judge reward for each modality/topic slice for each modeller model.",
        "",
        build_reward_matrix_table(
            "Modality / KG Topic",
            modality_topic_model_reward_matrix,
            filtered_models,
            modality_topic_label,
        ),
    ])

    lines.extend([
        "",
        "## Aggregate Results By Condition",
        "",
        "Values below are shown as mean +- std across conversation runs for each modality/model condition.",
        "",
        "| Condition | Runs | Turns | Tool calls | Conversation tokens | Reasoning tokens | Evaluation tokens | Node coverage | Mean judge reward | G-Eval coherence | Completion rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, item in sorted(summary.items()):
        lines.append(
            f"| {condition_label(label)} | {item.get('n_runs', 0)} | "
            f"{format_mean_std(item.get('avg_turns', 0), item.get('std_turns', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_tool_calls', 0), item.get('std_tool_calls', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_conversation_tokens', 0), item.get('std_conversation_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_reasoning_tokens', 0), item.get('std_reasoning_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_evaluation_tokens', 0), item.get('std_evaluation_tokens', 0), precision=0, use_commas=True)} | "
            f"{format_mean_std(item.get('avg_node_coverage', 0), item.get('std_node_coverage', 0), precision=1, percent=True)} | "
            f"{format_mean_std(item.get('avg_total_reward', 0), item.get('std_total_reward', 0), precision=1)} | "
            f"{format_mean_std(item.get('avg_geval_coherence', 0), item.get('std_geval_coherence', 0), precision=2)} | "
            f"{item.get('completion_rate', 0):.0%} |"
        )

    lines.extend([
        "",
        f"Dialogue-judge parse failures: {parse_failures['dialogue']}. Summary-judge parse failures: {parse_failures['summary']}.",
        "",
        "## Aggregate Results By KG Topic",
        "",
        "Values below are shown as mean values after pooling all currently included models, modalities, and problems within each KG topic.",
        "",
        "| KG Topic | Runs | Avg turns | Avg tool calls | Avg conv tokens | Avg reasoning tokens | Avg node coverage | Avg mean judge reward | Completion rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in topic_summary:
        lines.append(
            f"| {item['kg_topic']} | {item['runs']} | {item['turns']:.1f} | "
            f"{item['tool_calls']:.1f} | {item['tokens']:,.0f} | {item['reasoning']:,.0f} | "
            f"{item['coverage']:.1%} | {item['reward']:.1f} | {item['completion']:.0%} |"
        )

    lines.extend([
        "",
        "## Judge Reward Matrix By KG Topic",
        "",
        "Cells show mean of dialogue-judge and summary-judge reward for each KG topic/model slice after pooling across included modalities and problems.",
        "",
        build_reward_matrix_table("KG Topic", topic_reward_matrix, filtered_models),
        "",
        "## Aggregate Results By Problem",
        "",
        "Values below are shown as mean values after pooling all currently included models and modalities for each problem.",
        "",
        "| Problem | Runs | Avg turns | Avg tool calls | Avg conv tokens | Avg reasoning tokens | Avg node coverage | Avg mean judge reward | Completion rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in problem_summary:
        lines.append(
            f"| {item['problem_id']} | {item['runs']} | {item['turns']:.1f} | "
            f"{item['tool_calls']:.1f} | {item['tokens']:,.0f} | {item['reasoning']:,.0f} | "
            f"{item['coverage']:.1%} | {item['reward']:.1f} | {item['completion']:.0%} |"
        )

    lines.extend([
        "",
        "## Judge Reward Matrix By Problem",
        "",
        "Cells show mean of dialogue-judge and summary-judge reward for each problem/model slice after pooling across included modalities.",
        "",
        build_reward_matrix_table("Problem", problem_reward_matrix, filtered_models),
    ])

    lines.extend([
        "",
        "## Per-Problem Results",
        "",
        "| Model | Modality | Problem | Runs | Avg turns | Avg tool calls | Avg conv tokens | Avg reasoning tokens | Avg node coverage | Avg mean judge reward | Completion rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in per_problem_summary(rows):
        lines.append(
            f"| {item['model']} | {item['modality'].title()} | {item['problem_id']} | {item['runs']} | {item['turns']:.1f} | "
            f"{item['tool_calls']:.1f} | {item['tokens']:,.0f} | {item['reasoning']:,.0f} | "
            f"{item['coverage']:.1%} | {item['reward']:.1f} | {item['completion']:.0%} |"
        )

    lines.extend([
        "",
        "## Token Usage Totals",
        "",
        "| Condition | Conversation tokens | Reasoning tokens | Evaluation tokens | Tool calls |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in condition_keys:
        modality_rows = shard_rows.get(condition, [])
        lines.append(
            f"| {condition_label(condition)} | {sum_usage(modality_rows, 'usage_totals', 'total_tokens'):,} | "
            f"{sum_usage(modality_rows, 'usage_totals', 'reasoning_tokens'):,} | "
            f"{sum_usage(modality_rows, 'evaluation_usage_totals', 'total_tokens'):,} | "
            f"{sum(row.get('total_tool_calls', 0) for row in modality_rows):,} |"
        )

    if baseline_summary:
        lines.extend(compare_baseline(summary, baseline_summary, len(rows), len(baseline_rows)))

    lines.extend([
        "",
        "## Additional Analysis",
        "",
        f"Explicit-ref coverage rows: {sum(1 for row in rows if row.get('coverage', {}).get('coverage_source') == 'explicit_refs')} / {len(rows)}.",
        f"Rows with exactly one `load_knowledge_graph` call: {sum(one_load_graph(row) for row in rows if row.get('modality') == 'tools')} / {sum(1 for row in rows if row.get('modality') == 'tools')}.",
        "",
        "Suggested follow-up analyses:",
        "",
        "- Compare topic effects within each modality rather than only after pooling across modalities.",
        "- Plot explicit KG-reference count per modeller turn against judge recall and redundancy.",
        "- Inspect whether longer tool traversals improve coverage or only increase token cost.",
        "- Compare the control (`no_graph`) against `text` and `image` separately for curated MAMO problems versus problem-description problems.",
        "",
        "## Validity Notes",
        "",
        "- The previous v2 run remains useful as a diagnostic baseline, but it is confounded by simulator over-disclosure and modeller assumption-confirmation behavior.",
        "- This rerun relies on the stronger simulated-user model plus prompt instructions, not deterministic guardrail overrides, to keep absent facts unknown.",
        "- Stricter elicitation typically increases turns and token usage. Interpret efficiency differences as part modality effect and part cost of the cleaner protocol.",
        "- Tools modality can spend many calls traversing the KG; tool-call counts should be considered alongside coverage and reward.",
        "",
    ])
    return "\n".join(lines)

def describe_evaluation_status(rows: list[dict]) -> dict:
    eligible = [row for row in rows if isinstance(row.get("dialogue"), list) and row["dialogue"]]
    config_counts = defaultdict(int)
    for row in eligible:
        config = row.get("evaluation_config")
        if isinstance(config, dict):
            key = json.dumps(config, sort_keys=True)
            config_counts[key] += 1

    if not config_counts:
        return {
            "active_label": "",
            "summary_line": "No row-level re-evaluation config is recorded yet.",
            "details": "",
        }

    active_key, active_count = max(config_counts.items(), key=lambda item: item[1])
    active_config = json.loads(active_key)
    active_label = active_config.get("judge_model", "unknown")
    eval_version = active_config.get("eval_version")
    temperature = active_config.get("judge_temperature")
    max_tokens = active_config.get("judge_max_tokens")
    api_temperature = active_config.get("judge_temperature_api_parameter")
    if eval_version:
        active_label += f", eval version {eval_version}"
    if temperature is not None:
        active_label += f", requested temperature {temperature}"
    if max_tokens is not None:
        active_label += f", max tokens {max_tokens}"

    summary_line = (
        f"Re-evaluation progress: {active_count} / {len(eligible)} saved dialogues carry the active row-level judge config."
    )
    details = summary_line
    if eval_version:
        details += f" Eval version: `{eval_version}`."
    if api_temperature:
        details += f" Temperature API parameter: `{api_temperature}`."

    other_high = sum(
        1 for row in eligible
        if row.get("judge_model") == active_config.get("judge_model")
        and row.get("evaluation_config") != active_config
    )
    if other_high:
        details += f" {other_high} rows have earlier `{active_config.get('judge_model')}` evaluations with older metadata and are expected to be overwritten as the pass continues."

    errors = sum(1 for row in rows if row.get("evaluation_error"))
    if errors:
        details += f" {errors} row currently records an evaluation error."

    return {
        "active_label": active_label,
        "summary_line": summary_line,
        "details": details,
    }

def per_problem_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("model", ""), row.get("modality", ""), row.get("problem_id", ""))].append(row)

    items = []
    for (model, modality, problem_id), group in sorted(groups.items()):
        items.append({
            "model": model,
            "modality": modality,
            "problem_id": problem_id,
            "runs": len(group),
            "turns": mean(row.get("num_turns", 0) for row in group),
            "tool_calls": mean(row.get("total_tool_calls", 0) for row in group),
            "tokens": mean(row.get("usage_totals", {}).get("total_tokens", 0) for row in group),
            "reasoning": mean(row.get("usage_totals", {}).get("reasoning_tokens", 0) for row in group),
            "coverage": mean(row.get("coverage", {}).get("node_coverage", 0) for row in group),
            "reward": mean(judge_numeric(row) for row in group),
            "completion": mean(1 if row.get("completed") else 0 for row in group),
        })
    return items


def aggregate_by_problem(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("problem_id", "")].append(row)

    items = []
    for problem_id, group in sorted(groups.items()):
        items.append({
            "problem_id": problem_id,
            "runs": len(group),
            "turns": mean(row.get("num_turns", 0) for row in group),
            "tool_calls": mean(row.get("total_tool_calls", 0) for row in group),
            "tokens": mean(row.get("usage_totals", {}).get("total_tokens", 0) for row in group),
            "reasoning": mean(row.get("usage_totals", {}).get("reasoning_tokens", 0) for row in group),
            "coverage": mean(row.get("coverage", {}).get("node_coverage", 0) for row in group),
            "reward": mean(judge_numeric(row) for row in group),
            "completion": mean(1 if row.get("completed") else 0 for row in group),
        })
    return items


def aggregate_by_topic(rows: list[dict], topic_order: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        topic = row.get("kg_topic", "")
        if topic:
            groups[topic].append(row)

    items = []
    for topic in topic_order:
        group = groups.get(topic, [])
        if not group:
            continue
        items.append({
            "kg_topic": topic,
            "runs": len(group),
            "turns": mean(row.get("num_turns", 0) for row in group),
            "tool_calls": mean(row.get("total_tool_calls", 0) for row in group),
            "tokens": mean(row.get("usage_totals", {}).get("total_tokens", 0) for row in group),
            "reasoning": mean(row.get("usage_totals", {}).get("reasoning_tokens", 0) for row in group),
            "coverage": mean(row.get("coverage", {}).get("node_coverage", 0) for row in group),
            "reward": mean(judge_numeric(row) for row in group),
            "completion": mean(1 if row.get("completed") else 0 for row in group),
        })
    return items


def judge_reward_matrix(
    rows: list[dict],
    row_key: str,
    row_order: list[str],
    model_order: list[str],
) -> dict[str, dict[str, float | None]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get(row_key, ""), row.get("model", ""))].append(row)

    matrix: dict[str, dict[str, float | None]] = {}
    for row_label in row_order:
        matrix[row_label] = {}
        for model in model_order:
            group = grouped.get((row_label, model), [])
            matrix[row_label][model] = mean(judge_numeric(row) for row in group) if group else None
    return matrix


def build_reward_matrix_table(
    row_header: str,
    matrix: dict[str, dict[str, float | None]],
    model_order: list[str],
    row_label_formatter=None,
) -> str:
    row_label_formatter = row_label_formatter or (lambda value: value)
    headers = [row_header] + model_order
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in model_order]) + "|",
    ]
    for row_label, values in matrix.items():
        row = [row_label_formatter(row_label)]
        for model in model_order:
            value = values.get(model)
            row.append(f"{value:.1f}" if value is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compare_baseline(summary: dict, baseline_summary: dict, row_count: int, baseline_count: int) -> list[str]:
    lines = [
        "",
        "## Comparison To v2 Baseline",
        "",
        f"Current rows: {row_count}. Baseline rows: {baseline_count}.",
        "",
        "| Modality | v2 turns | Current turns | v2 tokens | Current tokens | v2 reward | Current reward | v2 coverage | Current coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in sorted(summary.items()):
        base = baseline_summary.get(label, {})
        lines.append(
            f"| {condition_label(label)} | {base.get('avg_turns', 0):.1f} | {item.get('avg_turns', 0):.1f} | "
            f"{base.get('avg_conversation_tokens', 0):,.0f} | {item.get('avg_conversation_tokens', 0):,.0f} | "
            f"{base.get('avg_total_reward', 0):.1f} | {item.get('avg_total_reward', 0):.1f} | "
            f"{base.get('avg_node_coverage', 0):.1%} | {item.get('avg_node_coverage', 0):.1%} |"
        )
    return lines


def dialogue_file_counts(run_dir: Path) -> dict[str, dict[str, int]]:
    counts = {}
    shards_root = run_dir / "shards"
    if not shards_root.exists():
        return counts

    direct_found = False
    for modality in DEFAULT_MODALITIES + OPTIONAL_MODALITIES:
        dialogue_dir = shards_root / modality / "dialogues"
        if dialogue_dir.exists():
            direct_found = True
        counts[modality] = {
            "md": len(list(dialogue_dir.glob("*.md"))) if dialogue_dir.exists() else 0,
            "json": len(list(dialogue_dir.glob("*.json"))) if dialogue_dir.exists() else 0,
        }

    if direct_found:
        return counts

    for dialogue_dir in sorted(shards_root.glob("**/dialogues")):
        label = str(dialogue_dir.parent.relative_to(shards_root)).replace("/", " :: ")
        counts[label] = {
            "md": len(list(dialogue_dir.glob("*.md"))),
            "json": len(list(dialogue_dir.glob("*.json"))),
        }
    return counts


def condition_label(raw: str) -> str:
    return raw.replace(" :: ", " / ").replace("_", " ").title()


def modality_topic_label(raw: str) -> str:
    modality, topic = raw.split("|||", maxsplit=1)
    return f"{condition_label(modality)} / {topic}"


def report_condition_keys(shard_rows: dict[str, list[dict]], fallback_modalities: list[str]) -> list[str]:
    if shard_rows:
        return list(shard_rows)
    return fallback_modalities


def filter_report_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("model") not in EXCLUDED_REPORT_MODELS and not row.get("error")
    ]


def filter_report_models(models: list[str]) -> list[str]:
    return [model for model in models if model not in EXCLUDED_REPORT_MODELS]


def aggregate_by_modality(rows: list[dict]) -> dict[str, dict]:
    pooled_rows = [{**row, "model": "all-models"} for row in rows]
    pooled_summary = aggregate_results(pooled_rows)

    summary = {}
    for item in pooled_summary.values():
        modality = item.get("modality", "unknown")
        summary[modality] = item
    return summary


def uses_nested_conditions(condition_keys: list[str]) -> bool:
    return any(" :: " in key or "/" in key for key in condition_keys)


def judge_parse_failures(rows: list[dict]) -> dict[str, int]:
    return {
        "dialogue": sum(1 for row in rows if row.get("judge", {}).get("error")),
        "summary": sum(1 for row in rows if row.get("judge_summary", {}).get("error")),
    }


def judge_value(value: dict | None) -> str:
    if not value or value.get("error"):
        return "parse failed"
    reward = judge_total_reward(value)
    return str(reward) if reward is not None else "n/a"


def judge_numeric(row: dict) -> float:
    value = mean_judge_metric(row, "total_reward")
    return value if value is not None else 0.0


def sum_usage(rows: list[dict], usage_key: str, token_key: str) -> int:
    return int(sum(row.get(usage_key, {}).get(token_key, 0) for row in rows))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def format_mean_std(
    mean_val: float,
    std_val: float,
    *,
    precision: int = 1,
    percent: bool = False,
    use_commas: bool = False,
) -> str:
    if percent:
        return f"{mean_val:.{precision}%} +- {std_val:.{precision}%}"

    comma = "," if use_commas else ""
    return f"{mean_val:{comma}.{precision}f} +- {std_val:{comma}.{precision}f}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def format_topic_cell(rows: list[dict], topic: str, expected_count: int) -> str:
    processed_count = sum(1 for row in rows if row.get("kg_topic") == topic)
    return f"{processed_count}/{expected_count}"


def group_problem_ids(rows: list[dict], key: str) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_value = row.get(key)
        problem_id = row.get("problem_id")
        if group_value and problem_id:
            grouped[group_value].add(problem_id)
    return {group: sorted(problem_ids) for group, problem_ids in sorted(grouped.items())}


def unique_problem_ids(rows: list[dict]) -> list[str]:
    return sorted({row.get("problem_id") for row in rows if row.get("problem_id")})


def add_combined_key(rows: list[dict], output_key: str, *source_keys: str) -> list[dict]:
    combined_rows = []
    for row in rows:
        if not all(row.get(source_key) for source_key in source_keys):
            continue
        combined_rows.append({
            **row,
            output_key: "|||".join(str(row.get(source_key, "")) for source_key in source_keys),
        })
    return combined_rows


def combined_label_order(first_labels: list[str], second_labels: list[str]) -> list[str]:
    return [f"{first_label}|||{second_label}" for first_label in first_labels for second_label in second_labels]


def one_load_graph(row: dict) -> bool:
    return sum(1 for item in row.get("tool_calls_detail", []) if item.get("name") == "load_knowledge_graph") == 1


if __name__ == "__main__":
    main()
