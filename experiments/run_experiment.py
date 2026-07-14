#!/usr/bin/env python3
"""Run and validate the knowledge-graph modality experiments."""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from .config import (
    ExperimentConfig,
    KG_IMAGE_PATHS,
    Modality,
    MODELS,
    REPO_ROOT,
    load_benchmark_problems,
    resolve_model_config,
)
from .conversation import ConversationRunner
from .evaluation.coverage import compute_coverage
from .evaluation.llm_judge import judge_dialogue, geval_dialogue
from .knowledge_graph_registry import get_graph_dot_text, get_graph_mermaid_text
from .evaluation.metrics import aggregate_results, judge_total_reward, results_to_table, save_results
from .kg_modalities import create_modality
from .simulated_user import SimulatedUser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RUN_INDEX_PATTERN = re.compile(r"_run(?P<run_index>\d+)_")


def run_single_conversation(
    modality_name: str,
    model_name: str,
    problem_id: str,
    problem: dict,
    config: ExperimentConfig,
) -> dict:
    """Run a single conversation and return raw result dict."""
    model_config = resolve_model_config(model_name, warn_alias=False)

    if modality_name == "text":
        kg_text = get_graph_mermaid_text(problem["kg_topic"])
    else:
        kg_text = get_graph_dot_text(problem["kg_topic"])

    # Create modality
    modality_kwargs = {}
    if modality_name == "image":
        modality_kwargs["output_dir"] = config.output_dir / "images"
        pre_rendered = KG_IMAGE_PATHS.get(problem["kg_topic"])
        if pre_rendered:
            modality_kwargs["image_path"] = pre_rendered
    modality = create_modality(
        modality_name, kg_text, problem["kg_topic"], **modality_kwargs
    )

    # Create simulated user
    user_config = resolve_model_config(config.user_model, warn_alias=False)
    user = SimulatedUser(problem["description"], model_config=user_config)

    # Run conversation
    runner = ConversationRunner(
        modality=modality,
        modeller_config=model_config,
        user=user,
        max_turns=config.max_turns,
        problem_id=problem_id,
    )

    start_time = time.time()
    conversation = runner.run()
    elapsed = time.time() - start_time

    return {
        "experiment_id": conversation.experiment_id,
        "modality": modality_name,
        "model": model_name,
        "user_model": config.user_model,
        "judge_model": config.judge_model,
        "problem_id": problem_id,
        "problem_domain": problem.get("domain", ""),
        "problem_source": problem.get("source", "unknown"),
        "kg_topic": problem["kg_topic"],
        "selection_reason": problem.get("selection_reason", ""),
        "num_turns": conversation.num_turns,
        "total_tool_calls": conversation.metadata.get("total_tool_calls", 0),
        "completed": conversation.metadata.get("completed", False),
        "completion_reason": conversation.metadata.get("completion_reason", ""),
        "loaded_kg_topic": conversation.metadata.get("loaded_kg_topic"),
        "elapsed_seconds": round(elapsed, 1),
        "usage_totals": conversation.metadata.get("usage_totals", {}),
        "modeller_usage_log": conversation.metadata.get("modeller_usage_log", []),
        "user_usage_log": conversation.metadata.get("user_usage_log", []),
        "tool_calls_detail": conversation.metadata.get("tool_calls_detail", []),
        "conversation": conversation,
    }


def evaluate_result(
    result: dict,
    problem: dict,
    config: ExperimentConfig,
) -> dict:
    """Add evaluation metrics to a result dict."""
    conversation = result["conversation"]
    kg_text = get_graph_dot_text(problem["kg_topic"])

    # Coverage
    coverage = compute_coverage(conversation, kg_text, problem["kg_topic"])
    result["coverage"] = coverage

    # Judge (both summary and dialogue variants)
    judge_dialog = judge_dialogue(
        conversation, problem["description"],
        judge_model=config.judge_model, eval_type="dialog",
    )
    result["judge"] = judge_dialog

    judge_summary = judge_dialogue(
        conversation, problem["description"],
        judge_model=config.judge_model, eval_type="summary",
    )
    result["judge_summary"] = judge_summary

    # G-Eval
    geval = geval_dialogue(conversation, judge_model=config.judge_model)
    result["geval"] = geval
    result["evaluation_usage_totals"] = _sum_usage_dicts([
        judge_dialog.get("usage", {}),
        judge_summary.get("usage", {}),
        geval.get("usage", {}),
    ])

    return result


def run_experiments(
    config: ExperimentConfig,
    problems: dict[str, dict],
    *,
    resume: bool = False,
):
    """Run the full experiment matrix."""
    valid_pairs, skipped_pairs = _resolve_model_modality_pairs(config)
    _log_skipped_model_modality_pairs(skipped_pairs)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_results(_build_manifest(config, problems), config.output_dir / "experiment_manifest.json")
    results_by_key: dict[tuple[str, str, str, int], dict] = {}
    discarded_resume_rows = 0
    if resume:
        results_by_key, discarded_resume_rows = _load_resume_results(config.output_dir)
        logger.info(
            "Resume enabled: loaded %d completed runs from %s and dropped %d stale/unkeyed rows",
            len(results_by_key),
            config.output_dir / "raw_results.json",
            discarded_resume_rows,
        )

    total_conditions = len(valid_pairs) * len(problems)
    total_runs = total_conditions * config.num_runs
    logger.info(
        f"Starting experiments: {len(valid_pairs)} valid modality/model pairs x "
        f"{len(problems)} problems x "
        f"{config.num_runs} runs = {total_runs} total conversations"
    )

    run_count = 0
    for modality, model_name in valid_pairs:
            for problem_id, problem in problems.items():
                for run_idx in range(config.num_runs):
                    run_count += 1
                    logger.info(
                        f"[{run_count}/{total_runs}] "
                        f"{modality.value}/{model_name}/{problem_id} "
                        f"(run {run_idx + 1}/{config.num_runs})"
                    )

                    result_key = (modality.value, model_name, problem_id, run_idx + 1)
                    if result_key in results_by_key:
                        logger.info("  -> SKIP existing completed run")
                        continue

                    try:
                        result = run_single_conversation(
                            modality.value, model_name, problem_id,
                            problem, config,
                        )
                        result["run_index"] = run_idx + 1
                        result = evaluate_result(
                            result, problem, config,
                        )
                        # Remove non-serializable conversation object for saving
                        conv = result.pop("conversation")
                        result["dialogue"] = _serialise_dialogue(conv)
                        dialogue_paths = _save_dialogue_files(
                            conv,
                            result,
                            config.output_dir,
                            run_idx,
                        )
                        result["dialogue_files"] = dialogue_paths

                        judge_total = judge_total_reward(result.get("judge"))
                        logger.info(
                            f"  -> {result['num_turns']} turns, "
                            f"coverage={result['coverage'].get('node_coverage', '?')}, "
                            f"judge_total={judge_total if judge_total is not None else '?'}, "
                            f"tokens={result['usage_totals'].get('total_tokens', '?')}"
                        )
                        results_by_key[result_key] = result

                    except Exception as e:
                        if _is_terminal_quota_error(e):
                            logger.error("  -> STOPPING: OpenAI reported insufficient quota: %s", e)
                            raise RuntimeError("OpenAI insufficient quota during generation") from None
                        logger.error(f"  -> FAILED: {e}", exc_info=True)
                        results_by_key[result_key] = {
                            "modality": modality.value,
                            "model": model_name,
                            "problem_id": problem_id,
                            "run_index": run_idx + 1,
                            "error": str(e),
                        }

                    # Save incrementally
                    save_results(
                        _sort_results_for_save(results_by_key.values()),
                        config.output_dir / "raw_results.json",
                    )

    # Final summary
    all_results = _sort_results_for_save(results_by_key.values())
    summary = aggregate_results(all_results)
    save_results(list(summary.values()), config.output_dir / "summary.json")

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(results_to_table(summary))
    print("=" * 80)

    return all_results


def _resolve_model_modality_pairs(
    config: ExperimentConfig,
) -> tuple[list[tuple[Modality, str]], list[tuple[Modality, str]]]:
    valid_pairs: list[tuple[Modality, str]] = []
    skipped_pairs: list[tuple[Modality, str]] = []

    model_configs = {
        name: resolve_model_config(name) for name in config.models
    }
    for modality in config.modalities:
        for model_name in config.models:
            model_config = model_configs[model_name]
            if modality == Modality.IMAGE and not model_config.supports_images:
                skipped_pairs.append((modality, model_name))
                continue
            valid_pairs.append((modality, model_name))

    return valid_pairs, skipped_pairs


def _log_skipped_model_modality_pairs(skipped_pairs: list[tuple[Modality, str]]) -> None:
    for modality, model_name in skipped_pairs:
        logger.warning(
            "Skipping unsupported combination: %s/%s (model does not support image inputs)",
            modality.value,
            model_name,
        )


def _validate_requested_pairs(config: ExperimentConfig) -> None:
    resolve_model_config(config.user_model)
    resolve_model_config(config.judge_model)
    valid_pairs, skipped_pairs = _resolve_model_modality_pairs(config)
    if not valid_pairs:
        skipped_labels = ", ".join(
            f"{modality.value}/{model_name}" for modality, model_name in skipped_pairs
        )
        raise SystemExit(
            "No valid model/modality combinations remain after capability filtering: "
            f"{skipped_labels}"
        )


def _load_resume_results(output_dir: Path) -> tuple[dict[tuple[str, str, str, int], dict], int]:
    raw_results_path = output_dir / "raw_results.json"
    if not raw_results_path.exists():
        return {}, 0

    existing_rows = json.loads(raw_results_path.read_text())
    results_by_key = {}
    discarded_rows = 0
    for row in existing_rows:
        run_index = _extract_run_index(row)
        if run_index is None or "error" in row:
            discarded_rows += 1
            continue

        key = (row.get("modality", ""), row.get("model", ""), row.get("problem_id", ""), run_index)
        if not all(key[:3]):
            discarded_rows += 1
            continue
        results_by_key[key] = row

    return results_by_key, discarded_rows


def _extract_run_index(row: dict) -> int | None:
    run_index = row.get("run_index")
    if isinstance(run_index, int) and run_index > 0:
        return run_index

    dialogue_files = row.get("dialogue_files", {})
    for path in [dialogue_files.get("json"), dialogue_files.get("markdown")]:
        if not isinstance(path, str):
            continue
        match = RUN_INDEX_PATTERN.search(path)
        if match:
            return int(match.group("run_index"))

    return None


def _is_terminal_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def _sort_results_for_save(results) -> list[dict]:
    return sorted(
        results,
        key=lambda row: (
            row.get("modality", ""),
            row.get("model", ""),
            row.get("problem_id", ""),
            row.get("run_index", sys.maxsize),
            row.get("experiment_id", ""),
        ),
    )


def _build_manifest(config: ExperimentConfig, problems: dict[str, dict]) -> dict:
    by_source = {}
    by_topic = {}
    for problem_id, problem in problems.items():
        by_source.setdefault(problem.get("source", "unknown"), []).append(problem_id)
        by_topic.setdefault(problem.get("kg_topic", "unknown"), []).append(problem_id)

    return {
        "modalities": [modality.value for modality in config.modalities],
        "models": list(config.models),
        "max_turns": config.max_turns,
        "num_runs": config.num_runs,
        "user_model": config.user_model,
        "judge_model": config.judge_model,
        "problem_count": len(problems),
        "problem_ids": sorted(problems),
        "problems_by_source": {key: sorted(value) for key, value in sorted(by_source.items())},
        "problems_by_topic": {key: sorted(value) for key, value in sorted(by_topic.items())},
    }


def _validate_graph_assets(problems: dict[str, dict]) -> dict[str, dict[str, str]]:
    """Validate every graph representation needed by selected problems."""
    assets: dict[str, dict[str, str]] = {}
    for topic in sorted({problem["kg_topic"] for problem in problems.values()}):
        mermaid_text = get_graph_mermaid_text(topic)
        dot_text = get_graph_dot_text(topic)
        image_path = KG_IMAGE_PATHS.get(topic)
        if not mermaid_text.strip() or not dot_text.strip():
            raise ValueError(f"Graph text is empty for topic '{topic}'")
        if not image_path or not Path(image_path).is_file():
            raise ValueError(f"Graph image is missing for topic '{topic}'")
        assets[topic] = {
            "mermaid": "ok",
            "dot": "ok",
            "image": str(Path(image_path).relative_to(REPO_ROOT)),
        }
    return assets


def _dry_run_report(config: ExperimentConfig, problems: dict[str, dict]) -> dict:
    """Return validation details without creating files or calling providers."""
    _validate_requested_pairs(config)
    valid_pairs, skipped_pairs = _resolve_model_modality_pairs(config)
    return {
        "status": "ok",
        "manifest": _build_manifest(config, problems),
        "valid_model_modality_pairs": len(valid_pairs),
        "skipped_model_modality_pairs": [
            f"{modality.value}/{model}"
            for modality, model in skipped_pairs
        ],
        "planned_conversations": (
            len(valid_pairs) * len(problems) * config.num_runs
        ),
        "graph_assets": _validate_graph_assets(problems),
    }


def _serialise_dialogue(conversation) -> list[dict]:
    return [
        {
            "role": turn.role,
            "content": turn.content,
            "thinking": turn.thinking,
            "usage": turn.usage,
            "tool_calls": turn.tool_calls,
            "kg_items_used": turn.kg_items_used,
        }
        for turn in conversation.turns
    ]


def _sum_usage_dicts(usages: list[dict]) -> dict:
    totals = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def _save_dialogue_files(
    conversation,
    result: dict,
    output_dir: Path,
    run_idx: int,
) -> dict:
    dialogue_dir = output_dir / "dialogues"
    dialogue_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{result['modality']}_{result['model']}_{result['problem_id']}_"
        f"run{run_idx + 1}_{result['experiment_id']}"
    ).replace("/", "-")

    json_path = dialogue_dir / f"{stem}.json"
    md_path = dialogue_dir / f"{stem}.md"
    reasoning_md_path = dialogue_dir / f"{stem}.reasoning.md"

    dialogue_payload = {
        "experiment_id": result["experiment_id"],
        "modality": result["modality"],
        "model": result["model"],
        "problem_id": result["problem_id"],
        "num_turns": result["num_turns"],
        "completed": result["completed"],
        "completion_reason": result["completion_reason"],
        "usage_totals": result.get("usage_totals", {}),
        "total_tool_calls": result.get("total_tool_calls", 0),
        "tool_calls_detail": result.get("tool_calls_detail", []),
        "turns": _serialise_dialogue(conversation),
    }
    save_results(dialogue_payload, json_path)

    with open(md_path, "w") as f:
        f.write(f"# Dialogue: {result['modality']} / {result['problem_id']}\n\n")
        f.write(f"Experiment ID: `{result['experiment_id']}`\n\n")
        f.write(f"Completed: `{result['completed']}` ({result['completion_reason']})\n\n")
        f.write(f"Total tool calls: `{result.get('total_tool_calls', 0)}`\n\n")
        f.write(f"Usage totals: `{json.dumps(result.get('usage_totals', {}))}`\n\n")
        for idx, turn in enumerate(conversation.turns, start=1):
            f.write(f"## {idx}. {turn.role.title()}\n\n")
            if turn.usage:
                f.write(f"Usage: `{json.dumps(turn.usage)}`\n\n")
            if turn.tool_calls:
                tool_names = ", ".join(tc.get("name", "unknown") for tc in turn.tool_calls)
                f.write(f"Tool calls: `{tool_names}`\n\n")
            f.write((turn.content or "[empty]").strip() + "\n\n")

    with open(reasoning_md_path, "w") as f:
        f.write(f"# Reasoning: {result['modality']} / {result['problem_id']}\n\n")
        f.write(f"Experiment ID: `{result['experiment_id']}`\n\n")
        reasoning_turns = 0
        for idx, turn in enumerate(conversation.turns, start=1):
            if turn.role != "modeller" or not turn.thinking:
                continue
            reasoning_turns += 1
            f.write(f"## {idx}. {turn.role.title()}\n\n")
            if turn.usage:
                f.write(f"Usage: `{json.dumps(turn.usage)}`\n\n")
            f.write(turn.thinking.strip() + "\n\n")

        if reasoning_turns == 0:
            f.write("[no reasoning captured]\n")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "reasoning_markdown": str(reasoning_md_path),
    }


def main():
    parser = argparse.ArgumentParser(description="KG Modality Experiment Runner")
    parser.add_argument(
        "--modality", choices=["image", "text", "tools", "no_graph"],
        nargs="+", default=None, help="Modalities to test (default: all)",
    )
    parser.add_argument(
        "--model", nargs="+", choices=sorted(MODELS), default=None,
        help="Models to test (default: gpt-5.4-mini-medium)",
    )
    parser.add_argument(
        "--problem", nargs="+", default=None,
        help="Specific benchmark problem IDs",
    )
    parser.add_argument(
        "--dataset", choices=["benchmark"], default="benchmark",
        help="Bundled dataset to use (default: benchmark)",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit the number of benchmark problems",
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per condition")
    parser.add_argument("--max-turns", type=int, default=40, help="Max turns per conversation")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--user-model", choices=sorted(MODELS), default="gpt-5.4-none",
        help="Model for the simulated user",
    )
    parser.add_argument(
        "--judge-model", choices=sorted(MODELS), default="gpt-5.4-medium",
        help="Model for the LLM judge",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing raw_results.json in the output directory",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip conversations, only evaluate existing results",
    )
    parser.add_argument("--input", type=str, help="Input JSON for evaluate-only mode")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration, benchmark, and assets without API calls",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        modalities=[Modality(m) for m in args.modality] if args.modality else [
            Modality.IMAGE, Modality.TEXT, Modality.TOOLS, Modality.NO_GRAPH
        ],
        models=args.model or ["gpt-5.4-mini-medium"],
        max_turns=args.max_turns,
        num_runs=args.runs,
        output_dir=Path(args.output),
        user_model=args.user_model,
        judge_model=args.judge_model,
    )
    problems = load_benchmark_problems()
    if args.problem:
        requested = set(args.problem)
        unknown = sorted(requested - set(problems))
        if unknown:
            raise SystemExit(f"Unknown benchmark problem IDs: {', '.join(unknown)}")
        problems = {
            problem_id: problem
            for problem_id, problem in problems.items()
            if problem_id in requested
        }
    if args.max_problems is not None:
        if args.max_problems <= 0:
            raise SystemExit("--max-problems must be greater than zero")
        problems = dict(list(problems.items())[:args.max_problems])

    if not problems:
        print("No problems loaded. Check dataset paths.")
        return

    logger.info("Loaded %d problems from the bundled benchmark", len(problems))

    if args.dry_run:
        print(json.dumps(_dry_run_report(config, problems), indent=2))
        return

    _validate_requested_pairs(config)
    if args.evaluate_only:
        if not args.input:
            raise SystemExit("--evaluate-only requires --input")
        with open(args.input) as f:
            results = json.load(f)
        summary = aggregate_results(results)
        print(results_to_table(summary))
    else:
        run_experiments(config, problems, resume=args.resume)


if __name__ == "__main__":
    main()
