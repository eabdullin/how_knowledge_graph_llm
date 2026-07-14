#!/usr/bin/env python3
"""Re-run evaluation for existing experiment dialogues without regenerating them."""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import (
    ConversationLog,
    ConversationTurn,
    Modality,
    MODELS,
    load_benchmark_problems,
    resolve_model_config,
)
from .evaluation.coverage import compute_coverage
from .evaluation.llm_judge import geval_dialogue, judge_dialogue
from .evaluation.metrics import aggregate_results, save_results
from .knowledge_graph_registry import get_graph_dot_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved experiment dialogues")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing shards/")
    parser.add_argument("--judge-model", default="gpt-5.4-high", choices=sorted(MODELS))
    parser.add_argument("--judge-temperature", type=float, default=0.8)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--eval-version", default=None, help="Optional prompt/evaluation pass label recorded per row")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows per raw file")
    parser.add_argument("--parallel-shards", type=int, default=1, help="Number of shard files to evaluate concurrently")
    parser.add_argument("--force", action="store_true", help="Re-evaluate rows even if this config already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    eval_config = {
        "judge_model": args.judge_model,
        "judge_temperature": args.judge_temperature,
        "judge_temperature_api_parameter": _temperature_api_parameter(args.judge_model),
        "judge_max_tokens": args.judge_max_tokens,
        "eval_fields": ["coverage", "judge", "judge_summary", "geval", "evaluation_usage_totals"],
    }
    if args.eval_version:
        eval_config["eval_version"] = args.eval_version

    raw_paths = _find_shard_raw_results(run_dir)
    if not raw_paths:
        raise SystemExit(f"No shard raw_results.json files found under {run_dir / 'shards'}")

    logger.info("Found %d shard raw result files", len(raw_paths))
    totals = {"files": len(raw_paths), "rows": 0, "eligible": 0, "updated": 0, "skipped": 0, "errors": 0}

    parallel_shards = max(1, args.parallel_shards)
    if parallel_shards == 1:
        shard_stats = [
            reevaluate_raw_file(
                raw_path,
                eval_config=eval_config,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
            )
            for raw_path in raw_paths
        ]
    else:
        worker_count = min(parallel_shards, len(raw_paths))
        logger.info("Evaluating shard files with %d parallel workers", worker_count)
        shard_stats = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    reevaluate_raw_file,
                    raw_path,
                    eval_config=eval_config,
                    limit=args.limit,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                for raw_path in raw_paths
            ]
            for future in as_completed(futures):
                shard_stats.append(future.result())

    for stats in shard_stats:
        for key in ["rows", "eligible", "updated", "skipped", "errors"]:
            totals[key] += stats[key]

    logger.info("Totals: %s", json.dumps(totals, sort_keys=True))


def reevaluate_raw_file(
    raw_path: Path,
    *,
    eval_config: dict,
    limit: int | None,
    force: bool,
    dry_run: bool,
) -> dict:
    rows = json.loads(raw_path.read_text())
    stats = {"rows": len(rows), "eligible": 0, "updated": 0, "skipped": 0, "errors": 0}
    logger.info("Processing %s (%d rows)", raw_path, len(rows))

    processed_in_file = 0
    for index, row in enumerate(rows):
        if limit is not None and processed_in_file >= limit:
            break
        if not _has_dialogue(row):
            stats["skipped"] += 1
            continue
        stats["eligible"] += 1
        if not force and row.get("evaluation_config") == eval_config:
            stats["skipped"] += 1
            continue

        processed_in_file += 1
        if dry_run:
            stats["updated"] += 1
            continue

        try:
            rows[index] = reevaluate_row(row, eval_config)
            stats["updated"] += 1
            logger.info(
                "  -> updated %d/%d eligible in %s (%s/%s/%s run %s)",
                stats["updated"],
                stats["eligible"],
                raw_path.parent.name,
                row.get("modality"),
                row.get("model"),
                row.get("problem_id"),
                row.get("run_index"),
            )
        except Exception as exc:
            if _is_terminal_quota_error(exc):
                row.pop("evaluation_error", None)
                save_results(rows, raw_path)
                logger.error("Stopping re-evaluation because OpenAI reported insufficient quota: %s", exc)
                raise RuntimeError(f"OpenAI insufficient quota while re-evaluating {raw_path}") from None
            logger.exception(
                "Failed to re-evaluate %s row %d (%s/%s/%s run %s)",
                raw_path,
                index,
                row.get("modality"),
                row.get("model"),
                row.get("problem_id"),
                row.get("run_index"),
            )
            row["evaluation_error"] = str(exc)
            stats["errors"] += 1
        finally:
            save_results(rows, raw_path)

    if not dry_run:
        summary = aggregate_results(rows)
        save_results(list(summary.values()), raw_path.parent / "summary.json")
    logger.info("Finished %s: %s", raw_path, json.dumps(stats, sort_keys=True))
    return stats


def reevaluate_row(row: dict, eval_config: dict) -> dict:
    conversation = _conversation_from_row(row)
    problem_description = _load_problem_description(row)
    kg_text = get_graph_dot_text(row["kg_topic"])

    row["coverage"] = compute_coverage(conversation, kg_text, row["kg_topic"])
    row["judge"] = judge_dialogue(
        conversation,
        problem_description,
        judge_model=eval_config["judge_model"],
        eval_type="dialog",
        temperature=eval_config["judge_temperature"],
        max_tokens=eval_config["judge_max_tokens"],
    )
    row["judge_summary"] = judge_dialogue(
        conversation,
        problem_description,
        judge_model=eval_config["judge_model"],
        eval_type="summary",
        temperature=eval_config["judge_temperature"],
        max_tokens=eval_config["judge_max_tokens"],
    )
    row["geval"] = geval_dialogue(
        conversation,
        judge_model=eval_config["judge_model"],
        temperature=eval_config["judge_temperature"],
        max_tokens=eval_config["judge_max_tokens"],
    )
    row["evaluation_usage_totals"] = _sum_usage_dicts([
        row["judge"].get("usage", {}),
        row["judge_summary"].get("usage", {}),
        row["geval"].get("usage", {}),
    ])
    row["judge_model"] = eval_config["judge_model"]
    row["evaluation_config"] = eval_config
    row.pop("evaluation_error", None)
    return row


def _find_shard_raw_results(run_dir: Path) -> list[Path]:
    direct_path = run_dir / "raw_results.json"
    if direct_path.is_file():
        return [direct_path]
    shard_dir = run_dir / "shards"
    return sorted(path for path in shard_dir.glob("*/*/raw_results.json") if path.is_file())


def _has_dialogue(row: dict) -> bool:
    return isinstance(row.get("dialogue"), list) and bool(row["dialogue"])


def _conversation_from_row(row: dict) -> ConversationLog:
    return ConversationLog(
        experiment_id=row.get("experiment_id", ""),
        modality=Modality(row["modality"]),
        model=row["model"],
        problem_id=row["problem_id"],
        turns=[_turn_from_dict(turn) for turn in row["dialogue"]],
        metadata={
            "completed": row.get("completed", False),
            "completion_reason": row.get("completion_reason", ""),
            "total_tool_calls": row.get("total_tool_calls", 0),
            "loaded_kg_topic": row.get("loaded_kg_topic"),
            "usage_totals": row.get("usage_totals", {}),
            "modeller_usage_log": row.get("modeller_usage_log", []),
            "user_usage_log": row.get("user_usage_log", []),
            "tool_calls_detail": row.get("tool_calls_detail", []),
        },
    )


def _turn_from_dict(turn: dict) -> ConversationTurn:
    return ConversationTurn(
        role=turn.get("role", ""),
        content=turn.get("content") or "",
        tool_calls=turn.get("tool_calls", []),
        tool_results=turn.get("tool_results", []),
        kg_items_used=turn.get("kg_items_used", []),
        thinking=turn.get("thinking"),
        usage=turn.get("usage", {}),
    )


def _load_problem_description(row: dict) -> str:
    problem_id = row["problem_id"]
    problems = load_benchmark_problems()
    if problem_id not in problems:
        raise KeyError(f"Problem description not found for {problem_id}")
    return problems[problem_id]["description"]


def _sum_usage_dicts(usages: list[dict]) -> dict:
    totals = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def _temperature_api_parameter(judge_model: str) -> str:
    config = resolve_model_config(judge_model, warn_alias=False)
    if config.provider.value == "openai" and config.model_id in {"gpt-5.4", "gpt-5.4-mini"}:
        return "omitted_unsupported_by_openai_responses"
    return "sent"


def _is_terminal_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
