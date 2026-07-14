"""LLM-as-judge evaluation.

Uses the paper's summary-based and dialogue-based judging prompts.
Also includes G-Eval style conversational quality assessment.
"""

import json
import logging
import re

from ..config import ConversationLog, ModelConfig, resolve_model_config
from ..models import create_backend

logger = logging.getLogger(__name__)

# ---- Summary-based judge ----
JUDGE_SUMMARY_PROMPT = """\
You are an AI evaluator specializing in assessing the quality of summaries.
Carefully check how the summary captured a linear programming problem statement.
Important information for this task is names and values of: decision variables, constraints of all type, an objective function.
Your primary goal is to rate the summary based on Information Recall, Information Precision, Information Redundancy.
Be critical and strict.
We do not require to build the model.

Criteria for summary:
- The summary should be well structured.
- The summary should include objective function.
- The summary should include all decision variables from the problem statement. Mentioning term "decision variable" is not required.
- The summary should include all constraints mention in the problem statement. Mentioning term "constraint" is not required.
- If problem statement contains numbers all numbers should be explicitly mentioned in the summary.
- The summary should no include any data manipulations (Very important!). Meaning it doesn't contain any calculations or transformations of original data. We treat this as less recall score, less redundancy score
- The summary should not introduce any new information that is not present in the problem statement.

Criteria for scoring:
- Information Precision - Does captured information match with the problem statement? Captured information should be from the problem statement and accurately present the information. Check one by one.
- Information Redundancy - Any redundant information? Did modeller repeat the same information in the summary (may be with different wording)? Information that doesn't exist in the problem statement should not be present in the summary.
- Information Recall - Did assistant capture all important information from the problem statement? Is anything missing?. Check one by one.

PROVIDE THE ANSWER IN A JSON FORMAT WITH FOLLOWING FIELDS:
```json
{{
    "information_recall_score" : "score from 1 to 5", 1 - all important information is missing, 5 - all important information is captured
    "information_precision_score" : "score from 1 to 5", 1 - all captured information is inaccurate or not from the problem statement, 5 - all captured information is accurate and from the problem statement
    "information_redundancy_score" : "score from 1 to 5", 1 - all captured information is redundant, 5 - no redundant information is present
}}
```

The Problem Statement:
'''
{problem_description}
'''

The Provided Summary:
'''
{summary}
'''\
"""

# ---- Dialogue-based judge ----
JUDGE_DIALOG_PROMPT = """\
You are an AI evaluator specializing in assessing the quality of dialogs.
Carefully check how the dialog captured a linear programming problem statement.
Important information for this task is names and values of: decision variables, constraints of all type, an objective function.
Your primary goal is to rate the dialog based on Information Recall, Information Precision, Information Redundancy.
Be critical and strict.
We do not require to build the model.

Criteria for dialog:
- The dialog should be well structured.
- The dialog should include all information from the problem statement. Mentioning term "information" is not required.
- The dialog should not include any data manipulations (Very important!). Meaning it doesn't contain any calculations or transformations of original data. We treat this as less recall score, less redundancy score
- The dialog should not introduce any new information that is not present in the problem statement.

Criteria for scoring:
- Information Precision - Does captured information match with the problem statement? Captured information should be from the problem statement and accurately present the information. Check one by one.
- Information Redundancy - Any redundant information? Did modeller repeat the same information in the dialog (may be with different wording)? Information that doesn't exist in the problem statement should not be present in the dialog.
- Information Recall - Did assistant capture all important information from the problem statement? Is anything missing?. Check one by one.

PROVIDE THE ANSWER IN A JSON FORMAT WITH FOLLOWING FIELDS:
```json
{{
    "information_recall_score" : "score from 1 to 5", 1 - all important information is missing, 5 - all important information is captured
    "information_precision_score" : "score from 1 to 5", 1 - all captured information is inaccurate or not from the problem statement, 5 - all captured information is accurate and from the problem statement
    "information_redundancy_score" : "score from 1 to 5", 1 - all captured information is redundant, 5 - no redundant information is present
}}
```

The Problem Statement:
'''
{problem_description}
'''

The Provided Dialog:
'''
{dialogue}
'''\
"""

# G-Eval style prompt for conversational quality
GEVAL_PROMPT = """\
You are evaluating the quality of an information-elicitation dialogue \
between an AI modeller and a client. Rate the following dimensions on \
a scale of 1-5:

1. **Coherence** (1-5): Is the conversation logically structured? Do \
questions follow naturally from answers?
2. **Consistency** (1-5): Does the modeller avoid contradicting itself \
or asking the same thing twice?
3. **Fluency** (1-5): Is the language natural and easy to understand?
4. **Relevance** (1-5): Are the questions pertinent to building the \
optimization model?
5. **Engagingness** (1-3): Is the conversation engaging and does the \
modeller adapt to the client?

For each criterion, provide a score and brief justification.
Output as JSON:
```json
{{
  "coherence": {{"score": N, "reason": "..."}},
  "consistency": {{"score": N, "reason": "..."}},
  "fluency": {{"score": N, "reason": "..."}},
  "relevance": {{"score": N, "reason": "..."}},
  "engagingness": {{"score": N, "reason": "..."}}
}}
```

The Dialogue:
```
{dialogue}
```\
"""


def format_dialogue(conversation: ConversationLog) -> str:
    """Format conversation turns into a readable dialogue string.

    Uses the dialogue format described in the associated paper.
    """
    lines = []
    for turn in conversation.turns:
        if turn.role == "modeller" and turn.content:
            lines.append(f"### Assistant:\n{turn.content}")
        elif turn.role == "user" and turn.content:
            lines.append(f"### User:\n{turn.content}")
    return "\n\n".join(lines)


def _extract_summary(conversation: ConversationLog) -> str | None:
    """Extract the markdown summary from the last modeller turn, if present."""
    for turn in reversed(conversation.turns):
        if turn.role == "modeller" and turn.content and "```markdown" in turn.content:
            parts = turn.content.split("```markdown")
            if len(parts) > 1:
                summary = parts[1]
                if "```" in summary:
                    summary = summary.split("```")[0]
                return summary.strip()
    return None


def judge_dialogue(
    conversation: ConversationLog,
    problem_description: str,
    judge_model: str = "gpt-5-mini",
    eval_type: str = "dialog",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict:
    """Run LLM judge on a conversation. Returns scores dict.

    eval_type: "dialog" (default, scores full dialogue) or "summary" (scores
    only the final markdown summary block).
    """
    config = resolve_model_config(judge_model, warn_alias=False)
    judge_config = ModelConfig(
        provider=config.provider,
        model_id=config.model_id,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=config.reasoning_effort,
    )
    backend = create_backend(judge_config)

    if eval_type == "summary":
        summary = _extract_summary(conversation)
        if summary is None:
            return {"error": "no_summary_found"}
        prompt = JUDGE_SUMMARY_PROMPT.format(
            problem_description=problem_description,
            summary=summary,
        )
    else:
        dialogue = format_dialogue(conversation)
        prompt = JUDGE_DIALOG_PROMPT.format(
            problem_description=problem_description,
            dialogue=dialogue,
        )

    response = backend.chat(
        messages=[{"role": "user", "content": prompt}],
    )

    raw_content = response.content or ""
    scores = _parse_json_response(raw_content)
    if scores is None:
        logger.warning(f"Failed to parse judge response: {raw_content[:200]}")
        return {"error": "parse_failed", "raw": raw_content[:500], "usage": response.usage}

    for key in ["information_recall_score", "information_precision_score", "information_redundancy_score"]:
        val = scores.get(key, 0)
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                val = 0
        scores[key] = val
    scores["eval_type"] = eval_type
    scores["usage"] = response.usage

    return scores


def geval_dialogue(
    conversation: ConversationLog,
    judge_model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> dict:
    """Run G-Eval style quality assessment on a conversation."""
    config = resolve_model_config(judge_model, warn_alias=False)
    judge_config = ModelConfig(
        provider=config.provider,
        model_id=config.model_id,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=config.reasoning_effort,
    )
    backend = create_backend(judge_config)

    dialogue = format_dialogue(conversation)
    prompt = GEVAL_PROMPT.format(dialogue=dialogue)

    response = backend.chat(
        messages=[{"role": "user", "content": prompt}],
    )

    scores = _parse_json_response(response.content)
    if scores is None:
        logger.warning(f"Failed to parse G-Eval response: {response.content[:200]}")
        return {"error": "parse_failed", "raw": response.content, "usage": response.usage}

    scores["usage"] = response.usage
    return scores


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response text."""
    # Try to find JSON in code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to parse the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find any JSON-like structure
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None
