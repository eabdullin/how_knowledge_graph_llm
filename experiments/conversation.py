"""Conversation runner: orchestrates modeller ↔ simulated user dialogue."""

import json
import logging
import re
import uuid

from .config import (
    ConversationLog,
    ConversationTurn,
    Modality,
    ModelConfig,
)
from .kg_modalities import KGModality
from .models import ModelBackend, ModelResponse, create_backend
from .simulated_user import SimulatedUser

logger = logging.getLogger(__name__)


class ConversationRunner:
    """Run a single conversation between a modeller and simulated user."""

    def __init__(
        self,
        modality: KGModality,
        modeller_config: ModelConfig,
        user: SimulatedUser,
        max_turns: int = 20,
        problem_id: str = "",
    ):
        self.modality = modality
        self.modeller = create_backend(modeller_config)
        self.modeller_config = modeller_config
        self.user = user
        self.max_turns = max_turns
        self.problem_id = problem_id

        self._system_prompt = modality.get_system_prompt()
        self._images = modality.get_images()
        self._messages: list[dict] = []
        self._tool_calls_log: list[dict] = []
        self._modeller_usage_log: list[dict] = []

    def run(self) -> ConversationLog:
        """Execute the full conversation and return the log."""
        log = ConversationLog(
            experiment_id=str(uuid.uuid4())[:8],
            modality=Modality(self.modality.modality_name),
            model=self.modeller_config.name,
            problem_id=self.problem_id,
        )

        if self.max_turns <= 0:
            log.metadata["completed"] = False
            log.metadata["completion_reason"] = "max_turns_reached"
            log.metadata["total_tool_calls"] = 0
            log.metadata["tool_calls_detail"] = []
            log.metadata["modeller_usage_log"] = []
            log.metadata["user_usage_log"] = []
            log.metadata["kg_items_used"] = []
            log.metadata["usage_totals"] = {}
            return log

        # First turn: modeller starts (no user message yet)
        first_response = self._modeller_turn(is_first=True)
        log.turns.append(ConversationTurn(
            role="modeller",
            content=first_response.content,
            tool_calls=[tc for tc in first_response.tool_calls],
            kg_items_used=first_response.kg_items_used,
            thinking=first_response.thinking,
            usage=first_response.usage,
        ))
        logger.info(f"Modeller: {first_response.content[:100]}...")

        if self._has_summary(first_response.content):
            log.metadata["completed"] = True
            log.metadata["completion_reason"] = "summary_produced"
            log.metadata["total_tool_calls"] = len(self._tool_calls_log)
            log.metadata["tool_calls_detail"] = self._tool_calls_log
            log.metadata["modeller_usage_log"] = self._modeller_usage_log
            log.metadata["user_usage_log"] = getattr(self.user, "usage_log", [])
            log.metadata["kg_items_used"] = [
                {
                    "turn_index": idx + 1,
                    "items": turn.kg_items_used,
                }
                for idx, turn in enumerate(log.turns)
                if turn.role == "modeller" and turn.kg_items_used
            ]
            loaded_topic = getattr(self.modality, "loaded_topic", None)
            if loaded_topic:
                log.metadata["loaded_kg_topic"] = loaded_topic
            log.metadata["usage_totals"] = _aggregate_usage(
                self._modeller_usage_log,
                getattr(self.user, "usage_log", []),
            )
            return log

        for _ in range(max(self.max_turns - 1, 0)):

            # User responds
            last_modeller_text = log.turns[-1].content
            user_reply = self.user.respond(last_modeller_text)
            log.turns.append(ConversationTurn(
                role="user",
                content=user_reply,
                usage=getattr(self.user, "last_usage", {}),
            ))
            logger.info(f"User: {user_reply[:100]}...")

            # Modeller responds
            modeller_resp = self._modeller_turn(user_message=user_reply)
            log.turns.append(ConversationTurn(
                role="modeller",
                content=modeller_resp.content,
                tool_calls=[tc for tc in modeller_resp.tool_calls],
                kg_items_used=modeller_resp.kg_items_used,
                thinking=modeller_resp.thinking,
                usage=modeller_resp.usage,
            ))
            logger.info(f"Modeller: {modeller_resp.content[:100]}...")
            if self._has_summary(modeller_resp.content):
                log.metadata["completed"] = True
                log.metadata["completion_reason"] = "summary_produced"
                break
        else:
            log.metadata["completed"] = False
            log.metadata["completion_reason"] = "max_turns_reached"

        log.metadata["total_tool_calls"] = len(self._tool_calls_log)
        log.metadata["tool_calls_detail"] = self._tool_calls_log
        log.metadata["modeller_usage_log"] = self._modeller_usage_log
        log.metadata["user_usage_log"] = getattr(self.user, "usage_log", [])
        log.metadata["kg_items_used"] = [
            {
                "turn_index": idx + 1,
                "items": turn.kg_items_used,
            }
            for idx, turn in enumerate(log.turns)
            if turn.role == "modeller" and turn.kg_items_used
        ]
        loaded_topic = getattr(self.modality, "loaded_topic", None)
        if loaded_topic:
            log.metadata["loaded_kg_topic"] = loaded_topic
        log.metadata["usage_totals"] = _aggregate_usage(
            self._modeller_usage_log,
            getattr(self.user, "usage_log", []),
        )
        return log

    def _modeller_turn(
        self, user_message: str | None = None, is_first: bool = False
    ) -> ModelResponse:
        """Execute one modeller turn, handling tool calls in a loop."""
        if is_first:
            self._messages.append(
                {"role": "user", "content": "Hello, I need help with my business."}
            )
        elif user_message:
            self._messages.append({"role": "user", "content": user_message})

        tool_used = False
        accumulated_content = ""
        accumulated_tool_calls = []
        accumulated_kg_items: list[str] = []
        thinking = None
        turn_usage_log = []

        while True:
            tools = self.modality.get_tools()
            tool_choice = self.modality.get_tool_choice(
                is_first=is_first,
                user_message=user_message,
            )

            response = self.modeller.chat(
                messages=self._messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                system=self._system_prompt,
                images=self._images,
            )
            self._modeller_usage_log.append({
                "call_index": len(self._modeller_usage_log) + 1,
                "phase": "tool_call" if response.has_tool_calls else "final_response",
                "tool_call_count": len(response.tool_calls),
                "tool_call_names": [tc["name"] for tc in response.tool_calls],
                "usage": response.usage,
            })

            if response.thinking:
                thinking = response.thinking

            turn_usage_log.append({"usage": response.usage})

            if response.has_tool_calls:
                tool_used = True
                # Add assistant message with tool calls in OpenAI-compatible format
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                self._messages.append(assistant_msg)
                accumulated_tool_calls.extend(response.tool_calls)

                # Execute each tool call
                for tc in response.tool_calls:
                    result = self.modality.handle_tool_call(tc["name"], tc["arguments"])
                    self._tool_calls_log.append({
                        "turn_index": len([m for m in self._messages if m.get("role") == "assistant"]) + 1,
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "argument_chars": len(json.dumps(tc["arguments"], ensure_ascii=False)),
                        "result": result[:500],  # Truncate for logging
                        "result_chars": len(result),
                    })
                    logger.debug(f"  Tool: {tc['name']}({tc['arguments']}) -> {result[:200]}")

                    # Add tool result to messages
                    # Format varies by provider but we normalize in the backend
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            # No tool calls — we have the final text response
            accumulated_content, accumulated_kg_items = _extract_hidden_kg_refs(response.content)
            self._messages.append({
                "role": "assistant",
                "content": accumulated_content,
            })

            return ModelResponse(
                content=accumulated_content,
                tool_calls=accumulated_tool_calls,
                kg_items_used=accumulated_kg_items,
                thinking=thinking,
                usage=_aggregate_usage(turn_usage_log),
            )

    @staticmethod
    def _has_summary(text: str) -> bool:
        """Check if the modeller produced a final summary."""
        if not text:
            return False
        return "```markdown" in text.lower() or "```\n" in text and "summary" in text.lower()


def _aggregate_usage(*logs: list[dict]) -> dict:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }
    for log in logs:
        for entry in log:
            usage = entry.get("usage", {}) if isinstance(entry, dict) else {}
            for key in totals:
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    totals[key] += value

    return {key: value for key, value in totals.items() if value}


_KG_REFS_PATTERN = re.compile(r"<kg_refs>(.*?)</kg_refs>", re.DOTALL)


def _extract_hidden_kg_refs(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []

    matches = list(_KG_REFS_PATTERN.finditer(text))
    if not matches:
        return text, []

    refs = []
    for match in matches:
        raw_refs = match.group(1).strip()
        if not raw_refs:
            continue
        for chunk in raw_refs.split("|"):
            item = chunk.strip(" \n\t-•")
            if item:
                refs.append(item)

    cleaned = _KG_REFS_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _collapse_repeated_question(cleaned)
    return cleaned, refs


def _collapse_repeated_question(text: str) -> str:
    match = re.fullmatch(r'(.+?\?)\s*\1(?:\s*\1)*', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
