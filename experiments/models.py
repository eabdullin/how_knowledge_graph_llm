"""Model backends for OpenAI and OpenAI-compatible APIs."""

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .config import ModelConfig, ModelProvider


logger = logging.getLogger(__name__)

class ModelResponse:
    """Unified response from any model backend."""

    def __init__(
        self,
        content: str,
        tool_calls: Optional[list[dict]] = None,
        kg_items_used: Optional[list[str]] = None,
        thinking: Optional[str] = None,
        usage: Optional[dict] = None,
    ):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.kg_items_used = kg_items_used or []
        self.thinking = thinking
        self.usage = usage or {}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ModelBackend(ABC):
    """Abstract base for model backends."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> ModelResponse:
        ...


class OpenAIBackend(ModelBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )

    def chat(self, messages, tools=None, tool_choice=None, system=None, images=None):
        msgs = list(messages)

        kwargs = {
            "model": self.config.model_id,
            "input": _convert_openai_messages_to_responses_input(
                msgs,
                system=system,
                images=images,
            ),
            "max_output_tokens": self.config.max_tokens,
            "prompt_cache_key": _build_openai_prompt_cache_key(
                self.config,
                system=system,
                tools=tools,
                images=images,
            ),
            "prompt_cache_retention": "24h",
        }
        reasoning_with_tools_unsupported = (
            bool(tools)
            and self.config.reasoning_effort is not None
            and self.config.model_id in {"gpt-5.4-mini", "gpt-5.4"}
        )
        if self.config.reasoning_effort and not reasoning_with_tools_unsupported:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        # gpt-5-mini and similar reasoning models only support temperature=1
        # Skip the parameter entirely for these models
        _no_temp_models = {
            "gpt-5-mini",
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "o1",
            "o3",
            "o3-mini",
            "o1-mini",
        }
        if self.config.model_id not in _no_temp_models:
            kwargs["temperature"] = self.config.temperature
        if tools:
            response_tools = _convert_openai_tools_to_responses(tools)
            kwargs["tools"] = response_tools
            if tool_choice:
                kwargs["tool_choice"] = _convert_openai_tool_choice_to_responses(
                    tool_choice,
                    response_tools,
                )

        kwargs["service_tier"] = "flex"

        response = self.client.responses.create(**kwargs)

        tool_calls = []
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "name": item.name,
                    "arguments": json.loads(item.arguments)
                    if item.arguments
                    else {},
                })

        return ModelResponse(
            content=response.output_text,
            tool_calls=tool_calls,
            usage=_normalize_openai_usage(response.usage),
        )


class OpenAIChatCompletionBackend(ModelBackend):
    def __init__(
        self,
        config: ModelConfig,
        *,
        client,
        provider_name: str,
        supports_reasoning_effort: bool = False,
        extra_body_fields: Optional[set[str]] = None,
    ):
        super().__init__(config)
        self.client = client
        self.provider_name = provider_name
        self.supports_reasoning_effort = supports_reasoning_effort
        self.extra_body_fields = extra_body_fields or set()

    def _build_chat_completion_kwargs(
        self,
        messages,
        tools=None,
        tool_choice=None,
        system=None,
        images=None,
    ) -> tuple[list[dict], dict]:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        if images:
            msgs = _inject_images_openai(msgs, images)

        kwargs = {
            "model": self.config.model_id,
            "messages": msgs,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        extra_body = dict(kwargs.get("extra_body") or {})
        for field_name in ["presence_penalty", "repetition_penalty", "top_p", "top_k"]:
            value = getattr(self.config, field_name)
            if value is not None:
                if field_name in self.extra_body_fields:
                    extra_body[field_name] = value
                else:
                    kwargs[field_name] = value
        if self.config.enable_thinking is not None:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = self.config.enable_thinking
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        if (
            self.supports_reasoning_effort
            and self.config.enable_thinking is None
            and self.config.reasoning_effort
            and self.config.reasoning_effort != "none"
        ):
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        return msgs, kwargs

    def _create_chat_completion(self, kwargs):
        return self.client.chat.completions.create(**kwargs), None

    def _parse_chat_completion_response(self, response) -> ModelResponse:
        msg = response.choices[0].message

        tool_calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                arguments = tc.function.arguments if tc.function else None
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(arguments) if arguments else {},
                })

        content = _normalize_openai_compatible_content(
            getattr(msg, "content", None)
        )
        thinking = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        usage = _normalize_openai_compatible_usage(getattr(response, "usage", None))
        return ModelResponse(
            content=content.replace("<think>", "").strip(),
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
        )

    def chat(self, messages, tools=None, tool_choice=None, system=None, images=None):
        msgs, kwargs = self._build_chat_completion_kwargs(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            system=system,
            images=images,
        )

        response, response_headers = self._create_chat_completion(kwargs)
        model_response = self._parse_chat_completion_response(response)

        if not model_response.tool_calls and not model_response.content and not model_response.thinking:
            retry_response, retry_headers = self._create_chat_completion(
                {
                    **kwargs,
                    "messages": msgs + [{
                        "role": "user",
                        "content": (
                            "Your previous reply was blank. Reply with exactly one plain-text "
                            "assistant message only."
                        ),
                    }],
                }
            )
            retry_model_response = self._parse_chat_completion_response(retry_response)
            model_response.content = retry_model_response.content
            model_response.tool_calls = retry_model_response.tool_calls
            model_response.thinking = retry_model_response.thinking
            model_response.usage = _merge_usage_dicts(
                model_response.usage,
                retry_model_response.usage,
            )
            response_headers = retry_headers or response_headers

        self._handle_success_headers(response_headers)
        return model_response

    def _handle_success_headers(self, headers: Optional[dict[str, str]]) -> None:
        return None


class OpenAICompatibleBackend(OpenAIChatCompletionBackend):
    def __init__(self, config: ModelConfig):
        from openai import OpenAI

        if not config.base_url:
            raise ValueError(
                f"No OpenAI-compatible base URL is configured for '{config.model_id}'. "
                "Set OPENAI_COMPATIBLE_BASE_URL or the model-specific base URL "
                "documented in .env.example."
            )

        api_key = (
            os.environ.get(config.api_key_env or "")
            or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
            or "not-required"
        )

        client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )
        super().__init__(
            config,
            client=client,
            provider_name="OpenAI-compatible",
            supports_reasoning_effort=True,
            extra_body_fields={"repetition_penalty", "top_k"},
        )


def _normalize_openai_usage(usage) -> dict:
    """Return stable token fields from OpenAI usage objects."""
    if usage is None:
        return {}

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None or output_tokens is not None:
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        normalized = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": getattr(usage, "total_tokens", input_tokens + output_tokens),
        }

        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                normalized["reasoning_tokens"] = reasoning_tokens

        input_details = getattr(usage, "input_tokens_details", None)
        if input_details is not None:
            cached_tokens = getattr(input_details, "cached_tokens", None)
            if cached_tokens is not None:
                normalized["cached_input_tokens"] = cached_tokens

        return normalized

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    normalized = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", prompt_tokens + completion_tokens),
    }

    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            normalized["reasoning_tokens"] = reasoning_tokens
        accepted_tokens = getattr(completion_details, "accepted_prediction_tokens", None)
        if accepted_tokens is not None:
            normalized["accepted_prediction_tokens"] = accepted_tokens
        rejected_tokens = getattr(completion_details, "rejected_prediction_tokens", None)
        if rejected_tokens is not None:
            normalized["rejected_prediction_tokens"] = rejected_tokens

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        if cached_tokens is not None:
            normalized["cached_input_tokens"] = cached_tokens

    return normalized


def _normalize_openai_compatible_usage(usage) -> dict:
    normalized = _normalize_openai_usage(usage)
    if normalized:
        return normalized
    return _normalize_openai_compatible_fallback_usage(usage)


def _normalize_openai_compatible_content(content: str | None) -> str:
    if not content:
        return ""

    text = content.strip()
    text = _strip_compat_scaffolding(text)
    text = _strip_compat_meta_prefix(text)
    if "<|start|>" not in text and "<|channel|>" not in text:
        return text

    final_blocks = re.findall(
        r"<\|channel\|>final<\|message\|>(.*?)(?=(?:<\|end\|>|<\|start\|>|$))",
        text,
        re.DOTALL,
    )
    for candidate in reversed(final_blocks):
        candidate = candidate.strip()
        if candidate:
            return candidate

    question_matches = re.findall(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if question_matches:
        candidate = question_matches[-1].replace('\\"', '"').strip()
        if candidate:
            return candidate

    stripped = re.sub(r"<\|[^>]+\|>", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = _strip_compat_scaffolding(stripped)
    return _strip_compat_meta_prefix(stripped)


def _strip_compat_scaffolding(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(
        r'^(?:client will answer\.?|user will answer\.?|assistant will answer\.?)',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'(?:^|\s)assistant commentary to=assistant json \{[^{}]*"kg_refs"[^{}]*\}\s*',
        ' ',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'(?:^|\s)assistant commentary to=assistant json\s*',
        ' ',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'(?:^|\s)\{\s*"kg_refs"\s*:\s*"[^"]*"\s*\}\s*',
        ' ',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = _dedupe_repeated_question(cleaned)
    return cleaned


def _dedupe_repeated_question(text: str) -> str:
    match = re.fullmatch(r'(.+?\?)\s*\1(?:\s*\1)*', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _strip_compat_meta_prefix(text: str) -> str:
    lower = text.lower()
    if lower.startswith("we need to ask one question"):
        quoted_questions = re.findall(r'"([^"]+\?)"', text)
        if quoted_questions:
            return quoted_questions[-1].strip()

        question_start = re.search(
            r"\b(what|which|how|when|where|why|who|is|are|do|does|did|can|could|would|should)\b",
            text,
            re.IGNORECASE,
        )
        if question_start:
            return text[question_start.start():].strip()

    return text


def _merge_usage_dicts(*usage_dicts: dict) -> dict:
    merged: dict[str, int | float] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
    return merged


def _normalize_openai_compatible_fallback_usage(usage) -> dict:
    normalized = _normalize_openai_usage(usage)
    if normalized:
        return normalized

    if usage is None:
        return {}

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)

    normalized = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if reasoning_tokens is not None:
        normalized["reasoning_tokens"] = reasoning_tokens
    return {key: value for key, value in normalized.items() if value}



# ---------------------------------------------------------------------------
# Helper: image injection
# ---------------------------------------------------------------------------

def _inject_images_openai(messages: list[dict], image_paths: list[str]) -> list[dict]:
    """Insert one OpenAI vision-format image message after any leading system messages."""
    import base64

    msgs = list(messages)
    content_parts = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    insert_at = 0
    while insert_at < len(msgs) and msgs[insert_at].get("role") == "system":
        insert_at += 1

    msgs.insert(
        insert_at,
        {
            "role": "user",
            "content": content_parts,
        },
    )
    return msgs


def _build_openai_responses_image_message(image_paths: list[str]) -> dict:
    """Build a single Responses API developer message containing image inputs."""
    import base64

    content_parts: list[dict] = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content_parts.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
        })
    return {
        "type": "message",
        "role": "user",
        "content": content_parts,
    }


def _convert_openai_messages_to_responses_input(
    messages: list[dict],
    *,
    system: Optional[str] = None,
    images: Optional[list[str]] = None,
) -> list[dict]:
    """Convert the local OpenAI-style message log into Responses API input items."""
    items: list[dict] = []
    if system:
        items.append({
            "type": "message",
            "role": "developer",
            "content": _convert_openai_content_to_responses(system),
        })
    if images:
        items.append(_build_openai_responses_image_message(images))

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg.get("content", ""),
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            assistant_content = msg.get("content")
            if assistant_content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": assistant_content,
                })

            for tc in msg["tool_calls"]:
                function = tc.get("function", {})
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": function["name"],
                    "arguments": function.get("arguments") or "{}",
                })
            continue

        items.append({
            "type": "message",
            "role": "developer" if role == "system" else role,
            "content": _convert_openai_content_to_responses(msg.get("content", "")),
        })

    return items


def _convert_openai_content_to_responses(content: str | list[dict]) -> str | list[dict]:
    """Convert OpenAI chat content blocks into Responses API content blocks."""
    if isinstance(content, str):
        return content

    parts: list[dict] = []
    for part in content or []:
        part_type = part.get("type")
        if part_type == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif part_type == "image_url":
            image = part.get("image_url") or {}
            if image.get("url"):
                parts.append({
                    "type": "input_image",
                    "image_url": image["url"],
                })

    return parts or ""


def _convert_openai_tools_to_responses(openai_tools: list[dict]) -> list[dict]:
    """Convert chat-completions function tool definitions to Responses API format."""
    response_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            response_tool = {
                "type": "function",
                "name": fn["name"],
                "parameters": fn.get("parameters"),
                "strict": fn.get("strict"),
            }
            if fn.get("description"):
                response_tool["description"] = fn["description"]
            response_tools.append(response_tool)
            continue

        response_tools.append(tool)

    return response_tools


def _convert_openai_tool_choice_to_responses(tool_choice, tools: list[dict]):
    """Map chat-completions tool choice values to Responses API tool choice values."""
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "required"}:
            return {
                "type": "allowed_tools",
                "mode": tool_choice,
                "tools": [
                    {"type": tool["type"], "name": tool["name"]}
                    if tool.get("type") == "function"
                    else {"type": tool["type"]}
                    for tool in tools
                ],
            }
        return tool_choice

    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return {
                "type": "function",
                "name": tool_choice["function"]["name"],
            }
        return tool_choice

    return tool_choice


def _build_openai_prompt_cache_key(
    config: ModelConfig,
    *,
    system: Optional[str],
    tools: Optional[list[dict]],
    images: Optional[list[str]],
) -> str:
    """Build a stable cache key so similar OpenAI requests share prompt-cache buckets."""
    tool_signature = []
    for tool in tools or []:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            tool_signature.append({
                "name": fn.get("name"),
                "parameters": fn.get("parameters"),
            })
        else:
            tool_signature.append(tool)

    payload = {
        "model": config.model_id,
        "reasoning_effort": config.reasoning_effort,
        "system": system or "",
        "tool_signature": tool_signature,
        "has_images": bool(images),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]
    return f"kg-llm:{config.model_id}:{digest}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(config: ModelConfig) -> ModelBackend:
    """Create the appropriate model backend for a config."""
    if config.provider == ModelProvider.OPENAI:
        return OpenAIBackend(config)
    elif config.provider == ModelProvider.OPENAI_COMPATIBLE:
        return OpenAICompatibleBackend(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
