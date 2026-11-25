"""Delivery checks for model-controlled response text, independent of routing."""

import re
from copy import copy
from dataclasses import asdict, fields, is_dataclass

from pydantic import BaseModel

from app.graph.prompts import answer_echoes_prompt

ANSWER_SAFETY_VERSION = "1.1"
_CONTENT_FIELDS = ("answer", "actions")
_DIAGNOSTIC_FIELDS = (
    "unsupported_claims",
    "reasoning_steps",
    "workflow_errors",
    "workflow_warnings",
)
_MODEL_TEXT_FIELDS = _CONTENT_FIELDS + _DIAGNOSTIC_FIELDS
_REDACTED_DIAGNOSTIC = "[internal diagnostic withheld]"


def _contains_echo(value):
    if isinstance(value, str):
        return answer_echoes_prompt(value)
    if isinstance(value, dict):
        return any(
            _contains_echo(key) or _contains_echo(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_echo(item) for item in value)
    if isinstance(value, BaseModel):
        return _contains_echo(value.model_dump(warnings=False))
    if is_dataclass(value) and not isinstance(value, type):
        return _contains_echo(asdict(value))
    return False


def prompt_echo_fields(response, *, content_only=False):
    """Return fixed field names only; never retain matching text or n-grams."""
    candidates = _CONTENT_FIELDS if content_only else _MODEL_TEXT_FIELDS
    return [key for key in candidates if _contains_echo(response.get(key))]


def _redact_diagnostic(value):
    if isinstance(value, str):
        return _REDACTED_DIAGNOSTIC if answer_echoes_prompt(value) else value
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            safe_key = _redact_diagnostic(key)
            if safe_key in result:
                safe_key = f"{_REDACTED_DIAGNOSTIC}:{index}"
            result[safe_key] = _redact_diagnostic(item)
        return result
    if isinstance(value, list):
        return [_redact_diagnostic(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_diagnostic(item) for item in value)
    if isinstance(value, BaseModel):
        values = {
            key: _redact_diagnostic(getattr(value, key, None))
            for key in type(value).model_fields
        }
        values.update(
            {
                key: _redact_diagnostic(item)
                for key, item in (value.model_extra or {}).items()
            }
        )
        result = value.model_copy(update=values)
        # A computed field may derive its text outside the stored model fields.
        return (
            _redact_diagnostic(result.model_dump(warnings=False))
            if _contains_echo(result)
            else result
        )
    if is_dataclass(value) and not isinstance(value, type):
        result = copy(value)
        for field in fields(value):
            object.__setattr__(
                result, field.name, _redact_diagnostic(getattr(value, field.name))
            )
        return result
    return value


def _normalize_reasoning_step(value):
    if isinstance(value, dict):
        result = dict(value)
    elif isinstance(value, BaseModel):
        result = value.model_dump(warnings=False)
    elif is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    else:
        return {
            "node": "delivery",
            "action": "error",
            "detail": "[malformed diagnostic withheld]",
        }
    for key, fallback in (("node", "unknown"), ("action", "error")):
        label = result.get(key)
        if not isinstance(label, str) or not re.fullmatch(r"[a-z_]{1,80}", label):
            result[key] = fallback
    if isinstance(value, BaseModel):
        return value.model_copy(update=result)
    if is_dataclass(value) and not isinstance(value, type):
        record = copy(value)
        for key in ("node", "action"):
            if hasattr(record, key):
                object.__setattr__(record, key, result[key])
        return record
    return result


def diagnostic_messages(value):
    """Keep direct error/warning entries visible to eval and handoff gates."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item)[:200] for item in value]
    return ["[malformed diagnostic withheld]"]


def redact_diagnostic_echoes(response):
    """Sanitize an owned copy without changing answer validity or severity.

    Diagnostic entries and trusted node/action labels retain their shape.
    Errors stay errors, including when their detail must be withheld.
    """
    redacted_fields = [
        key for key in _DIAGNOSTIC_FIELDS if _contains_echo(response.get(key))
    ]
    result = {
        **response,
        **{
            key: _redact_diagnostic(response[key])
            for key in _DIAGNOSTIC_FIELDS
            if key in response
        },
    }
    if "reasoning_steps" in result:
        steps = result["reasoning_steps"]
        if not isinstance(steps, (list, tuple)):
            steps = [steps]
        result["reasoning_steps"] = [_normalize_reasoning_step(step) for step in steps]
    return result, redacted_fields


def response_contains_prompt_echo(response):
    """Check outward model text, including metadata and action descriptions.

    Retrieved source text is evidence, not model output. It is not added to
    the confidential instruction corpus or mistaken for a generated leak.
    """
    return bool(prompt_echo_fields(response))
