"""Delivery checks for model-controlled response text, independent of routing."""

from dataclasses import asdict, is_dataclass

from app.graph.prompts import answer_echoes_prompt

ANSWER_SAFETY_VERSION = "1.0"
_MODEL_TEXT_FIELDS = (
    "answer",
    "actions",
    "unsupported_claims",
    "reasoning_steps",
    "workflow_errors",
    "workflow_warnings",
)


def _contains_echo(value):
    if isinstance(value, str):
        return answer_echoes_prompt(value)
    if isinstance(value, dict):
        return any(_contains_echo(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_echo(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return _contains_echo(asdict(value))
    return False


def response_contains_prompt_echo(response):
    """Check outward model text, including metadata and action descriptions.

    Retrieved source text is evidence, not model output. It is not added to
    the confidential instruction corpus or mistaken for a generated leak.
    """
    return any(_contains_echo(response.get(key)) for key in _MODEL_TEXT_FIELDS)
