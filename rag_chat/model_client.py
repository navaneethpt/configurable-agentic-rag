"""Shared model transport and structured-response repair."""
import json
from pydantic import BaseModel, ValidationError
from .agent_contracts import ChatError
from .providers import DEFAULT_MODEL, GROQ_MODEL
MODEL = GROQ_MODEL  # legacy public constant; saved values resolve as the default alias

def _complete(client, messages, *, max_tokens=1024, model=DEFAULT_MODEL, schema=None):
    return client.complete(messages=messages, model=model, max_tokens=max_tokens, schema=schema)


def _structured(client, schema: type[BaseModel], messages, role: str, *, model=DEFAULT_MODEL) -> BaseModel:
    """Request JSON and repair one malformed response without exposing it to the UI."""
    messages = [*messages, {"role": "system", "content": (
        "Match this JSON Schema exactly: " + json.dumps(schema.model_json_schema())
    )}]
    max_tokens = client.structured_max_tokens
    text = _complete(client, messages, max_tokens=max_tokens, model=model, schema=schema.model_json_schema())
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        repair = [
            {"role": "system", "content": (
                f"Return only valid JSON matching this schema for the {role} response. "
                "Do not add Markdown or explanation."
            )},
            {"role": "user", "content": json.dumps({
                "schema": schema.model_json_schema(), "invalid_response": text,
            })},
        ]
        repaired = _complete(client, repair, max_tokens=max_tokens, model=model, schema=schema.model_json_schema())
        try:
            return schema.model_validate_json(repaired)
        except ValidationError:
            raise ChatError(f"The {role} returned an invalid structured response. Please try again.") from None
