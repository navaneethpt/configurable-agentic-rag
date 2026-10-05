"""Shared model transport and structured-response repair."""
import json
from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, ValidationError
from .agent_contracts import ChatError
MODEL = "openai/gpt-oss-20b"

def _complete(client, messages, *, max_tokens=1024, model=MODEL):
    try:
        response = client.chat.completions.create(
            model=model, reasoning_effort="low", messages=messages,
            max_completion_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise ChatError("The model returned an empty response. Please try again.")
        return text.strip()
    except ChatError:
        raise
    except AuthenticationError:
        raise ChatError("Groq rejected the API key. Check GROQ_API_KEY and restart the app.") from None
    except RateLimitError:
        raise ChatError("Groq's rate limit was reached. Wait a moment and try again.") from None
    except (APITimeoutError, APIConnectionError):
        raise ChatError("Cannot reach Groq. Check your connection and try again.") from None
    except Exception:
        raise ChatError("Groq could not complete the request. Please try again.") from None


def _structured(client, schema: type[BaseModel], messages, role: str, *, model=MODEL) -> BaseModel:
    """Request JSON and repair one malformed response without exposing it to the UI."""
    messages = [*messages, {"role": "system", "content": (
        "Match this JSON Schema exactly: " + json.dumps(schema.model_json_schema())
    )}]
    text = _complete(client, messages, max_tokens=768, model=model)
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
        repaired = _complete(client, repair, max_tokens=768, model=model)
        try:
            return schema.model_validate_json(repaired)
        except ValidationError:
            raise ChatError(f"The {role} returned an invalid structured response. Please try again.") from None


