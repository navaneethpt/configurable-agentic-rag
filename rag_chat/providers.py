"""Provider selection and SDK adapters; credentials never enter workflow data."""
from dataclasses import dataclass, field
import os
from typing import Mapping

from groq import Groq, APIConnectionError, APITimeoutError, AuthenticationError, APIStatusError, RateLimitError
from google import genai
from google.genai import errors, types
import httpx

from .agent_contracts import ChatError

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_MODEL = "default"
REQUEST_TIMEOUT = 45.0


@dataclass(frozen=True)
class ProviderSettings:
    provider: str | None
    default_model: str | None
    api_key: str = field(repr=False)
    configuration_error: str | None = None


def provider_settings(env: Mapping[str, str] | None = None) -> ProviderSettings:
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"groq", "gemini"}:
        return ProviderSettings(None, None, "", "Set LLM_PROVIDER to groq or gemini and restart the backend.")
    model = env.get("LLM_MODEL", "").strip() or (GROQ_MODEL if provider == "groq" else GEMINI_MODEL)
    key_name = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    key = env.get(key_name, "").strip()
    error = None
    if len(model) > 100:
        error = "LLM_MODEL must contain at most 100 characters. Update it and restart the backend."
    elif (model == DEFAULT_MODEL
          or (provider == "gemini" and not model.removeprefix("models/").startswith("gemini-"))
          or (provider == "groq" and model.removeprefix("models/").startswith("gemini-"))):
        error = "Set LLM_MODEL to a model ID from the selected provider, or leave it unset for its default. Restart the backend."
    elif not key:
        error = f"Set {key_name} in the backend environment or .env and restart the backend."
    return ProviderSettings(provider, model, key, error)


class ModelClient:
    """Common extension client: complete returns text and raises safe ChatError."""
    provider: str
    structured_max_tokens = 768

    def __init__(self, default_model, sdk):
        self.default_model = default_model
        self.sdk = sdk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sdk.close()

    def resolve_model(self, model):
        selected = self.default_model if model in {DEFAULT_MODEL, GROQ_MODEL} else model
        gemini_name = selected.removeprefix("models/").startswith("gemini-")
        if (self.provider == "gemini" and not gemini_name) or (self.provider == "groq" and gemini_name):
            raise ChatError(f"The agent model is incompatible with {self.provider.title()}. "
                            "Choose a model from the selected provider or set the agent model to default.")
        return selected

    def complete(self, messages, model=DEFAULT_MODEL, max_tokens=1024, schema=None) -> str:
        raise NotImplementedError

    def text(self, value):
        if not value or not value.strip():
            raise ChatError(f"{self.provider.title()} returned an empty response. Please try again.")
        return value.strip()


class GroqClient(ModelClient):
    provider = "groq"

    def __init__(self, api_key=None, default_model=GROQ_MODEL, *, sdk=None):
        super().__init__(default_model, sdk if sdk is not None else Groq(
            api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=1))

    def complete(self, messages, model=DEFAULT_MODEL, max_tokens=1024, schema=None):
        selected = self.resolve_model(model)
        try:
            response = self.sdk.chat.completions.create(model=selected, reasoning_effort="low",
                messages=messages, max_completion_tokens=max_tokens)
            choice = response.choices[0]
            reason = getattr(choice, "finish_reason", None)
            if reason == "length":
                raise ChatError("Groq truncated the response. Try a shorter question or another agent model.")
            if reason == "content_filter":
                raise ChatError("Groq blocked the response. Rephrase the question and try again.")
            return self.text(choice.message.content)
        except ChatError:
            raise
        except AuthenticationError:
            raise ChatError("Groq rejected the API key. Check GROQ_API_KEY and restart the backend.") from None
        except RateLimitError:
            raise ChatError("Groq's rate limit was reached. Wait a moment and try again.") from None
        except (APITimeoutError, APIConnectionError):
            raise ChatError("Cannot reach Groq. Check your connection and try again.") from None
        except APIStatusError as exc:
            if exc.status_code == 403:
                raise ChatError("Groq rejected access. Check GROQ_API_KEY and model access.") from None
            if exc.status_code in {400, 404}:
                raise ChatError("Groq rejected the model or request. Check LLM_MODEL and the agent model setting.") from None
            raise ChatError("Groq could not complete the request. Please try again.") from None
        except Exception:
            raise ChatError("Groq could not complete the request. Please try again.") from None


class GeminiClient(ModelClient):
    provider = "gemini"
    # Gemini counts thinking tokens in its output limit; do not reuse Groq's small JSON budget.
    structured_max_tokens = 4096

    def __init__(self, api_key=None, default_model=GEMINI_MODEL, *, sdk=None):
        super().__init__(default_model, sdk if sdk is not None else genai.Client(
            api_key=api_key, vertexai=False, http_options=types.HttpOptions(
                timeout=int(REQUEST_TIMEOUT * 1000),
                retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=1, jitter=0))))

    def complete(self, messages, model=DEFAULT_MODEL, max_tokens=1024, schema=None):
        selected = self.resolve_model(model)
        system = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
        contents = [types.Content(role="model" if item["role"] == "assistant" else "user",
                                 parts=[types.Part.from_text(text=item["content"])])
                    for item in messages if item["role"] != "system"]
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
        if selected.removeprefix("models/") == GEMINI_MODEL:
            config.thinking_config = types.ThinkingConfig(thinking_level="low")
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = schema
        try:
            response = self.sdk.models.generate_content(model=selected, contents=contents, config=config)
            feedback = getattr(response, "prompt_feedback", None)
            block = getattr(feedback, "block_reason", None)
            if block and block != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED:
                raise ChatError("Gemini blocked the request. Rephrase the question and try again.")
            candidates = response.candidates or []
            if candidates:
                reason = candidates[0].finish_reason
                if reason == types.FinishReason.MAX_TOKENS:
                    raise ChatError("Gemini truncated the response. Try a shorter question or another agent model.")
                if reason not in {None, types.FinishReason.STOP, types.FinishReason.FINISH_REASON_UNSPECIFIED}:
                    raise ChatError("Gemini blocked or could not finish the response. Rephrase the question and try again.")
            return self.text(response.text)
        except ChatError:
            raise
        except errors.APIError as exc:
            if exc.code in {401, 403}:
                message = "Gemini rejected the API key or access. Check GEMINI_API_KEY and model access, then restart the backend."
            elif exc.code == 429:
                message = "Gemini's quota or rate limit was reached. Wait a moment or check your AI Studio quota."
            elif exc.code in {400, 404}:
                message = "Gemini rejected the model or request. Check GEMINI_API_KEY, LLM_MODEL and the agent model setting."
            elif exc.code == 408:
                message = "Cannot reach Gemini. Check your connection and try again."
            else:
                message = "Gemini could not complete the request. Please try again."
            raise ChatError(message) from None
        except (httpx.TransportError, TimeoutError, ConnectionError):
            raise ChatError("Cannot reach Gemini. Check your connection and try again.") from None
        except Exception:
            raise ChatError("Gemini could not complete the request. Please try again.") from None


def create_model_client(settings: ProviderSettings) -> ModelClient:
    if settings.configuration_error:
        raise ChatError(settings.configuration_error)
    adapter = GroqClient if settings.provider == "groq" else GeminiClient
    return adapter(api_key=settings.api_key, default_model=settings.default_model)
