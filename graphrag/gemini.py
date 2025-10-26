"""Google Gemini powered response generation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


_DOTENV_LOADED = False


class GeminiImportError(RuntimeError):
    """Raised when the Gemini SDK is not installed."""


@dataclass
class GeminiConfig:
    """Configuration needed to call the Gemini API."""

    api_key: str
    model: str = "models/gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        self.model = _normalise_model_name(self.model)

    @classmethod
    def from_env(
        cls,
        env_var: str = "GOOGLE_GEMINI_API_KEY",
        *,
        model: str = "models/gemini-2.0-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> "GeminiConfig":
        global _DOTENV_LOADED
        if load_dotenv and not _DOTENV_LOADED:
            load_dotenv()
            _DOTENV_LOADED = True

        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(
                f"Gemini API key not found. Set the {env_var} environment variable."
            )
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def _normalise_model_name(name: str) -> str:
    if name.startswith("models/"):
        return name
    return f"models/{name}"


class GeminiAnswerGenerator:
    """Wraps the google-generativeai SDK to produce answers with retrieved context."""

    def __init__(self, config: GeminiConfig):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise GeminiImportError(
                "google-generativeai is required. Install with "
                "`pip install google-generativeai`."
            ) from exc

        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(_normalise_model_name(config.model))
        self._temperature = config.temperature
        self._max_output_tokens = config.max_output_tokens

    def build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        context_block = "\n\n".join(
            f"[{idx + 1}] {context.strip()}" for idx, context in enumerate(contexts)
        )
        prompt = (
            "You are an assistant who answers questions using the provided context. "
            "Cite specific evidence when possible and keep the tone instructional.\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Context passages:\n{context_block}\n\n"
            "Instructions:\n"
            "- Respond in Markdown.\n"
            "- Begin with a concise answer paragraph.\n"
            "- Follow with 2-3 bullet points highlighting supporting evidence.\n"
            "- If the context lacks the answer, state that explicitly."
        )
        return prompt

    def answer(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return "No retrieved context was provided for Gemini."
        prompt = self.build_prompt(question, contexts)

        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_output_tokens,
            },
        )
        text = _extract_primary_text(response)
        if not text.strip():
            finish_reason = _first_finish_reason(response)
            if finish_reason:
                return (
                    "Gemini did not return text content. "
                    f"Finish reason: {finish_reason}. "
                    "Try lowering top-k, reducing prompt size, or increasing --gemini-max-output-tokens."
                )
            return "Gemini returned an empty response."
        return text.strip()


def _extract_primary_text(response: object) -> str:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        text_fragments = [
            getattr(part, "text", "")
            for part in parts
            if getattr(part, "text", "")
        ]
        if text_fragments:
            return "\n".join(text_fragments)
    return ""


def _first_finish_reason(response: object) -> Optional[str]:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            return str(reason)
    return None


__all__ = ["GeminiAnswerGenerator", "GeminiConfig", "GeminiImportError"]
