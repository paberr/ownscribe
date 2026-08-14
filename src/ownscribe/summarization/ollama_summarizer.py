"""Ollama-based summarization."""

from __future__ import annotations

import ollama

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import DEFAULT_CONTEXT_SIZE, Summarizer
from ownscribe.summarization.prompts import clean_response


class OllamaSummarizer(Summarizer):
    """Summarizes transcripts using a local Ollama model."""

    def __init__(self, config: SummarizationConfig, templates: dict | None = None) -> None:
        super().__init__(config, templates)
        self._client = ollama.Client(host=config.host)
        self._probed_context_size: int | None = None

    @property
    def context_size(self) -> int:
        if self._config.context_size > 0:
            return self._config.context_size
        if self._probed_context_size is None:
            self._probed_context_size = self._probe_context_size()
        return self._probed_context_size or DEFAULT_CONTEXT_SIZE

    def _probe_context_size(self) -> int:
        """Ask the server for the model's context window. Returns 0 if unknown."""
        try:
            info = self._client.show(self._config.model)
            model_info = info.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key:
                    return int(value)
        except Exception:
            pass
        return 0

    def chat(
        self, system_prompt: str, user_prompt: str,
        json_mode: bool = False, json_schema: dict | None = None,
    ) -> str:
        kwargs = {}
        if json_mode:
            kwargs["format"] = "json"
        response = self._client.chat(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return clean_response(response["message"]["content"])

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return clean_response(response["message"]["content"])

    def generate_title(self, summary_text: str) -> str:
        from ownscribe.summarization.prompts import TITLE_PROMPT, TITLE_SYSTEM

        response = self._client.chat(
            model=self._config.model,
            messages=[
                {"role": "system", "content": TITLE_SYSTEM},
                {"role": "user", "content": TITLE_PROMPT.format(summary=summary_text)},
            ],
        )
        return clean_response(response["message"]["content"]).strip()
