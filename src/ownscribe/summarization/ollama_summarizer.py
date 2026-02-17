"""Ollama-based summarization."""

from __future__ import annotations

import ollama

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import clean_response


class OllamaSummarizer(Summarizer):
    """Summarizes transcripts using a local Ollama model."""

    def __init__(self, config: SummarizationConfig, templates: dict | None = None) -> None:
        self._config = config
        self._templates = templates or {}
        self._client = ollama.Client(host=config.host)

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def summarize(self, transcript_text: str) -> str:
        from ownscribe.summarization.prompts import resolve_template

        system, prompt = resolve_template(self._config.template, self._templates)
        user = prompt.format(transcript=transcript_text)

        response = self._client.chat(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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
