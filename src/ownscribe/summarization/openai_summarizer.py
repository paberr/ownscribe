"""OpenAI-compatible API summarization (LM Studio, llama.cpp server, etc.)."""

from __future__ import annotations

import openai

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import clean_response


class OpenAISummarizer(Summarizer):
    """Summarizes transcripts using an OpenAI-compatible API."""

    def __init__(self, config: SummarizationConfig, templates: dict | None = None) -> None:
        self._config = config
        self._templates = templates or {}
        # For local servers, no API key needed — use a dummy
        base_url = config.host
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self._client = openai.OpenAI(base_url=base_url, api_key="not-needed")

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def summarize(self, transcript_text: str) -> str:
        from ownscribe.summarization.prompts import resolve_template

        system, prompt = resolve_template(self._config.template, self._templates)
        user = prompt.format(transcript=transcript_text)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return clean_response(response.choices[0].message.content or "")

    def generate_title(self, summary_text: str) -> str:
        from ownscribe.summarization.prompts import TITLE_PROMPT, TITLE_SYSTEM

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": TITLE_SYSTEM},
                {"role": "user", "content": TITLE_PROMPT.format(summary=summary_text)},
            ],
        )
        return clean_response(response.choices[0].message.content or "").strip()
