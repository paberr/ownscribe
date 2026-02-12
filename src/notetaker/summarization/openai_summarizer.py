"""OpenAI-compatible API summarization (LM Studio, llama.cpp server, etc.)."""

from __future__ import annotations

import click
import openai

from notetaker.config import SummarizationConfig
from notetaker.summarization.base import Summarizer
from notetaker.summarization.prompts import MEETING_SUMMARY_PROMPT, MEETING_SUMMARY_SYSTEM, clean_response


class OpenAISummarizer(Summarizer):
    """Summarizes transcripts using an OpenAI-compatible API."""

    def __init__(self, config: SummarizationConfig) -> None:
        self._config = config
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
        prompt = MEETING_SUMMARY_PROMPT.format(transcript=transcript_text)
        click.echo(f"Summarizing with {self._config.model}...")

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": MEETING_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return clean_response(response.choices[0].message.content or "")
