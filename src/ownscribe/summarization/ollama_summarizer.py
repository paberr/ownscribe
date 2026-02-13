"""Ollama-based summarization."""

from __future__ import annotations

import click
import ollama

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import MEETING_SUMMARY_PROMPT, MEETING_SUMMARY_SYSTEM, clean_response


class OllamaSummarizer(Summarizer):
    """Summarizes transcripts using a local Ollama model."""

    def __init__(self, config: SummarizationConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.host)

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def summarize(self, transcript_text: str) -> str:
        prompt = MEETING_SUMMARY_PROMPT.format(transcript=transcript_text)
        click.echo(f"Summarizing with {self._config.model}...")

        response = self._client.chat(
            model=self._config.model,
            messages=[
                {"role": "system", "content": MEETING_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return clean_response(response["message"]["content"])
