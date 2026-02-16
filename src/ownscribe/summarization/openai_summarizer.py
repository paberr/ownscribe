"""OpenAI-compatible API summarization (LM Studio, llama.cpp server, etc.)."""

from __future__ import annotations

import openai

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import clean_response, get_system_prompt, get_user_prompt


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
        system = get_system_prompt(self._config.system_prompt)
        user = get_user_prompt(self._config.prompt).format(transcript=transcript_text)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return clean_response(response.choices[0].message.content or "")
