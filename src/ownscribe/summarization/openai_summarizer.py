"""OpenAI-compatible API summarization (LM Studio, llama.cpp server, etc.)."""

from __future__ import annotations

import openai

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import clean_response


class OpenAISummarizer(Summarizer):
    """Summarizes transcripts using an OpenAI-compatible API."""

    def __init__(self, config: SummarizationConfig, templates: dict | None = None) -> None:
        super().__init__(config, templates)
        base_url = config.host
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        # Most local servers ignore the key; servers that require auth (e.g. oMLX)
        # read it from config.api_key or the OPENAI_API_KEY env var.
        api_key = config.api_key or "not-needed"
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self, system_prompt: str, user_prompt: str,
        json_mode: bool = False, json_schema: dict | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formats_to_try: list[dict | None] = [None]
        if json_mode:
            if json_schema is not None:
                formats_to_try = [
                    {"type": "json_object"},
                    {"type": "json_schema", "json_schema": json_schema},
                    None,
                ]
            else:
                formats_to_try = [{"type": "json_object"}, None]
        for fmt in formats_to_try:
            try:
                kwargs = {}
                if fmt is not None:
                    kwargs["response_format"] = fmt
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    **kwargs,
                )
                return clean_response(response.choices[0].message.content or "")
            except openai.BadRequestError:
                continue
        return ""

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
