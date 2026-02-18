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

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formats_to_try: list[dict | None] = [None]
        if json_mode:
            formats_to_try = [
                {"type": "json_object"},
                {"type": "json_schema", "json_schema": {
                    "name": "search_results",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"relevant": {"type": "array", "items": {"type": "string"}}},
                        "required": ["relevant"],
                        "additionalProperties": False,
                    },
                }},
                None,
            ]
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
