from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ownscribe.config import Config
    from ownscribe.summarization.base import Summarizer


def create_summarizer(config: Config) -> Summarizer:
    """Create the appropriate summarizer based on config."""
    if config.summarization.backend == "local":
        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        return LlamaCppSummarizer(config.summarization, config.templates)
    elif config.summarization.backend == "openai":
        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        return OpenAISummarizer(config.summarization, config.templates)
    else:
        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        return OllamaSummarizer(config.summarization, config.templates)
