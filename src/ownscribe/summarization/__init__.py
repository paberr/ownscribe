from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ownscribe.config import Config
    from ownscribe.summarization.base import Summarizer


_INSTALL_HINTS = {
    "openai": "uv add 'ownscribe[openai]'",
    "ollama": "uv add 'ownscribe[ollama]'",
}


def create_summarizer(config: Config) -> Summarizer:
    """Create the appropriate summarizer based on config."""
    backend = config.summarization.backend
    if backend == "local":
        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        return LlamaCppSummarizer(config.summarization, config.templates)
    try:
        if backend == "openai":
            from ownscribe.summarization.openai_summarizer import OpenAISummarizer

            return OpenAISummarizer(config.summarization, config.templates)
        else:
            from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

            return OllamaSummarizer(config.summarization, config.templates)
    except ImportError as exc:
        hint = _INSTALL_HINTS.get(backend, f"uv add 'ownscribe[{backend}]'")
        raise ImportError(
            f"The '{backend}' summarization backend requires additional dependencies.\n"
            f"Install with: {hint}"
        ) from exc
