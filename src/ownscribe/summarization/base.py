"""Abstract base class for summarizers."""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

from ownscribe.summarization.chunking import (
    MIN_CHUNK_TOKENS,
    TokenCounter,
    chunk_text,
    truncate_to_tokens,
)
from ownscribe.summarization.prompts import (
    REDUCE_SUMMARY_PROMPT,
    REDUCE_SUMMARY_SYSTEM,
    format_partials,
    resolve_template,
)

if TYPE_CHECKING:
    from ownscribe.config import SummarizationConfig, TemplateConfig

logger = logging.getLogger(__name__)

#: Context window assumed when neither the config nor the backend reports one.
DEFAULT_CONTEXT_SIZE = 8192

# Prompt and completion share the context window, so a share of it has to stay
# free or the model runs out of room mid-sentence.
_COMPLETION_RESERVE = 1024
_MIN_COMPLETION_RESERVE = 256


def estimate_tokens(text: str) -> int:
    """Rough token count for backends without a real tokenizer (~4 chars/token)."""
    return len(text) // 4


def _group_by_budget(items: list[str], budget: int, count_tokens: TokenCounter) -> list[list[str]]:
    """Greedily group *items* so each group stays within *budget* tokens."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for item in items:
        item_tokens = count_tokens(item)
        if current and current_tokens + item_tokens > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens

    if current:
        groups.append(current)
    return groups


class Summarizer(abc.ABC):
    """Base class for summarization backends.

    Subclasses provide the transport (`_complete`, `chat`, `is_available`);
    everything about fitting a transcript into the context window lives here, so
    a long meeting is summarized correctly whatever backend runs it.
    """

    def __init__(
        self,
        config: SummarizationConfig,
        templates: dict[str, TemplateConfig] | None = None,
    ) -> None:
        self._config = config
        self._templates = templates or {}

    # -- context budgeting ---------------------------------------------------

    @property
    def context_size(self) -> int:
        """Usable context window in tokens, covering prompt and completion together."""
        if self._config.context_size > 0:
            return self._config.context_size
        return DEFAULT_CONTEXT_SIZE

    def count_tokens(self, text: str) -> int:
        """Count the tokens in *text*. Backends with a real tokenizer override this."""
        return estimate_tokens(text)

    def completion_reserve(self) -> int:
        """Tokens kept free for the model's own output."""
        return min(_COMPLETION_RESERVE, max(_MIN_COMPLETION_RESERVE, self.context_size // 4))

    def _input_budget(self, system_prompt: str, prompt_template: str) -> int:
        """Tokens left for transcript text once scaffolding and headroom are paid for."""
        scaffold = system_prompt + prompt_template.replace("{transcript}", "").replace("{partials}", "")
        budget = self.context_size - self.completion_reserve() - self.count_tokens(scaffold)
        return max(budget, MIN_CHUNK_TOKENS)

    # -- summarization -------------------------------------------------------

    def summarize(self, transcript_text: str) -> str:
        """Summarize a transcript, splitting it up when it does not fit the window."""
        system, prompt = resolve_template(self._config.template, self._templates)
        text = transcript_text.strip()
        budget = self._input_budget(system, prompt)

        if self.count_tokens(text) <= budget:
            return self._complete(system, prompt.format(transcript=text))

        chunks = chunk_text(text, budget, self.count_tokens)
        if not chunks:
            return ""
        if len(chunks) == 1:
            return self._complete(system, prompt.format(transcript=chunks[0]))

        logger.info("Transcript exceeds the context window; summarizing in %d chunks", len(chunks))
        partials = [self._complete(system, prompt.format(transcript=chunk)) for chunk in chunks]
        return self._reduce(system, [p for p in partials if p.strip()])

    def _reduce(self, template_system: str, partials: list[str]) -> str:
        """Consolidate per-chunk summaries into one, under the same template.

        Reduces repeatedly when the partials do not fit a single pass, so the
        result is independent of how many chunks the transcript needed.
        """
        if not partials:
            return ""

        # Keep the template's own system prompt so a custom persona and section
        # layout survive the reduce step.
        system = f"{template_system}\n\n{REDUCE_SUMMARY_SYSTEM}"
        budget = self._input_budget(system, REDUCE_SUMMARY_PROMPT)

        while len(partials) > 1:
            groups = _group_by_budget(partials, budget, self.count_tokens)
            if len(groups) >= len(partials):
                # No two partials fit together: halve the list by force, trimming
                # each one so a pair is guaranteed to fit and the loop terminates.
                half = max(budget // 2, MIN_CHUNK_TOKENS)
                trimmed = [truncate_to_tokens(p, half, self.count_tokens) for p in partials]
                groups = [trimmed[i : i + 2] for i in range(0, len(trimmed), 2)]
            partials = [self._reduce_group(system, group) for group in groups]

        return partials[0]

    def _reduce_group(self, system: str, group: list[str]) -> str:
        """Merge one group of partial summaries; a lone partial passes through."""
        if len(group) == 1:
            return group[0]
        return self._complete(system, REDUCE_SUMMARY_PROMPT.format(partials=format_partials(group)))

    # -- backend hooks -------------------------------------------------------

    @abc.abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion. Errors propagate so the caller can report them."""

    @abc.abstractmethod
    def generate_title(self, summary_text: str) -> str:
        """Generate a short meeting title from a summary."""

    @abc.abstractmethod
    def chat(
        self, system_prompt: str, user_prompt: str,
        json_mode: bool = False, json_schema: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the summarization backend is reachable."""

    def close(self) -> None:  # noqa: B027 — intentional optional hook, not abstract
        """Release any native resources. No-op by default; must be idempotent."""

    def __enter__(self) -> Summarizer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
