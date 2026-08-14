"""Splitting transcripts into context-sized chunks for map-reduce summarization."""

from __future__ import annotations

import re
from collections.abc import Callable

# Sentence boundary, used only when a single transcript segment is itself too
# large for one chunk.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: Never build a chunk smaller than this, however tight the budget looks.
MIN_CHUNK_TOKENS = 256

#: Share of a chunk's budget repeated from the previous chunk.
DEFAULT_OVERLAP_RATIO = 0.05

TokenCounter = Callable[[str], int]


def split_units(text: str, budget: int, count_tokens: TokenCounter) -> list[str]:
    """Split *text* into units of at most *budget* tokens each.

    Splits on segment boundaries (lines) first and on sentence boundaries only
    when a segment does not fit on its own, so a sentence is never cut in the
    middle. Breaking between words is a last resort for a single sentence that
    is longer than a whole chunk.
    """
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if count_tokens(line) <= budget:
            units.append(line)
            continue
        for raw_sentence in _SENTENCE_RE.split(line):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            if count_tokens(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(_split_between_words(sentence, budget, count_tokens))
    return units


def _split_between_words(text: str, budget: int, count_tokens: TokenCounter) -> list[str]:
    """Break a single oversized sentence between words."""
    words = text.split()
    if not words:
        return []

    # Start from a proportional guess and halve until every piece fits, so the
    # result does not depend on how well the estimate matched the tokenizer.
    per_chunk = max(1, int(len(words) * budget / max(count_tokens(text), 1) * 0.9))
    while per_chunk > 1:
        pieces = [" ".join(words[i : i + per_chunk]) for i in range(0, len(words), per_chunk)]
        if all(count_tokens(piece) <= budget for piece in pieces):
            return pieces
        per_chunk //= 2
    return words


def _tail_within(units: list[str], overlap_budget: int, count_tokens: TokenCounter) -> list[str]:
    """Return the trailing units that together fit within *overlap_budget*."""
    if overlap_budget <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit)
        if total + unit_tokens > overlap_budget:
            break
        tail.insert(0, unit)
        total += unit_tokens
    return tail


def pack_chunks(
    units: list[str],
    budget: int,
    overlap_budget: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Greedily pack *units* into chunks of at most *budget* tokens.

    Every chunk after the first repeats the tail of its predecessor, so a topic
    that straddles a boundary still has context on both sides. The overlap is
    capped per boundary so the incoming unit is always guaranteed to fit.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > budget:
            chunks.append("\n".join(current))
            current = _tail_within(current, min(overlap_budget, budget - unit_tokens), count_tokens)
            current_tokens = sum(count_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_text(
    text: str,
    budget: int,
    count_tokens: TokenCounter,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[str]:
    """Split *text* into overlapping chunks that each fit *budget* tokens."""
    budget = max(budget, MIN_CHUNK_TOKENS)
    units = split_units(text, budget, count_tokens)
    if not units:
        return []
    return pack_chunks(units, budget, int(budget * overlap_ratio), count_tokens)


def truncate_to_tokens(text: str, max_tokens: int, count_tokens: TokenCounter) -> str:
    """Cut *text* down to at most *max_tokens* tokens."""
    if max_tokens <= 0:
        return ""
    total = count_tokens(text)
    if total <= max_tokens:
        return text

    cut = max(1, int(len(text) * max_tokens / max(total, 1)))
    while cut > 1 and count_tokens(text[:cut]) > max_tokens:
        cut = int(cut * 0.9)
    return text[:cut]
