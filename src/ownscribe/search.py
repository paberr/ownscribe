"""Semantic search over meeting notes using a two-stage LLM pipeline."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

import click

from ownscribe.config import Config
from ownscribe.progress import Spinner
from ownscribe.summarization import create_summarizer
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import (
    SEARCH_ANSWER_PROMPT,
    SEARCH_ANSWER_SYSTEM,
    SEARCH_FIND_PROMPT,
    SEARCH_FIND_SYSTEM,
)

_DEFAULT_CONTEXT_SIZE = 8192


class Meeting(NamedTuple):
    folder_name: str
    display_name: str
    summary_path: Path
    transcript_path: Path | None


def ask(config: Config, question: str, since: str | None, limit: int | None) -> None:
    """Run a two-stage search over meeting notes and print the answer."""
    output_dir = config.output.resolved_dir
    meetings, skipped = _discover_meetings(output_dir, since=since, limit=limit)

    if not meetings:
        click.echo("No meetings found.")
        return

    if skipped:
        click.echo(f"({skipped} meetings without summaries were skipped)")

    summarizer = create_summarizer(config)
    context_size = _resolve_context_size(config)

    # Stage 1
    label = f"Searching {len(meetings)} meetings"
    with Spinner(label) as spinner:
        relevant = _find_relevant_meetings(
            summarizer, question, meetings, context_size, spinner=spinner,
        )
        spinner.update(label)  # restore label so exit message is clean

    if not relevant:
        click.echo("No relevant meetings found for your question.")
        return

    click.echo(f"Found {len(relevant)} relevant meetings:")
    for m in relevant:
        click.echo(f"  - {m.display_name}")

    # Stage 2
    with Spinner("Analyzing transcripts"):
        answer, skipped_transcripts = _answer_from_transcripts(summarizer, question, relevant, context_size)
        answer = _verify_quotes(answer, _load_transcripts(relevant))

    if skipped_transcripts:
        click.echo(f"({skipped_transcripts} transcripts did not fit within context budget, they were skipped)")

    click.echo(answer)



def _resolve_context_size(config: Config) -> int:
    if config.summarization.context_size > 0:
        return config.summarization.context_size

    if config.summarization.backend == "ollama":
        try:
            import ollama

            client = ollama.Client(host=config.summarization.host)
            info = client.show(config.summarization.model)
            # Ollama returns model info with context window details
            model_info = info.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key:
                    return int(value)
        except Exception:
            pass

    return _DEFAULT_CONTEXT_SIZE


# -- Discovery --


_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(?:_(.+))?$")


def _parse_folder_name(name: str) -> tuple[str, str] | None:
    """Parse a folder name like '2026-02-13_1501_quarterly-planning'.

    Also accepts folders without a slug, e.g. '2026-02-16_1433'.
    Returns (date_str, display_name) or None if the name doesn't match.
    """
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    date_str = m.group(1)
    hour, minute = m.group(2), m.group(3)
    slug = m.group(4)
    if slug:
        title = slug.replace("-", " ").title()
        return date_str, f"{date_str} {hour}:{minute} — {title}"
    return date_str, f"{date_str} {hour}:{minute}"


def _discover_meetings(
    output_dir: Path, since: str | None, limit: int | None,
) -> tuple[list[Meeting], int]:
    if not output_dir.exists():
        return [], 0

    meetings: list[Meeting] = []
    skipped = 0

    for folder in sorted(output_dir.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        parsed = _parse_folder_name(folder.name)
        if parsed is None:
            continue

        date_str, display_name = parsed

        # Find summary file
        summary_path = None
        for ext in ("summary.md", "summary.json"):
            candidate = folder / ext
            if candidate.exists():
                summary_path = candidate
                break

        if summary_path is None:
            skipped += 1
            continue

        # Find transcript file
        transcript_path = None
        for ext in ("transcript.md", "transcript.json"):
            candidate = folder / ext
            if candidate.exists():
                transcript_path = candidate
                break

        # Apply --since filter
        if since:
            try:
                since_date = date.fromisoformat(since)
                meeting_date = date.fromisoformat(date_str)
                if meeting_date < since_date:
                    continue
            except ValueError:
                pass

        meetings.append(Meeting(folder.name, display_name, summary_path, transcript_path))

    # Apply --limit cap (meetings are already newest-first)
    if limit is not None and limit > 0:
        meetings = meetings[:limit]

    return meetings, skipped


# -- Token estimation --


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


# -- Chunking --


def _build_summary_chunks(
    meetings: list[Meeting], context_budget: int,
) -> list[list[Meeting]]:
    effective = int(context_budget * 0.8)
    overhead = 1000  # system prompt + question + response headroom
    budget = max(effective - overhead, 500)

    chunks: list[list[Meeting]] = []
    current_chunk: list[Meeting] = []
    current_size = 0

    for m in meetings:
        summary_text = m.summary_path.read_text()
        header = f"## [{m.folder_name}]\n"
        entry_tokens = _estimate_tokens(header + summary_text)

        if current_chunk and current_size + entry_tokens > budget:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(m)
        current_size += entry_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# -- Stage 1: Find relevant meetings --


_JSON_RE = re.compile(r'\{[^{}]*"relevant"[^{}]*\}', re.DOTALL)
_ARRAY_RE = re.compile(r'\[.*?\]', re.DOTALL)


def _parse_relevant_ids(response: str) -> list[str] | None:
    """Extract meeting IDs from a Stage 1 LLM response.

    Returns a list of IDs, or None if parsing fails entirely.
    """
    # Try direct JSON parse
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "relevant" in data:
            ids = data["relevant"]
            if isinstance(ids, list):
                return [str(i) for i in ids]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: regex for JSON object with "relevant"
    m = _JSON_RE.search(response)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "relevant" in data:
                ids = data["relevant"]
                if isinstance(ids, list):
                    return [str(i) for i in ids]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: bare array
    m = _ARRAY_RE.search(response)
    if m:
        try:
            ids = json.loads(m.group())
            if isinstance(ids, list):
                return [str(i) for i in ids]
        except (json.JSONDecodeError, TypeError):
            pass

    return None


_STOP_WORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for", "on",
    "with", "at", "by", "from", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "and", "but", "or", "if", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "am", "s", "t", "d", "ll", "ve", "re",
])


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from *text*, filtering stop words."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    filtered = words - _STOP_WORDS
    return filtered if filtered else words


def _keyword_fallback(
    question: str, meetings: list[Meeting],
) -> list[Meeting]:
    """Return meetings whose summary or transcript shares keywords with *question*."""
    kw = _extract_keywords(question)
    if not kw:
        return []

    matches: list[Meeting] = []
    for m in meetings:
        summary_text = m.summary_path.read_text()
        summary_words = set(re.findall(r"[a-z]+", summary_text.lower()))
        if kw & summary_words:
            matches.append(m)
            continue
        if m.transcript_path and m.transcript_path.exists():
            transcript_text = m.transcript_path.read_text()
            transcript_words = set(re.findall(r"[a-z]+", transcript_text.lower()))
            if kw & transcript_words:
                matches.append(m)
    return matches


def _find_relevant_meetings(
    summarizer: Summarizer,
    question: str,
    meetings: list[Meeting],
    context_size: int,
    spinner: Spinner | None = None,
) -> list[Meeting]:
    chunks = _build_summary_chunks(meetings, context_size)
    all_relevant_ids: set[str] = set()
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        if total_chunks > 1 and spinner is not None:
            spinner.update(f"Searching meetings (batch {i + 1}/{total_chunks})")

        known_ids = {m.folder_name for m in chunk}

        # Build summaries text
        summaries_parts = []
        for m in chunk:
            summary_text = m.summary_path.read_text()
            summaries_parts.append(f"## [{m.folder_name}]\n{summary_text}")
        summaries = "\n\n".join(summaries_parts)

        prompt = SEARCH_FIND_PROMPT.format(question=question, summaries=summaries)
        response = summarizer.chat(SEARCH_FIND_SYSTEM, prompt, json_mode=True)

        ids = _parse_relevant_ids(response)
        if ids is None:
            # Total fallback: include all from this chunk
            all_relevant_ids.update(known_ids)
        else:
            # Validate: only keep IDs that exist in this chunk
            all_relevant_ids.update(id_ for id_ in ids if id_ in known_ids)

    if not all_relevant_ids:
        keyword_matches = _keyword_fallback(question, meetings)
        if keyword_matches:
            if spinner is not None:
                spinner.update("Falling back to keyword search")
            return _rank_meetings(question, keyword_matches)

    # Build result list preserving order, then rank
    meeting_by_id = {m.folder_name: m for m in meetings}
    relevant = [meeting_by_id[id_] for id_ in all_relevant_ids if id_ in meeting_by_id]
    return _rank_meetings(question, relevant)


# -- Ranking --


def _rank_meetings(question: str, meetings: list[Meeting]) -> list[Meeting]:
    """Rank meetings by keyword overlap, speaker mention, and recency."""
    question_words = set(question.lower().split())

    def score(m: Meeting) -> tuple[float, float, str]:
        summary_text = m.summary_path.read_text().lower()
        summary_words = set(summary_text.split())

        # Keyword overlap
        overlap = len(question_words & summary_words)

        # Speaker boost: if a capitalized word in the question appears in the summary
        speaker_boost = 0.0
        for word in question.split():
            if word[0:1].isupper() and len(word) > 1 and word.lower() in summary_text:
                speaker_boost = 2.0
                break

        # Recency: folder name sorts lexicographically by date
        return (overlap + speaker_boost, speaker_boost, m.folder_name)

    return sorted(meetings, key=score, reverse=True)


# -- Stage 2: Answer from transcripts --


def _load_transcripts(meetings: list[Meeting]) -> dict[str, str]:
    """Load transcript text keyed by folder name."""
    transcripts: dict[str, str] = {}
    for m in meetings:
        if m.transcript_path and m.transcript_path.exists():
            transcripts[m.folder_name] = m.transcript_path.read_text()
    return transcripts


def _answer_from_transcripts(
    summarizer: Summarizer,
    question: str,
    meetings: list[Meeting],
    context_size: int,
) -> tuple[str, int]:
    effective = int(context_size * 0.8)
    overhead = 1500  # system prompt + question + response headroom
    budget = max(effective - overhead, 500)

    transcript_parts: list[str] = []
    used_tokens = 0
    included = 0
    skipped = 0

    for m in meetings:
        if m.transcript_path is None or not m.transcript_path.exists():
            skipped += 1
            continue

        text = m.transcript_path.read_text()
        entry = f"## [{m.folder_name}] {m.display_name}\n{text}"
        entry_tokens = _estimate_tokens(entry)

        if used_tokens + entry_tokens > budget:
            skipped += 1
            continue

        transcript_parts.append(entry)
        used_tokens += entry_tokens
        included += 1

    if not transcript_parts:
        return "No transcript text available for the relevant meetings.", skipped

    transcripts_text = "\n\n".join(transcript_parts)
    prompt = SEARCH_ANSWER_PROMPT.format(question=question, transcripts=transcripts_text)
    return summarizer.chat(SEARCH_ANSWER_SYSTEM, prompt), skipped


# -- Quote verification --


def _extract_quotes(text: str) -> list[str]:
    """Extract quoted text from the answer (> blockquotes and "..." quotes)."""
    quotes: list[str] = []

    # Blockquotes: lines starting with >
    blockquote_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            blockquote_lines.append(stripped.lstrip("> ").strip())
        else:
            if blockquote_lines:
                quotes.append(" ".join(blockquote_lines))
                blockquote_lines = []
    if blockquote_lines:
        quotes.append(" ".join(blockquote_lines))

    # Inline quotes: text in "..."
    for m in re.finditer(r'"([^"]{20,})"', text):
        quotes.append(m.group(1))

    return quotes


def _key_phrases(quote: str, min_words: int = 8, max_words: int = 12) -> list[str]:
    """Extract key phrases (8-12 word spans) from a quote for verification."""
    words = quote.split()
    if len(words) <= max_words:
        return [" ".join(words)] if len(words) >= 4 else []

    phrases = []
    # Take a few spans spread across the quote
    step = max(1, (len(words) - min_words) // 3)
    for start in range(0, len(words) - min_words + 1, step):
        end = min(start + max_words, len(words))
        phrases.append(" ".join(words[start:end]))
        if len(phrases) >= 3:
            break
    return phrases


def _verify_quotes(answer: str, transcripts: dict[str, str]) -> str:
    """Check quotes against transcripts and mark unverified ones."""
    if not transcripts:
        return answer

    all_text = "\n".join(transcripts.values()).lower()
    quotes = _extract_quotes(answer)

    unverified_quotes: set[str] = set()
    for quote in quotes:
        phrases = _key_phrases(quote)
        if not phrases:
            continue
        verified = any(phrase.lower() in all_text for phrase in phrases)
        if not verified:
            unverified_quotes.add(quote)

    if not unverified_quotes:
        return answer

    # Annotate unverified quotes in the answer
    result_lines: list[str] = []
    for line in answer.split("\n"):
        result_lines.append(line)
        stripped = line.strip()
        # Check if this line ends a blockquote that contains an unverified quote
        if stripped.startswith(">"):
            quote_text = stripped.lstrip("> ").strip()
            for uq in unverified_quotes:
                if quote_text and quote_text in uq:
                    result_lines.append("[unverified]")
                    break
    return "\n".join(result_lines)
