"""Prompt templates for meeting summarization."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_ORPHAN_THINK_CLOSE_RE = re.compile(r"^[\s\S]*?</think>\s*", re.IGNORECASE)


def clean_response(text: str) -> str:
    """Strip reasoning/thinking tags from LLM responses."""
    text = _THINK_RE.sub("", text).strip()
    if "</think>" in text.lower():
        text = _ORPHAN_THINK_CLOSE_RE.sub("", text).strip()
    return text

MEETING_SUMMARY_SYSTEM = (
    "You are a meeting notes assistant. You produce clear, structured summaries of meeting transcripts."
)

MEETING_SUMMARY_PROMPT = """Summarize the following meeting transcript into structured meeting notes.

Include these sections:
## Summary
A brief 2-3 sentence overview of what the meeting was about.

## Key Points
Bullet points of the main topics discussed and decisions made.

## Action Items
Bullet points of any tasks, assignments, or follow-ups mentioned. Include who is responsible if mentioned.

## Decisions
Bullet points of any explicit decisions that were made.

---

Transcript:
{transcript}"""

LECTURE_SUMMARY_SYSTEM = (
    "You are an academic note-taking assistant. "
    "You produce clear, structured notes from lecture and seminar transcripts."
)

LECTURE_SUMMARY_PROMPT = """Summarize the following lecture transcript into structured notes.

Include these sections:
## Summary
A brief 2-3 sentence overview of what the lecture covered.

## Key Concepts
Bullet points of the main concepts, terms, and definitions discussed. Include brief explanations.

## Key Takeaways
Bullet points of the most important insights and conclusions.

---

Transcript:
{transcript}"""

BRIEF_SUMMARY_SYSTEM = (
    "You are a concise summarization assistant. You produce short, scannable summaries."
)

BRIEF_SUMMARY_PROMPT = """Summarize the following transcript into 3-5 concise bullet points.
Return only the bullet points, no headers or additional structure.

Transcript:
{transcript}"""


TEMPLATES: dict[str, dict[str, str]] = {
    "meeting": {"system": MEETING_SUMMARY_SYSTEM, "prompt": MEETING_SUMMARY_PROMPT},
    "lecture": {"system": LECTURE_SUMMARY_SYSTEM, "prompt": LECTURE_SUMMARY_PROMPT},
    "brief": {"system": BRIEF_SUMMARY_SYSTEM, "prompt": BRIEF_SUMMARY_PROMPT},
}


TITLE_SYSTEM = "You generate short meeting titles."

TITLE_PROMPT = (
    "Based on this meeting summary, generate a short title of 3-5 words. "
    "Return ONLY the title, nothing else. No quotes, no punctuation, no explanation."
    "\n\n{summary}"
)


def resolve_template(
    template_name: str, user_templates: dict | None = None,
) -> tuple[str, str]:
    """Resolve a template name to (system_prompt, user_prompt).

    Looks up user-defined templates first, then built-ins. Falls back to "meeting".
    A user template inherits missing fields from the built-in of the same name.
    """
    name = template_name or "meeting"
    user_templates = user_templates or {}
    builtin = TEMPLATES.get(name, TEMPLATES["meeting"])

    if name in user_templates:
        t = user_templates[name]
        return (
            t.system_prompt or builtin["system"],
            t.prompt or builtin["prompt"],
        )
    return builtin["system"], builtin["prompt"]


def list_templates() -> list[str]:
    """Return the names of all built-in templates."""
    return list(TEMPLATES.keys())


# --- Search prompts ---

SEARCH_FIND_SYSTEM = (
    "You are a meeting search assistant. Given meeting summaries and a question, "
    "identify which meetings are relevant to answering the question. "
    "You MUST choose only from the provided meeting IDs. "
    "Return at most 5 meetings per request. When in doubt, include the meeting "
    "— it is better to include a marginally relevant meeting than to miss one. "
    'Return a JSON object: {"relevant": ["id1", "id2"]}. '
    'If none are relevant, return {"relevant": []}.'
)

SEARCH_FIND_PROMPT = """Question: {question}

Meetings:
{summaries}

Return ONLY valid JSON: {{"relevant": ["meeting-id-1", "meeting-id-2"]}}"""

SEARCH_ANSWER_SYSTEM = (
    "You are a meeting assistant. Answer the user's question based on the meeting "
    "transcripts provided. Each transcript is prefixed with a header like "
    "'## [folder-name] Date Time — Title'. "
    "When quoting or referencing information, always cite the meeting it came from "
    "by including the meeting display name (e.g., '2026-02-13 15:01 — Quarterly Planning'). "
    "Use verbatim text from the transcript for quotes. "
    "Include the speaker label and timestamp for each quote. "
    "If the answer is not found in the transcripts, say so."
)

SEARCH_ANSWER_PROMPT = """Question: {question}

{transcripts}"""
