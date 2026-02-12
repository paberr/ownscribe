"""Prompt templates for meeting summarization."""

MEETING_SUMMARY_SYSTEM = """You are a meeting notes assistant. You produce clear, structured summaries of meeting transcripts."""

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
