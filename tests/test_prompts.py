"""Tests for prompt templates."""

from ownscribe.summarization.prompts import MEETING_SUMMARY_PROMPT, MEETING_SUMMARY_SYSTEM


class TestMeetingSummaryPrompt:
    def test_contains_transcript_placeholder(self):
        assert "{transcript}" in MEETING_SUMMARY_PROMPT

    def test_format_works(self):
        result = MEETING_SUMMARY_PROMPT.format(transcript="Hello, this is a test.")
        assert "Hello, this is a test." in result
        assert "{transcript}" not in result

    def test_system_prompt_is_nonempty(self):
        assert len(MEETING_SUMMARY_SYSTEM) > 0
