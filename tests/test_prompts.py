"""Tests for prompt templates."""

from ownscribe.summarization.prompts import (
    MEETING_SUMMARY_PROMPT,
    MEETING_SUMMARY_SYSTEM,
    get_system_prompt,
    get_user_prompt,
)


class TestMeetingSummaryPrompt:
    def test_contains_transcript_placeholder(self):
        assert "{transcript}" in MEETING_SUMMARY_PROMPT

    def test_format_works(self):
        result = MEETING_SUMMARY_PROMPT.format(transcript="Hello, this is a test.")
        assert "Hello, this is a test." in result
        assert "{transcript}" not in result

    def test_system_prompt_is_nonempty(self):
        assert len(MEETING_SUMMARY_SYSTEM) > 0


class TestGetSystemPrompt:
    def test_returns_default_when_empty(self):
        assert get_system_prompt("") == MEETING_SUMMARY_SYSTEM

    def test_returns_custom_when_provided(self):
        custom = "You are a pirate."
        assert get_system_prompt(custom) == custom


class TestGetUserPrompt:
    def test_returns_default_when_empty(self):
        assert get_user_prompt("") == MEETING_SUMMARY_PROMPT

    def test_returns_custom_when_provided(self):
        custom = "Summarize this: {transcript}"
        assert get_user_prompt(custom) == custom
