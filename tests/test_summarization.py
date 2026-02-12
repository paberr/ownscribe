"""Tests for summarization helpers."""

from notetaker.summarization.prompts import clean_response


class TestCleanResponse:
    def test_strips_think_tags(self):
        raw = "<think>reasoning about the meeting</think>\n## Summary\nclean"
        assert clean_response(raw) == "## Summary\nclean"

    def test_no_tags_unchanged(self):
        text = "## Summary\nNo thinking here."
        assert clean_response(text) == text

    def test_multiline_thinking_block(self):
        raw = (
            "<think>\nline1\nline2\nline3\n</think>\n"
            "## Summary\nActual content"
        )
        assert clean_response(raw) == "## Summary\nActual content"

    def test_case_insensitive(self):
        raw = "<THINK>stuff</THINK>\nresult"
        assert clean_response(raw) == "result"

    def test_empty_think_block(self):
        raw = "<think></think>result"
        assert clean_response(raw) == "result"

    def test_orphaned_close_think_tag(self):
        raw = "1. Analyze\n2. Plan\n</think>\n## Summary\nActual content"
        assert clean_response(raw) == "## Summary\nActual content"
