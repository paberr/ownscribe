"""Tests for prompt templates."""

from ownscribe.config import TemplateConfig
from ownscribe.summarization.prompts import (
    MEETING_SUMMARY_PROMPT,
    MEETING_SUMMARY_SYSTEM,
    TEMPLATES,
    list_templates,
    resolve_template,
)


class TestTemplates:
    def test_all_templates_have_transcript_placeholder(self):
        for name, t in TEMPLATES.items():
            assert "{transcript}" in t["prompt"], f"{name} template missing {{transcript}}"

    def test_all_templates_have_system_and_prompt(self):
        for name, t in TEMPLATES.items():
            assert t["system"], f"{name} template has empty system prompt"
            assert t["prompt"], f"{name} template has empty prompt"

    def test_list_templates_returns_all(self):
        names = list_templates()
        assert "meeting" in names
        assert "lecture" in names
        assert "brief" in names
        assert len(names) == 3


class TestResolveTemplate:
    def test_default_empty_name_returns_meeting(self):
        system, prompt = resolve_template("")
        assert system == MEETING_SUMMARY_SYSTEM
        assert prompt == MEETING_SUMMARY_PROMPT

    def test_meeting_by_name(self):
        system, prompt = resolve_template("meeting")
        assert system == MEETING_SUMMARY_SYSTEM
        assert prompt == MEETING_SUMMARY_PROMPT

    def test_lecture_by_name(self):
        system, prompt = resolve_template("lecture")
        assert "academic" in system.lower() or "lecture" in system.lower()
        assert "Key Concepts" in prompt

    def test_brief_by_name(self):
        system, prompt = resolve_template("brief")
        assert "concise" in system.lower()
        assert "bullet" in prompt.lower()

    def test_unknown_name_falls_back_to_meeting(self):
        system, prompt = resolve_template("nonexistent")
        assert system == MEETING_SUMMARY_SYSTEM
        assert prompt == MEETING_SUMMARY_PROMPT

    def test_user_defined_template(self):
        user_templates = {
            "custom": TemplateConfig(
                system_prompt="You are a pirate.",
                prompt="Arr! Summarize: {transcript}",
            ),
        }
        system, prompt = resolve_template("custom", user_templates)
        assert system == "You are a pirate."
        assert prompt == "Arr! Summarize: {transcript}"

    def test_partial_user_template_inherits_builtin(self):
        user_templates = {
            "meeting": TemplateConfig(
                system_prompt="",  # empty → inherit built-in
                prompt="Custom meeting prompt: {transcript}",
            ),
        }
        system, prompt = resolve_template("meeting", user_templates)
        assert system == MEETING_SUMMARY_SYSTEM  # inherited
        assert prompt == "Custom meeting prompt: {transcript}"  # overridden

    def test_user_template_overrides_builtin_of_same_name(self):
        user_templates = {
            "meeting": TemplateConfig(
                system_prompt="Custom system.",
                prompt="Custom prompt: {transcript}",
            ),
        }
        system, prompt = resolve_template("meeting", user_templates)
        assert system == "Custom system."
        assert prompt == "Custom prompt: {transcript}"


class TestMeetingSummaryPrompt:
    def test_contains_transcript_placeholder(self):
        assert "{transcript}" in MEETING_SUMMARY_PROMPT

    def test_format_works(self):
        result = MEETING_SUMMARY_PROMPT.format(transcript="Hello, this is a test.")
        assert "Hello, this is a test." in result
        assert "{transcript}" not in result

    def test_system_prompt_is_nonempty(self):
        assert len(MEETING_SUMMARY_SYSTEM) > 0
