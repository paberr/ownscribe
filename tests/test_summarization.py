"""Tests for summarization helpers and backends."""

from __future__ import annotations

from ownscribe.config import SummarizationConfig
from ownscribe.summarization.prompts import clean_response


class TestCleanResponse:
    def test_strips_think_tags(self):
        raw = "<think>reasoning about the meeting</think>\n## Summary\nclean"
        assert clean_response(raw) == "## Summary\nclean"

    def test_no_tags_unchanged(self):
        text = "## Summary\nNo thinking here."
        assert clean_response(text) == text

    def test_multiline_thinking_block(self):
        raw = "<think>\nline1\nline2\nline3\n</think>\n## Summary\nActual content"
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


class TestOllamaCustomPrompts:
    """Test that custom prompts are passed through to Ollama."""

    def test_custom_system_and_user_prompt(self, httpserver):
        import json

        response_body = {
            "message": {"role": "assistant", "content": "Custom summary."},
            "done": True,
        }
        httpserver.expect_request("/api/chat", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(
            host=httpserver.url_for(""),
            backend="ollama",
            model="test-model",
            system_prompt="You are a pirate.",
            prompt="Arr! Summarize: {transcript}",
        )

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        summarizer.summarize("Alice: Hello")

        # Check what was sent to the server
        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You are a pirate."
        assert body["messages"][1]["content"] == "Arr! Summarize: Alice: Hello"


class TestOpenAICustomPrompts:
    """Test that custom prompts are passed through to OpenAI."""

    def test_custom_system_and_user_prompt(self, httpserver):
        import json

        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Custom summary."},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(
            host=httpserver.url_for(""),
            backend="openai",
            model="test-model",
            system_prompt="You are a pirate.",
            prompt="Arr! Summarize: {transcript}",
        )

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        summarizer.summarize("Alice: Hello")

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You are a pirate."
        assert body["messages"][1]["content"] == "Arr! Summarize: Alice: Hello"


class TestOllamaSummarizer:
    """Test OllamaSummarizer against a mock HTTP server."""

    def test_summarize(self, httpserver):
        response_body = {
            "message": {"role": "assistant", "content": "<think>reasoning</think>\n## Summary\nMeeting went well."},
            "done": True,
        }
        httpserver.expect_request("/api/chat", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="ollama", model="test-model")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        result = summarizer.summarize("Alice: Hello\nBob: Hi")

        # Verify think tags are cleaned
        assert "<think>" not in result
        assert "## Summary" in result
        assert "Meeting went well." in result

    def test_is_available_success(self, httpserver):
        httpserver.expect_request("/api/tags", method="GET").respond_with_json({"models": []})

        config = SummarizationConfig(host=httpserver.url_for(""), backend="ollama", model="test-model")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        assert summarizer.is_available() is True

    def test_is_available_failure(self):
        config = SummarizationConfig(host="http://localhost:1", backend="ollama", model="test-model")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        assert summarizer.is_available() is False


class TestOpenAISummarizer:
    """Test OpenAISummarizer against a mock HTTP server."""

    def test_summarize(self, httpserver):
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "## Summary\nDecisions were made."},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        result = summarizer.summarize("Alice: Hello\nBob: Hi")

        assert "## Summary" in result
        assert "Decisions were made." in result

    def test_is_available_success(self, httpserver):
        httpserver.expect_request("/v1/models", method="GET").respond_with_json({"data": [], "object": "list"})

        config = SummarizationConfig(host=httpserver.url_for(""), backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        assert summarizer.is_available() is True

    def test_is_available_failure(self):
        config = SummarizationConfig(host="http://localhost:1", backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        assert summarizer.is_available() is False

    def test_summarize_cleans_think_tags(self, httpserver):
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>internal reasoning</think>\n## Summary\nCleaned output.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        result = summarizer.summarize("transcript text")

        assert "<think>" not in result
        assert "## Summary" in result
        assert "Cleaned output." in result
