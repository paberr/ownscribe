"""Tests for summarization helpers and backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ownscribe.config import Config, SummarizationConfig, TemplateConfig
from ownscribe.summarization import create_summarizer
from ownscribe.summarization.prompts import (
    LECTURE_SUMMARY_SYSTEM,
    clean_response,
)


class TestCreateSummarizerMissingDeps:
    """Test that create_summarizer raises helpful errors when optional deps are missing."""

    @pytest.mark.parametrize(
        "backend,module,extra",
        [
            ("ollama", "ownscribe.summarization.ollama_summarizer", "ollama"),
            ("openai", "ownscribe.summarization.openai_summarizer", "openai"),
            ("local", "ownscribe.summarization.llama_cpp_summarizer", "local"),
        ],
    )
    def test_missing_backend_dep(self, backend, module, extra):
        config = Config()
        config.summarization.backend = backend

        with patch.dict("sys.modules", {module: None}), pytest.raises(ImportError, match=f"ownscribe\\[{extra}\\]"):
            create_summarizer(config)


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
    """Test that custom prompts via user-defined templates are passed through to Ollama."""

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
            template="pirate",
        )
        templates = {
            "pirate": TemplateConfig(
                system_prompt="You are a pirate.",
                prompt="Arr! Summarize: {transcript}",
            ),
        }

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config, templates)
        summarizer.summarize("Alice: Hello")

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You are a pirate."
        assert body["messages"][1]["content"] == "Arr! Summarize: Alice: Hello"


class TestOpenAICustomPrompts:
    """Test that custom prompts via user-defined templates are passed through to OpenAI."""

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
            template="pirate",
        )
        templates = {
            "pirate": TemplateConfig(
                system_prompt="You are a pirate.",
                prompt="Arr! Summarize: {transcript}",
            ),
        }

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config, templates)
        summarizer.summarize("Alice: Hello")

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You are a pirate."
        assert body["messages"][1]["content"] == "Arr! Summarize: Alice: Hello"


class TestOllamaTemplatePassthrough:
    """Test that built-in templates are resolved correctly by Ollama."""

    def test_lecture_template(self, httpserver):
        import json

        response_body = {
            "message": {"role": "assistant", "content": "Lecture notes."},
            "done": True,
        }
        httpserver.expect_request("/api/chat", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(
            host=httpserver.url_for(""),
            backend="ollama",
            model="test-model",
            template="lecture",
        )

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        summarizer.summarize("Today we discuss photosynthesis.")

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == LECTURE_SUMMARY_SYSTEM
        assert "Today we discuss photosynthesis." in body["messages"][1]["content"]
        assert "Key Concepts" in body["messages"][1]["content"]


class TestOpenAITemplatePassthrough:
    """Test that built-in templates are resolved correctly by OpenAI."""

    def test_lecture_template(self, httpserver):
        import json

        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Lecture notes."},
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
            template="lecture",
        )

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        summarizer.summarize("Today we discuss photosynthesis.")

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == LECTURE_SUMMARY_SYSTEM
        assert "Today we discuss photosynthesis." in body["messages"][1]["content"]
        assert "Key Concepts" in body["messages"][1]["content"]


class TestOllamaGenerateTitle:
    """Test OllamaSummarizer.generate_title against a mock HTTP server."""

    def test_generate_title(self, httpserver):
        import json

        response_body = {
            "message": {"role": "assistant", "content": "Q3 Budget Review"},
            "done": True,
        }
        httpserver.expect_request("/api/chat", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="ollama", model="test-model")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        result = summarizer.generate_title("The meeting covered Q3 budget.")

        assert result == "Q3 Budget Review"

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You generate short meeting titles."
        assert "Q3 budget" in body["messages"][1]["content"]

    def test_generate_title_strips_think_tags(self, httpserver):
        response_body = {
            "message": {"role": "assistant", "content": "<think>hmm</think>\nBudget Planning"},
            "done": True,
        }
        httpserver.expect_request("/api/chat", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="ollama", model="test-model")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        result = summarizer.generate_title("summary text")

        assert "<think>" not in result
        assert result == "Budget Planning"


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


class TestOpenAIGenerateTitle:
    """Test OpenAISummarizer.generate_title against a mock HTTP server."""

    def test_generate_title(self, httpserver):
        import json

        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Q3 Budget Review"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        result = summarizer.generate_title("The meeting covered Q3 budget.")

        assert result == "Q3 Budget Review"

        request = httpserver.log[0][0]
        body = json.loads(request.data)
        assert body["messages"][0]["content"] == "You generate short meeting titles."
        assert "Q3 budget" in body["messages"][1]["content"]

    def test_generate_title_strips_think_tags(self, httpserver):
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "<think>hmm</think>\nBudget Planning"},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(response_body)

        config = SummarizationConfig(host=httpserver.url_for(""), backend="openai", model="test-model")

        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        summarizer = OpenAISummarizer(config)
        result = summarizer.generate_title("summary text")

        assert "<think>" not in result
        assert result == "Budget Planning"


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


# ---------------------------------------------------------------------------
# LlamaCppSummarizer tests
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> dict:
    """Build a fake llama-cpp create_chat_completion return value."""
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture()
def mock_llama():
    """Patch llama_cpp.Llama and _ensure_model so no real model is loaded."""
    llm_instance = MagicMock()
    with (
        patch(
            "ownscribe.summarization.llama_cpp_summarizer._ensure_model",
            return_value="/fake/model.gguf",
        ),
        patch(
            "ownscribe.summarization.llama_cpp_summarizer.Llama",
            return_value=llm_instance,
            create=True,
        ) as llama_cls,
        patch(
            "llama_cpp.Llama",
            llama_cls,
            create=True,
        ),
    ):
        yield llm_instance


class TestLlamaCppSummarizer:
    """Test LlamaCppSummarizer.summarize."""

    def test_summarize(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("## Summary\nMeeting went well.")

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.summarize("Alice: Hello\nBob: Hi")

        assert "## Summary" in result
        assert "Meeting went well." in result
        mock_llama.create_chat_completion.assert_called_once()

    def test_summarize_cleans_think_tags(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response(
            "<think>reasoning</think>\n## Summary\nCleaned."
        )

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.summarize("transcript")

        assert "<think>" not in result
        assert "## Summary" in result
        assert "Cleaned." in result

    def test_is_available(self, mock_llama):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        assert summarizer.is_available() is True

    def test_is_available_without_llama_cpp(self):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        with patch.dict("sys.modules", {"llama_cpp": None}):
            assert summarizer.is_available() is False


class TestLlamaCppGenerateTitle:
    """Test LlamaCppSummarizer.generate_title."""

    def test_generate_title(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("Q3 Budget Review")

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.generate_title("The meeting covered Q3 budget.")

        assert result == "Q3 Budget Review"
        call_args = mock_llama.create_chat_completion.call_args
        assert call_args[1]["messages"][0]["content"] == "You generate short meeting titles."
        assert "Q3 budget" in call_args[1]["messages"][1]["content"]

    def test_generate_title_strips_think_tags(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("<think>hmm</think>\nBudget Planning")

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.generate_title("summary text")

        assert "<think>" not in result
        assert result == "Budget Planning"


class TestLlamaCppChat:
    """Test LlamaCppSummarizer.chat with json_mode and json_schema."""

    def test_chat(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("Hello!")

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.chat("system", "user")

        assert result == "Hello!"
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        assert "response_format" not in call_kwargs

    def test_chat_json_mode(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response('{"key": "value"}')

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.chat("system", "user", json_mode=True)

        assert result == '{"key": "value"}'
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_chat_json_schema_fallback(self, mock_llama):
        """When json_schema format fails, should fall back to json_object."""
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        # First call with schema raises, second with json_object succeeds
        mock_llama.create_chat_completion.side_effect = [
            Exception("schema not supported"),
            _mock_llm_response('{"key": "val"}'),
        ]

        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        result = summarizer.chat("system", "user", json_mode=True, json_schema=schema)

        assert result == '{"key": "val"}'
        assert mock_llama.create_chat_completion.call_count == 2


class TestLlamaCppCustomPrompts:
    """Test that custom prompts via user-defined templates are passed through."""

    def test_custom_system_and_user_prompt(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("Custom summary.")

        config = SummarizationConfig(backend="local", model="phi-4-mini", template="pirate")
        templates = {
            "pirate": TemplateConfig(
                system_prompt="You are a pirate.",
                prompt="Arr! Summarize: {transcript}",
            ),
        }

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config, templates)
        summarizer.summarize("Alice: Hello")

        call_args = mock_llama.create_chat_completion.call_args
        assert call_args[1]["messages"][0]["content"] == "You are a pirate."
        assert call_args[1]["messages"][1]["content"] == "Arr! Summarize: Alice: Hello"


class TestLlamaCppTemplatePassthrough:
    """Test that built-in templates are resolved correctly."""

    def test_lecture_template(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("Lecture notes.")

        config = SummarizationConfig(backend="local", model="phi-4-mini", template="lecture")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        summarizer.summarize("Today we discuss photosynthesis.")

        call_args = mock_llama.create_chat_completion.call_args
        assert call_args[1]["messages"][0]["content"] == LECTURE_SUMMARY_SYSTEM
        assert "Today we discuss photosynthesis." in call_args[1]["messages"][1]["content"]
        assert "Key Concepts" in call_args[1]["messages"][1]["content"]


class TestEnsureModel:
    """Test _ensure_model with various model specifications."""

    def test_hf_prefix_parsing(self):
        with patch(
            "huggingface_hub.hf_hub_download",
            return_value="/fake/path.gguf",
        ) as mock_dl:
            from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

            result = _ensure_model("hf:myorg/myrepo/model.gguf")

        mock_dl.assert_called_once_with(repo_id="myorg/myrepo", filename="model.gguf")
        assert str(result) == "/fake/path.gguf"

    def test_hf_prefix_invalid(self):
        from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

        with pytest.raises(ValueError, match="Invalid HuggingFace model spec"):
            _ensure_model("hf:noslash")

    def test_registry_lookup(self):
        with patch(
            "huggingface_hub.hf_hub_download",
            return_value="/fake/phi.gguf",
        ) as mock_dl:
            from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

            result = _ensure_model("phi-4-mini")

        mock_dl.assert_called_once_with(
            repo_id="unsloth/Phi-4-mini-instruct-GGUF",
            filename="Phi-4-mini-instruct-Q4_K_M.gguf",
        )
        assert str(result) == "/fake/phi.gguf"

    def test_direct_path(self, tmp_path):
        model_file = tmp_path / "my_model.gguf"
        model_file.touch()

        from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

        result = _ensure_model(str(model_file))
        assert result == model_file

    def test_unknown_model(self):
        from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

        with pytest.raises(FileNotFoundError, match="Unknown model"):
            _ensure_model("nonexistent-model")
