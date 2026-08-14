"""Tests for summarization helpers and backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ownscribe.config import Config, SummarizationConfig, TemplateConfig
from ownscribe.summarization import create_summarizer
from ownscribe.summarization.base import DEFAULT_CONTEXT_SIZE, Summarizer
from ownscribe.summarization.chunking import MIN_CHUNK_TOKENS
from ownscribe.summarization.prompts import (
    LECTURE_SUMMARY_SYSTEM,
    clean_response,
    resolve_template,
)


def _request_body(httpserver, path: str) -> dict:
    """Body of the first request the fake server received for *path*.

    Ollama probes /api/show for the model's context window, so the chat request
    is not reliably the first one in the log.
    """
    import json

    for request, _ in httpserver.log:
        if request.path == path:
            return json.loads(request.data)
    raise AssertionError(f"no request recorded for {path}")


class TestCreateSummarizerMissingDeps:
    """Test that create_summarizer raises helpful errors when optional deps are missing."""

    @pytest.mark.parametrize(
        "backend,module,extra",
        [
            ("ollama", "ownscribe.summarization.ollama_summarizer", "ollama"),
            ("openai", "ownscribe.summarization.openai_summarizer", "openai"),
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

        body = _request_body(httpserver, "/api/chat")
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

        body = _request_body(httpserver, "/api/chat")
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


class TestLlamaCppClose:
    """Test deterministic cleanup of the local model."""

    def test_close_frees_loaded_model(self, mock_llama):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        summarizer._get_llm()
        summarizer.close()

        mock_llama.close.assert_called_once()
        assert summarizer._llm is None

    def test_close_is_idempotent(self, mock_llama):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        summarizer._get_llm()
        summarizer.close()
        summarizer.close()

        mock_llama.close.assert_called_once()

    def test_close_without_load_is_noop(self, mock_llama):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        summarizer.close()

        mock_llama.close.assert_not_called()

    def test_close_suppresses_errors(self, mock_llama):
        mock_llama.close.side_effect = RuntimeError("boom")
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(config)
        summarizer._get_llm()
        summarizer.close()

        assert summarizer._llm is None

    def test_context_manager_closes_model(self, mock_llama):
        config = SummarizationConfig(backend="local", model="phi-4-mini")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        with LlamaCppSummarizer(config) as summarizer:
            assert summarizer._get_llm() is mock_llama

        mock_llama.close.assert_called_once()


class TestSummarizerCloseContract:
    """Backends without native resources inherit a no-op close + context manager."""

    def test_ollama_close_is_noop_and_context_manager(self):
        config = SummarizationConfig(host="http://localhost:1", backend="ollama", model="x")

        from ownscribe.summarization.ollama_summarizer import OllamaSummarizer

        summarizer = OllamaSummarizer(config)
        with summarizer as entered:
            assert entered is summarizer
        summarizer.close()


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


@pytest.fixture()
def mock_llama_cls():
    """Like mock_llama, but exposes the patched Llama class for kwarg assertions."""
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
        patch("llama_cpp.Llama", llama_cls, create=True),
    ):
        yield llama_cls, llm_instance


class TestLlamaCppContextSize:
    """n_ctx must follow config.summarization.context_size, not a hardcoded 8192."""

    def test_n_ctx_uses_configured_context_size(self, mock_llama_cls):
        llama_cls, _llm = mock_llama_cls
        config = SummarizationConfig(backend="local", model="phi-4-mini", context_size=32768)

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        LlamaCppSummarizer(config)._get_llm()

        assert llama_cls.call_args.kwargs["n_ctx"] == 32768

    def test_n_ctx_defaults_when_unset(self, mock_llama_cls):
        llama_cls, _llm = mock_llama_cls
        config = SummarizationConfig(backend="local", model="phi-4-mini", context_size=0)

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        LlamaCppSummarizer(config)._get_llm()

        assert llama_cls.call_args.kwargs["n_ctx"] == DEFAULT_CONTEXT_SIZE


class TestLlamaCppTokenCounting:
    def test_count_tokens_uses_the_model_tokenizer(self, mock_llama_cls):
        _llama_cls, llm = mock_llama_cls
        llm.tokenize.return_value = [1, 2, 3, 4, 5]

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(SummarizationConfig(backend="local"))

        assert summarizer.count_tokens("some transcript text") == 5
        llm.tokenize.assert_called_with(b"some transcript text", add_bos=False, special=False)

    def test_count_tokens_falls_back_when_the_tokenizer_fails(self, mock_llama_cls):
        _llama_cls, llm = mock_llama_cls
        llm.tokenize.side_effect = RuntimeError("no tokenizer")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(SummarizationConfig(backend="local"))

        assert summarizer.count_tokens("a" * 100) == 25

    def test_empty_text_is_zero_tokens(self, mock_llama_cls):
        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        assert LlamaCppSummarizer(SummarizationConfig(backend="local")).count_tokens("") == 0


class TestLlamaCppCompletionHeadroom:
    def test_completion_is_capped_so_the_prompt_cannot_eat_the_window(self, mock_llama):
        mock_llama.create_chat_completion.return_value = _mock_llm_response("## Summary\nDone.")

        from ownscribe.summarization.llama_cpp_summarizer import LlamaCppSummarizer

        summarizer = LlamaCppSummarizer(SummarizationConfig(backend="local", context_size=8192))
        summarizer.summarize("SPEAKER_00: Short meeting.")

        kwargs = mock_llama.create_chat_completion.call_args.kwargs
        assert kwargs["max_tokens"] == summarizer.completion_reserve()
        assert 0 < kwargs["max_tokens"] < 8192


# -- Context budgeting and map-reduce chunking --


class _RecordingSummarizer(Summarizer):
    """A Summarizer that records prompts instead of calling a model.

    Counts one token per word, so budgets in these tests read literally.
    """

    def __init__(self, config, templates=None, replies: list[str] | None = None):
        super().__init__(config, templates)
        self.completions: list[tuple[str, str]] = []
        self._replies = list(replies or [])

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        self.completions.append((system_prompt, user_prompt))
        if self._replies:
            return self._replies[min(len(self.completions), len(self._replies)) - 1]
        return f"partial {len(self.completions)}"

    def generate_title(self, summary_text: str) -> str:
        return "Title"

    def chat(self, system_prompt, user_prompt, json_mode=False, json_schema=None) -> str:
        return ""

    def is_available(self) -> bool:
        return True


def _transcript(words: int) -> str:
    """A speaker-labelled transcript of at least *words* words."""
    lines: list[str] = []
    written = 0
    i = 0
    while written < words:
        line = f"SPEAKER_0{i % 2}: We discussed topic number {i} at some length today."
        lines.append(line)
        written += len(line.split())
        i += 1
    return "\n".join(lines)


class TestContextBudget:
    def test_context_size_follows_config(self):
        config = SummarizationConfig(backend="local", context_size=32768)
        assert _RecordingSummarizer(config).context_size == 32768

    def test_context_size_defaults_when_unset(self):
        config = SummarizationConfig(backend="local", context_size=0)
        assert _RecordingSummarizer(config).context_size == DEFAULT_CONTEXT_SIZE

    def test_completion_headroom_is_reserved(self):
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=8192))

        reserve = summarizer.completion_reserve()
        system, prompt = resolve_template("meeting", {})

        assert reserve > 0
        # The prompt can never claim the whole window: headroom and scaffolding
        # are both subtracted before any transcript text is allowed in.
        assert summarizer._input_budget(system, prompt) <= 8192 - reserve


class TestSummarizeSingleShot:
    def test_short_transcript_is_one_call(self):
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=8192))

        result = summarizer.summarize("SPEAKER_00: Short meeting, nothing to report.")

        assert len(summarizer.completions) == 1
        _system, user = summarizer.completions[0]
        assert "SPEAKER_00: Short meeting, nothing to report." in user
        assert result == "partial 1"

    def test_transcript_that_fits_is_not_chunked(self):
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=8192))
        text = _transcript(500)

        summarizer.summarize(text)

        assert len(summarizer.completions) == 1
        # Passed through whole, not reassembled from chunks.
        assert text in summarizer.completions[0][1]


class TestSummarizeMapReduce:
    def test_long_transcript_maps_then_reduces(self):
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=1000))

        result = summarizer.summarize(_transcript(3000))

        # Several map calls over chunks, then one reduce over the partials.
        assert len(summarizer.completions) > 2
        reduce_system, reduce_user = summarizer.completions[-1]
        assert "part 1 of" in reduce_user
        assert result == f"partial {len(summarizer.completions)}"
        assert "consolidate" in reduce_system.lower()

    def test_no_prompt_exceeds_the_context_window(self):
        # The regression: a long meeting used to be formatted into one prompt
        # that overflowed n_ctx, and llama-cpp raised instead of truncating.
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=1000))

        summarizer.summarize(_transcript(5000))

        for system, user in summarizer.completions:
            used = summarizer.count_tokens(system) + summarizer.count_tokens(user)
            assert used + summarizer.completion_reserve() <= 1000

    def test_every_part_of_the_transcript_is_summarized(self):
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=1000))
        text = _transcript(3000)

        summarizer.summarize(text)

        mapped = " ".join(user for _system, user in summarizer.completions[:-1])
        for i in range(len(text.splitlines())):
            assert f"topic number {i} at" in mapped

    def test_reduce_runs_repeatedly_when_partials_do_not_fit(self):
        # Small window and bulky partials: one reduce pass cannot hold them all,
        # so the partials have to be folded together over several rounds.
        summarizer = _RecordingSummarizer(
            SummarizationConfig(context_size=1500),
            replies=["partial notes " * 100],
        )

        result = summarizer.summarize(_transcript(6000))

        assert result
        reduce_calls = [user for _system, user in summarizer.completions if "part 1 of" in user]
        assert len(reduce_calls) > 1
        for system, user in summarizer.completions:
            used = summarizer.count_tokens(system) + summarizer.count_tokens(user)
            assert used + summarizer.completion_reserve() <= 1500

    def test_absurdly_small_window_clamps_instead_of_shredding(self):
        # A window too small even for the prompt scaffolding cannot be honoured;
        # chunking falls back to a floor rather than splitting word by word.
        summarizer = _RecordingSummarizer(SummarizationConfig(context_size=400))

        summarizer.summarize(_transcript(3000))

        mapped = [user for _system, user in summarizer.completions[:-1]]
        assert mapped
        # Chunk count tracks the floor, not the word count: a 3000-word
        # transcript becomes a dozen or so calls, not hundreds.
        assert len(mapped) < 2 * (3000 // MIN_CHUNK_TOKENS)


class TestReduceIsTemplateAgnostic:
    def test_reduce_keeps_the_custom_system_prompt(self):
        templates = {
            "vibes": TemplateConfig(
                system_prompt="You are a vibes reporter.",
                prompt="Report on:\n{transcript}",
            )
        }
        config = SummarizationConfig(context_size=1000, template="vibes")
        summarizer = _RecordingSummarizer(config, templates)

        summarizer.summarize(_transcript(3000))

        reduce_system, _reduce_user = summarizer.completions[-1]
        assert reduce_system.startswith("You are a vibes reporter.")

    def test_reduce_does_not_hardcode_the_builtin_sections(self):
        templates = {
            "vibes": TemplateConfig(
                system_prompt="You are a vibes reporter.",
                prompt="Report on:\n{transcript}",
            )
        }
        config = SummarizationConfig(context_size=1000, template="vibes")
        summarizer = _RecordingSummarizer(
            config, templates, replies=["## Vibes\n- upbeat\n\n## Snacks\n- chips"],
        )

        summarizer.summarize(_transcript(3000))

        _reduce_system, reduce_user = summarizer.completions[-1]
        # It consolidates whatever sections the partials happen to use ...
        assert "## Vibes" in reduce_user
        assert "## Snacks" in reduce_user
        # ... and never names the built-in meeting template's sections.
        for builtin_section in ("Key Points", "Action Items", "Decisions"):
            assert builtin_section not in reduce_user
