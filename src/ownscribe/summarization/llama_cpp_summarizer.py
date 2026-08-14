"""Local summarization via llama-cpp-python."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

from ownscribe.progress import DownloadProgressEvent, DownloadProgressWriter
from ownscribe.summarization.base import Summarizer
from ownscribe.summarization.prompts import clean_response

if TYPE_CHECKING:
    from llama_cpp import Llama

    from ownscribe.config import SummarizationConfig, TemplateConfig

logger = logging.getLogger(__name__)

# Short name → (HuggingFace repo, GGUF filename)
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "phi-4-mini": (
        "unsloth/Phi-4-mini-instruct-GGUF",
        "Phi-4-mini-instruct-Q4_K_M.gguf",
    ),
}


def _ensure_model(
    model_name: str,
    on_progress: Callable[[DownloadProgressEvent], None] | None = None,
) -> Path:
    """Download the GGUF model if not already cached. Returns the local path."""
    from huggingface_hub import hf_hub_download

    if model_name.startswith("hf:"):
        # Format: "hf:owner/repo/filename.gguf"
        parts = model_name[3:]  # strip "hf:"
        slash_idx = parts.rfind("/")
        if slash_idx == -1:
            raise ValueError(
                f"Invalid HuggingFace model spec '{model_name}'. Expected format: hf:owner/repo/filename.gguf"
            )
        repo_id = parts[:slash_idx]
        filename = parts[slash_idx + 1 :]
    elif model_name in _MODEL_REGISTRY:
        repo_id, filename = _MODEL_REGISTRY[model_name]
    else:
        # Treat as a direct path to a GGUF file
        path = Path(model_name).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(_MODEL_REGISTRY)}, "
            "a path to a GGUF file, or hf:owner/repo/filename.gguf"
        )

    try:
        if on_progress is not None:
            writer = DownloadProgressWriter(on_progress)
            with redirect_stdout(writer), redirect_stderr(writer):
                path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            writer.flush()
            return path
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model '{model_name}' from {repo_id}: {exc}\n"
            "Check your internet connection and try again."
        ) from exc


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect fd 2 to /dev/null to silence C-level stderr (e.g. ggml_metal_init).

    Falls back to a no-op if fd manipulation is unavailable.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_fd = os.dup(2)
    except OSError:
        yield
        return
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(devnull_fd)
        os.close(old_fd)


class LlamaCppSummarizer(Summarizer):
    """Summarizes transcripts using a local GGUF model via llama-cpp-python."""

    def __init__(
        self,
        config: SummarizationConfig,
        templates: dict[str, TemplateConfig] | None = None,
    ) -> None:
        super().__init__(config, templates)
        self._llm: Llama | None = None

    def close(self) -> None:
        """Free the loaded model deterministically. Idempotent."""
        llm = self._llm
        self._llm = None
        if llm is not None:
            # Plain try/except, not contextlib.suppress: this runs from __del__ at
            # interpreter shutdown when the `contextlib` global is already None.
            try:  # noqa: SIM105
                llm.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:  # noqa: SIM105
            self.close()
        except Exception:
            pass

    def _get_llm(self) -> Llama:
        """Lazy-load the model on first use."""
        if self._llm is not None:
            return self._llm

        from llama_cpp import Llama

        model_path = _ensure_model(self._config.model)
        logger.info("Loading model from %s", model_path)
        with _suppress_stderr():
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.context_size,
                n_gpu_layers=-1,  # auto: offloads to Metal/CUDA when available, falls back to CPU
                verbose=False,
            )
        return self._llm

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's own tokenizer, not a character estimate."""
        if not text:
            return 0
        try:
            llm = self._get_llm()
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        except Exception:
            logger.debug("Falling back to estimated token count", exc_info=True)
            return super().count_tokens(text)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        llm = self._get_llm()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if json_mode:
            if json_schema is not None:
                formats_to_try: list[dict | None] = [
                    {"type": "json_object", "schema": json_schema},
                    {"type": "json_object"},
                    None,
                ]
            else:
                formats_to_try = [{"type": "json_object"}, None]
        else:
            formats_to_try = [None]

        for fmt in formats_to_try:
            try:
                kwargs: dict = {}
                if fmt is not None:
                    kwargs["response_format"] = fmt
                response = llm.create_chat_completion(messages=messages, **kwargs)
                return clean_response(response["choices"][0]["message"]["content"] or "")
            except Exception:
                continue
        return ""

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Prompt and completion share n_ctx: cap the answer so the model
            # cannot be cut off mid-sentence by its own prompt.
            max_tokens=self.completion_reserve(),
        )
        return clean_response(response["choices"][0]["message"]["content"] or "")

    def generate_title(self, summary_text: str) -> str:
        from ownscribe.summarization.prompts import TITLE_PROMPT, TITLE_SYSTEM

        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": TITLE_SYSTEM},
                {"role": "user", "content": TITLE_PROMPT.format(summary=summary_text)},
            ],
        )
        return clean_response(response["choices"][0]["message"]["content"] or "").strip()
