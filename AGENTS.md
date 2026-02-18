# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                        # run all tests
uv run pytest -v                     # verbose
uv run pytest -v -k test_search      # run a specific test module
uv run pytest -v -k "TestRankMeetings::test_speaker_boost"  # single test
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # auto-format
```

## Architecture

**ownscribe** is a CLI tool for local meeting recording, transcription, and summarization. The main pipeline is: Record → Transcribe → Summarize → Output.

### Plugin systems with abstract base classes

Each stage has a base class in its subpackage and one or more implementations:

- **Audio** (`audio/base.py`): `CoreAudioRecorder` (macOS, wraps a Swift binary in `swift/`) and `SoundDeviceRecorder` (cross-platform fallback). Selected in `pipeline.py:_create_recorder()`.
- **Transcription** (`transcription/base.py`): `WhisperXTranscriber` (single impl). Data models (`Segment`, `Word`, `TranscriptResult`) live in `transcription/models.py`.
- **Summarization** (`summarization/base.py`): `OllamaSummarizer` and `OpenAISummarizer`. Factory is `summarization/__init__.py:create_summarizer()` — used by both `pipeline.py` and `search.py`.
- **Output** (`output/`): `markdown.py` and `json_output.py`, selected by `config.output.format`.

### Key modules

- **`cli.py`** — Click command group. Entry point: `ownscribe.cli:cli`. All subcommands (`ask`, `transcribe`, `summarize`, `devices`, `apps`, `config`, `cleanup`).
- **`pipeline.py`** — Orchestrates the record → transcribe → summarize flow. Creates timestamped output dirs (`~/ownscribe/YYYY-MM-DD_HHMM_slug/`).
- **`search.py`** — Two-stage LLM search over meeting notes. Stage 1 scores summaries for relevance, stage 2 synthesizes answers from full transcripts. Has keyword fallback and quote verification. Helper functions return data; only `ask()` calls `click.echo`.
- **`config.py`** — Dataclass hierarchy (`Config` → `AudioConfig`, `TranscriptionConfig`, `SummarizationConfig`, etc.). Loaded from `~/.config/ownscribe/config.toml` with env var overrides (`HF_TOKEN`, `OLLAMA_HOST`).
- **`summarization/prompts.py`** — Built-in prompt templates (meeting, lecture, brief) plus search prompts. Users can define custom templates in config TOML.

### Testing conventions

- Uses `pytest` with `pytest-httpserver` for mocking HTTP APIs (Ollama, OpenAI).
- Shared fixtures in `conftest.py`: `sample_transcript`, `diarized_transcript`, `synthetic_wav`.
- Tests use `FakeSummarizer` (in `test_search.py`) or `unittest.mock` for pipeline tests.
- Markers: `@pytest.mark.hardware` (auto-skipped in CI), `@pytest.mark.macos` (auto-skipped on non-macOS).
- When mocking the shared summarizer factory in pipeline tests, patch `ownscribe.pipeline.create_summarizer` (it's imported at module level).

## Style

- Python 3.12+. Ruff with line-length 120.
- `from __future__ import annotations` in all modules.
- Lazy imports for heavy dependencies (whisperx, ollama, openai) — imported inside functions, not at module level.
- Helper functions return data; orchestrator functions (`ask()`, `run_pipeline()`) handle all `click.echo` output.
