# ownscribe

[![PyPI](https://img.shields.io/pypi/v/ownscribe)](https://pypi.org/project/ownscribe/)
[![CI](https://github.com/paberr/ownscribe/actions/workflows/ci.yml/badge.svg)](https://github.com/paberr/ownscribe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Local-first meeting transcription and summarization CLI.
Record, transcribe, and summarize meetings and system audio entirely on your machine – no cloud, no bots, no data leaving your device.

> System audio capture requires **macOS 14.2 or later**. Other platforms can use the sounddevice backend with an external audio source.

## Table of Contents

- [Privacy](#privacy)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Summarization Templates](#summarization-templates)
- [Speaker Diarization](#speaker-diarization)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Privacy

ownscribe **does not**:

- send audio to external servers
- upload transcripts
- require cloud APIs
- store data outside your machine

All audio, transcripts, and summaries remain local.

<p align="center">
  <img src="docs/demo-pipeline.gif" alt="ownscribe demo" width="750">
</p>

## Features

- **System audio capture** — records all system audio natively via Core Audio Taps (macOS 14.2+), no virtual audio drivers needed
- **Microphone capture** — optionally record system + mic audio simultaneously with `--mic`
- **WhisperX transcription** — fast, accurate speech-to-text with word-level timestamps
- **Speaker diarization** — optional speaker identification via pyannote (requires HuggingFace token)
- **Pipeline progress** — live checklist showing transcription, diarization sub-steps, and summarization progress
- **Local LLM summarization** — structured meeting notes with a built-in model (Phi-4-mini); also supports Ollama, LM Studio, or any OpenAI-compatible server
- **Summarization templates** — built-in presets for meetings, lectures, and quick briefs; define your own in config
- **Ask your meetings** — ask natural-language questions across all your meeting notes; uses a two-stage LLM pipeline with keyword fallback
  <br><img src="docs/demo-ask.gif" alt="ownscribe ask demo" width="700">
- **Silence auto-stop** — automatically stops recording after sustained silence (default: 5 minutes, configurable)
- **One command** — just run `ownscribe`, press Ctrl+C when done, get transcript + summary

## Requirements

- macOS 14.2+ (for system audio capture)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`
- Xcode Command Line Tools (`xcode-select --install`)

Summarization works out of the box — a local model (Phi-4-mini, ~2.4 GB) downloads automatically on first run. Optionally, you can use [Ollama](https://ollama.ai), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible server instead (see [Configuration](#configuration)).

Works with any app that outputs audio through Core Audio (Zoom, Teams, Meet, etc.).

> **Tip:** Your terminal app (Terminal, iTerm2, VS Code, etc.) needs **Screen Recording** permission to capture system audio.
> Open the settings panel directly with:
> ```bash
> open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
> ```
> Enable your terminal app, then restart it.

## Installation

### Quick start with uvx

```bash
uvx ownscribe
```

On macOS, the Swift audio capture helper is downloaded automatically on first run.

### Alternative summarization backends

The built-in local model works out of the box. If you'd rather call a hosted backend, install the matching extra:

```bash
uv add 'ownscribe[ollama]'   # use Ollama
uv add 'ownscribe[openai]'   # use any OpenAI-compatible server (LM Studio, llama-server, etc.)
uv add 'ownscribe[all]'      # install both
```

### From source

```bash
# Clone the repo
git clone https://github.com/paberr/ownscribe.git
cd ownscribe

# Build the Swift audio capture helper (optional - auto-downloads if skipped)
bash swift/build.sh

# Install with all backends
uv sync --extra all
```

## Usage

### Record, transcribe, and summarize a meeting

```bash
ownscribe                    # records system audio, Ctrl+C to stop
```

This will:
1. Capture system audio until you press Ctrl+C (or auto-stop after 5 minutes of silence)
2. Transcribe with WhisperX
3. Summarize with your local LLM
4. Save everything to `~/ownscribe/YYYY-MM-DD_HHMMSS/`

On first run, WhisperX / pyannote and the summarization model may download model files. ownscribe shows a `Preparing models` step and best-effort download progress in the TUI while this happens. Use `ownscribe warmup` to pre-download all models.

### Options

```bash
ownscribe --mic                               # capture system audio + default mic (press 'm' to mute/unmute)
ownscribe --mic-device "MacBook Pro Microphone" # capture system audio + specific mic
ownscribe --device "MacBook Pro Microphone"   # use mic instead of system audio
ownscribe --no-summarize                      # skip LLM summarization
ownscribe --diarize                           # enable speaker identification
ownscribe --language en                       # set transcription language (default: auto-detect)
ownscribe --model large-v3                    # use a larger Whisper model
ownscribe --format json                       # output as JSON instead of markdown
ownscribe --no-keep-recording                 # auto-delete WAV files after transcription
ownscribe --template lecture                  # use the lecture summarization template
ownscribe --silence-timeout 600               # auto-stop after 10 minutes of silence
ownscribe --silence-timeout 0                 # disable silence auto-stop
```

### Subcommands

```bash
ownscribe devices                  # list audio devices (uses native CoreAudio when available)
ownscribe apps                     # list running apps with PIDs for use with --pid
ownscribe warmup                   # prefetch WhisperX/pyannote models before a meeting
ownscribe transcribe recording.wav # transcribe an audio file (saves alongside the input)
ownscribe summarize transcript.md  # summarize a transcript (saves alongside the input)
ownscribe resume ./2026-02-20_1736 # resume a failed/partial pipeline in a directory
ownscribe ask "question"           # search your meetings with a natural-language question
ownscribe config                   # open config file in $EDITOR
ownscribe cleanup                  # remove ownscribe data from disk
```

Use `warmup` ahead of time to avoid first-run model download delays while recording:

```bash
ownscribe warmup                    # prefetch Whisper model (+ diarization if enabled in config)
ownscribe warmup --language en      # also prefetch alignment model for English
ownscribe warmup --with-diarization # force diarization warmup for this run
```

### Searching Meeting Notes

Use `ask` to search across all your meeting notes with natural-language questions:

```bash
ownscribe ask "What did Anna say about the deadline?"
ownscribe ask "budget decisions" --since 2026-01-01
ownscribe ask "action items from last week" --limit 5
```

This runs a two-stage pipeline:
1. **Find** — sends meeting summaries to the LLM to identify which meetings are relevant
2. **Answer** — sends the full transcripts of relevant meetings to the LLM to produce an answer with quotes

If the LLM finds no relevant meetings, a keyword fallback searches summaries and transcripts directly.

## Configuration

Config is stored at `~/.config/ownscribe/config.toml`. Run `ownscribe config` to create and edit it.

```toml
[audio]
backend = "coreaudio"     # "coreaudio" or "sounddevice"
device = ""               # empty = system audio
mic = false               # also capture microphone input
mic_device = ""           # specific mic device name (empty = default)
silence_timeout = 300     # seconds of silence before auto-stop; 0 = disabled

[transcription]
model = "base"            # tiny, base, small, medium, large-v3
language = ""             # empty = auto-detect

[diarization]
enabled = false
hf_token = ""             # HuggingFace token for pyannote
telemetry = false         # allow HuggingFace Hub + pyannote metrics telemetry
device = "auto"           # "auto" (mps if available), "mps", or "cpu"

[summarization]
enabled = true
backend = "local"         # "local" (built-in, no server needed), "ollama", or "openai"
model = "phi-4-mini"      # local: "phi-4-mini", path to GGUF, or hf:owner/repo/file.gguf; ollama/openai: model name
# host = "http://localhost:11434"  # only for ollama/openai backends
# template = "meeting"    # "meeting", "lecture", "brief", or a custom name
# context_size = 0        # 0 = auto-detect from model; set manually for OpenAI-compatible backends

# Custom templates (optional):
# [templates.my-standup]
# system_prompt = "You summarize daily standups."
# prompt = "List each person's update:\n{transcript}"

[output]
dir = "~/ownscribe"
format = "markdown"       # "markdown" or "json"
keep_recording = true     # false = auto-delete WAV after transcription
```

**Precedence:** CLI flags > environment variables (`HF_TOKEN`, `OLLAMA_HOST`) > config file > defaults.

## Summarization Templates

Built-in templates control how transcripts are summarized:

| Template | Best for | Output style |
|----------|----------|-------------|
| `meeting` | Meetings, standups, 1:1s | Summary, Key Points, Action Items, Decisions |
| `lecture` | Lectures, seminars, talks | Summary, Key Concepts, Key Takeaways |
| `brief` | Quick overviews | 3-5 bullet points |

Use `--template` on the CLI or set `template` in `[summarization]` config. Default is `meeting`.

Define custom templates in config:

```toml
[templates.my-standup]
system_prompt = "You summarize daily standups."
prompt = "List each person's update:\n{transcript}"
```

Then use with `--template my-standup` or `template = "my-standup"` in config.

## Speaker Diarization

Speaker identification requires a HuggingFace token with access to the pyannote models:

1. Accept the terms for both models on HuggingFace:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Create a token at https://huggingface.co/settings/tokens
3. Set `HF_TOKEN` env var or add `hf_token` to config
4. Run with `--diarize`

On Apple Silicon Macs, diarization automatically uses the Metal Performance Shaders (MPS) GPU backend for ~10x faster processing. Set `device = "cpu"` in the `[diarization]` config section to disable this.

## Acknowledgments

ownscribe builds on some excellent open-source projects:

- [WhisperX](https://github.com/m-bain/whisperX) — fast speech recognition with word-level timestamps and speaker diarization
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [llama.cpp](https://github.com/ggerganov/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — local LLM inference
- [Ollama](https://ollama.ai) — local LLM serving
- [Click](https://click.palletsprojects.com) — CLI framework

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and open contribution areas.

## License

MIT
