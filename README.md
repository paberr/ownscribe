# ownscribe

[![PyPI](https://img.shields.io/pypi/v/ownscribe)](https://pypi.org/project/ownscribe/)
[![CI](https://github.com/paberr/ownscribe/actions/workflows/ci.yml/badge.svg)](https://github.com/paberr/ownscribe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Local-first meeting transcription and summarization CLI.
Record, transcribe, and summarize meetings and system audio entirely on your machine – no cloud, no bots, no data leaving your device.

> System audio capture requires **macOS 14.2 or later**. Other platforms can use the sounddevice backend with an external audio source.

## Privacy

ownscribe **does not**:

- send audio to external servers
- upload transcripts
- require cloud APIs
- store data outside your machine

All audio, transcripts, and summaries remain local.

<!-- TODO: Add asciinema demo or terminal screenshot here -->

## Features

- **System audio capture** — records all system audio natively via Core Audio Taps (macOS 14.2+), no virtual audio drivers needed
- **Microphone capture** — optionally record system + mic audio simultaneously with `--mic`
- **WhisperX transcription** — fast, accurate speech-to-text with word-level timestamps
- **Speaker diarization** — optional speaker identification via pyannote (requires HuggingFace token)
- **Pipeline progress** — live checklist showing transcription, diarization sub-steps, and summarization progress
- **Local LLM summarization** — structured meeting notes via Ollama, LM Studio, or any OpenAI-compatible server
- **Custom prompts** — override the built-in summarization system and user prompts in config
- **One command** — just run `ownscribe`, press Ctrl+C when done, get transcript + summary

## Requirements

- macOS 14.2+ (for system audio capture)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`
- Xcode Command Line Tools (`xcode-select --install`)
- One of:
  - [Ollama](https://ollama.ai) — `brew install ollama`
  - [LM Studio](https://lmstudio.ai)
  - Any OpenAI-compatible local server

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

### From source

```bash
# Clone the repo
git clone https://github.com/paberr/ownscribe.git
cd ownscribe

# Build the Swift audio capture helper (optional — auto-downloads if skipped)
bash swift/build.sh

# Install with transcription support
uv sync --extra transcription

# Pull a model for summarization (if using Ollama)
ollama pull mistral
```

## Usage

### Record, transcribe, and summarize a meeting

```bash
ownscribe                    # records system audio, Ctrl+C to stop
```

This will:
1. Capture system audio until you press Ctrl+C
2. Transcribe with WhisperX
3. Summarize with your local LLM
4. Save everything to `~/ownscribe/YYYY-MM-DD_HHMMSS/`

### Options

```bash
ownscribe --mic                               # capture system audio + default mic (press 'm' to mute/unmute)
ownscribe --mic-device "MacBook Pro Microphone" # capture system audio + specific mic
ownscribe --device "MacBook Pro Microphone"   # use mic instead of system audio
ownscribe --no-summarize                      # skip LLM summarization
ownscribe --diarize                           # enable speaker identification
ownscribe --language en                        # set transcription language (default: auto-detect)
ownscribe --model large-v3                    # use a larger Whisper model
ownscribe --format json                       # output as JSON instead of markdown
ownscribe --no-keep-recording                 # auto-delete WAV files after transcription
```

### Subcommands

```bash
ownscribe devices                  # list audio devices (uses native CoreAudio when available)
ownscribe apps                     # list running apps with PIDs for use with --pid
ownscribe transcribe recording.wav # transcribe an existing audio file
ownscribe summarize transcript.md  # summarize an existing transcript
ownscribe config                   # open config file in $EDITOR
ownscribe cleanup                  # remove ownscribe data from disk
```

## Configuration

Config is stored at `~/.config/ownscribe/config.toml`. Run `ownscribe config` to create and edit it.

```toml
[audio]
backend = "coreaudio"     # "coreaudio" or "sounddevice"
device = ""               # empty = system audio
mic = false               # also capture microphone input
mic_device = ""           # specific mic device name (empty = default)

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
backend = "ollama"        # "ollama" or "openai"
model = "mistral"
host = "http://localhost:11434"
system_prompt = ""        # custom system prompt (empty = built-in default)
prompt = ""               # custom user prompt; must contain {transcript}

[output]
dir = "~/ownscribe"
format = "markdown"       # "markdown" or "json"
keep_recording = true     # false = auto-delete WAV after transcription
```

**Precedence:** CLI flags > environment variables (`HF_TOKEN`, `OLLAMA_HOST`) > config file > defaults.

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
- [Ollama](https://ollama.ai) — local LLM serving
- [Click](https://click.palletsprojects.com) — CLI framework

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and open contribution areas.

## License

MIT
