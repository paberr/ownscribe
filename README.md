# ownscribe

Local-first meeting transcription and summarization CLI.
Record, transcribe, and summarize meetings and system audio entirely on your machine – no cloud, no bots, no data leaving your device.

## Features

- **System audio capture** — records all system audio natively via Core Audio Taps (macOS 14.2+), no virtual audio drivers needed
- **Microphone capture** — optionally record system + mic audio simultaneously with `--mic`
- **WhisperX transcription** — fast, accurate speech-to-text with word-level timestamps
- **Speaker diarization** — optional speaker identification via pyannote (requires HuggingFace token)
- **Local LLM summarization** — structured meeting notes via Ollama, LM Studio, or any OpenAI-compatible server
- **One command** — just run `ownscribe`, press Ctrl+C when done, get transcript + summary

## Requirements

- macOS 14.2+ (for system audio capture)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Xcode Command Line Tools (`xcode-select --install`)
- One of:
  - [Ollama](https://ollama.ai) — `brew install ollama`
  - [LM Studio](https://lmstudio.ai)
  - Any OpenAI-compatible local server

Works with any app that outputs audio through Core Audio (Zoom, Teams, Meet, etc.).

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
ownscribe --mic                               # capture system audio + default mic
ownscribe --mic-device "MacBook Pro Microphone" # capture system audio + specific mic
ownscribe --device "MacBook Pro Microphone"   # use mic instead of system audio
ownscribe --no-summarize                      # skip LLM summarization
ownscribe --diarize                           # enable speaker identification
ownscribe --model large-v3                    # use a larger Whisper model
ownscribe --format json                       # output as JSON instead of markdown
```

### Subcommands

```bash
ownscribe devices                  # list audio devices (uses native CoreAudio when available)
ownscribe apps                     # list running apps with PIDs for use with --pid
ownscribe transcribe recording.wav # transcribe an existing audio file
ownscribe summarize transcript.md  # summarize an existing transcript
ownscribe config                   # open config file in $EDITOR
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

[summarization]
enabled = true
backend = "ollama"        # "ollama" or "openai"
model = "mistral"
host = "http://localhost:11434"

[output]
dir = "~/ownscribe"
format = "markdown"       # "markdown" or "json"
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

## Acknowledgments

ownscribe builds on some excellent open-source projects:

- [WhisperX](https://github.com/m-bain/whisperX) — fast speech recognition with word-level timestamps and speaker diarization
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [Ollama](https://ollama.ai) — local LLM serving
- [Click](https://click.palletsprojects.com) — CLI framework

## License

MIT
