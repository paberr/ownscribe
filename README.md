# notetaker

Fully local meeting transcription and summarization CLI for macOS. Record system audio, transcribe with WhisperX, and summarize with a local LLM. All processing stays on-device.

## Features

- **System audio capture** — records all system audio natively via Core Audio Taps (macOS 14.2+), no virtual audio drivers needed
- **WhisperX transcription** — fast, accurate speech-to-text with word-level timestamps
- **Speaker diarization** — optional speaker identification via pyannote (requires HuggingFace token)
- **Local LLM summarization** — structured meeting notes via Ollama, LM Studio, or any OpenAI-compatible server
- **One command** — just run `notetaker`, press Ctrl+C when done, get transcript + summary

## Requirements

- macOS 14.2+ (for system audio capture)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Xcode Command Line Tools (`xcode-select --install`)
- One of:
  - [Ollama](https://ollama.ai) — `brew install ollama`
  - [LM Studio](https://lmstudio.ai)
  - Any OpenAI-compatible local server

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd local-meeting-notes

# Build the Swift audio capture helper
bash swift/build.sh

# Install with transcription support
uv sync --extra transcription

# Pull a model for summarization (if using Ollama)
ollama pull mistral
```

## Usage

### Record, transcribe, and summarize a meeting

```bash
notetaker                    # records system audio, Ctrl+C to stop
```

This will:
1. Capture system audio until you press Ctrl+C
2. Transcribe with WhisperX
3. Summarize with your local LLM
4. Save everything to `~/notetaker/YYYY-MM-DD_HHMMSS/`

### Options

```bash
notetaker --device "MacBook Pro Microphone"   # use mic instead of system audio
notetaker --no-summarize                      # skip LLM summarization
notetaker --diarize                           # enable speaker identification
notetaker --model large-v3                    # use a larger Whisper model
notetaker --format json                       # output as JSON instead of markdown
```

### Subcommands

```bash
notetaker devices                  # list audio input devices
notetaker transcribe recording.wav # transcribe an existing audio file
notetaker summarize transcript.md  # summarize an existing transcript
notetaker config                   # open config file in $EDITOR
```

## Configuration

Config is stored at `~/.config/notetaker/config.toml`. Run `notetaker config` to create and edit it.

```toml
[audio]
backend = "coreaudio"     # "coreaudio" or "sounddevice"
device = ""               # empty = system audio

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
dir = "~/notetaker"
format = "markdown"       # "markdown" or "json"
```

**Precedence:** CLI flags > environment variables (`HF_TOKEN`, `OLLAMA_HOST`) > config file > defaults.

## Speaker Diarization

Speaker identification requires a HuggingFace token with access to [pyannote models](https://huggingface.co/pyannote/speaker-diarization-3.1):

1. Accept the model terms on HuggingFace
2. Create a token at https://huggingface.co/settings/tokens
3. Set `HF_TOKEN` env var or add `hf_token` to config
4. Run with `--diarize`

## License

MIT
