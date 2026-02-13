# Contributing to ownscribe

Thanks for your interest in contributing! This project is fully local meeting transcription and summarization — all processing stays on-device.

## Development setup

```bash
# Clone the repo
git clone https://github.com/paberr/ownscribe.git
cd ownscribe

# Install dependencies (including dev tools)
uv sync --dev

# Build the Swift audio capture helper (macOS only, optional)
bash swift/build.sh

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/
```

## Running tests

```bash
uv run pytest -v                    # all tests
uv run pytest -v -m "not hardware"  # skip tests requiring audio hardware
uv run pytest -v -k test_cli        # run a specific test module
```

Tests that require macOS are auto-skipped on other platforms. Tests marked `@pytest.mark.hardware` are auto-skipped in CI.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run `uv run ruff check src/ tests/` before submitting a PR. The CI will also check this.

## Open contribution areas

Here are some areas where contributions would be especially welcome:

### Linux/Windows audio capture backends
Currently, system audio capture only works on macOS via Core Audio Taps. Adding support for PulseAudio/PipeWire (Linux) or WASAPI (Windows) would make the tool cross-platform. See `src/ownscribe/audio/base.py` for the recorder interface. A two-device mode for SoundDeviceRecorder (separate system audio + mic streams) would also bring mic mute toggle support to non-macOS platforms.

### GUI frontend
Build a desktop GUI (Electron, Tauri, or native) that wraps the CLI. Could show a recording indicator, live transcript preview, and summary output.

### Integration with meeting tools
A big difference to cloud-based apps is that those usually connect to meetings directly and capture the audio through that. This way, if your microphone is muted, it's not recorded. With the mic mute toggle (press `m` during recording), you can now mute/unmute the mic stream without stopping the recording. Further integration ideas include detecting the meeting app's mute state automatically.

### Speaker name assignment
Currently, diarization produces anonymous labels (`SPEAKER_00`, `SPEAKER_01`). Allow users to map these to real names, either interactively after transcription or via a pre-configured mapping.

### Additional LLM backends
Add support for more summarization backends beyond Ollama and OpenAI-compatible servers — for example, Anthropic, Google, or local inference via llama.cpp directly. See `src/ownscribe/summarization/base.py` for the summarizer interface.

### Improved diarization
The current diarization works well for 2-3 speakers but can struggle with larger meetings. Investigate better models or post-processing to improve accuracy for many-participant calls.

## Submitting changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Ensure tests pass (`uv run pytest`) and linting is clean (`uv run ruff check src/ tests/`)
4. Submit a pull request with a clear description of the change
