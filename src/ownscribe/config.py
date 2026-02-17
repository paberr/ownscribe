"""Configuration management with TOML loading and defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path("~/.config/ownscribe").expanduser()
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG_TOML = """\
[audio]
backend = "coreaudio"     # "coreaudio" (default) or "sounddevice"
device = ""               # empty = system audio; or device name/index for sounddevice
mic = false               # also capture microphone input
mic_device = ""           # specific mic device name (empty = default)

[transcription]
model = "base"            # whisper model: tiny, base, small, medium, large-v3
language = ""             # empty = auto-detect

[diarization]
enabled = false           # set to true + provide hf_token to enable
hf_token = ""             # HuggingFace token for pyannote models
min_speakers = 0          # 0 = auto-detect
max_speakers = 0
telemetry = false         # set to true to allow HuggingFace Hub + pyannote metrics telemetry
device = "auto"           # "auto" (mps if available), "mps", or "cpu"

[summarization]
enabled = true
backend = "ollama"        # "ollama" or "openai" (for LM Studio, llama.cpp server, etc.)
model = "mistral"         # model name
host = "http://localhost:11434"  # ollama: :11434, LM Studio: :1234
# template = "meeting"    # built-in: "meeting", "lecture", or "brief"

# Custom templates (optional):
# [templates.my-notes]
# system_prompt = "You are a helpful assistant."
# prompt = "Summarize:\\n{transcript}"

[output]
dir = "~/ownscribe"       # base output directory
format = "markdown"       # "markdown" or "json"
keep_recording = true     # keep WAV files after transcription; false = auto-delete
"""


@dataclass
class AudioConfig:
    backend: str = "coreaudio"
    device: str = ""
    mic: bool = False
    mic_device: str = ""


@dataclass
class TranscriptionConfig:
    model: str = "base"
    language: str = ""


@dataclass
class DiarizationConfig:
    enabled: bool = False
    hf_token: str = ""
    min_speakers: int = 0
    max_speakers: int = 0
    telemetry: bool = False
    device: str = "auto"


@dataclass
class SummarizationConfig:
    enabled: bool = True
    backend: str = "ollama"
    model: str = "mistral"
    host: str = "http://localhost:11434"
    template: str = ""


@dataclass
class TemplateConfig:
    system_prompt: str = ""
    prompt: str = ""


@dataclass
class OutputConfig:
    dir: str = "~/ownscribe"
    format: str = "markdown"
    keep_recording: bool = True

    @property
    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser()


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    templates: dict[str, TemplateConfig] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Config:
        """Load config from TOML file, falling back to defaults."""
        config = cls()

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
            config = _merge_toml(config, data)

        # Env var overrides
        if hf_token := os.environ.get("HF_TOKEN"):
            config.diarization.hf_token = hf_token
        if ollama_host := os.environ.get("OLLAMA_HOST"):
            config.summarization.host = ollama_host

        return config


def _merge_toml(config: Config, data: dict) -> Config:
    """Merge TOML data into config dataclass."""
    if "audio" in data:
        for k, v in data["audio"].items():
            if hasattr(config.audio, k):
                setattr(config.audio, k, v)

    if "transcription" in data:
        for k, v in data["transcription"].items():
            if hasattr(config.transcription, k):
                setattr(config.transcription, k, v)

    if "diarization" in data:
        for k, v in data["diarization"].items():
            if hasattr(config.diarization, k):
                setattr(config.diarization, k, v)

    if "summarization" in data:
        for k, v in data["summarization"].items():
            if hasattr(config.summarization, k):
                setattr(config.summarization, k, v)

    if "output" in data:
        for k, v in data["output"].items():
            if hasattr(config.output, k):
                setattr(config.output, k, v)

    if "templates" in data:
        for name, t_data in data["templates"].items():
            config.templates[name] = TemplateConfig(
                system_prompt=t_data.get("system_prompt", ""),
                prompt=t_data.get("prompt", ""),
            )

    return config


def ensure_config_file() -> Path:
    """Create default config file if it doesn't exist. Returns the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return CONFIG_PATH
