"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from notetaker.config import Config, _merge_toml, ensure_config_file, OutputConfig


class TestDefaults:
    def test_default_audio_backend(self):
        cfg = Config()
        assert cfg.audio.backend == "coreaudio"

    def test_default_transcription_model(self):
        cfg = Config()
        assert cfg.transcription.model == "base"

    def test_default_summarization_enabled(self):
        cfg = Config()
        assert cfg.summarization.enabled is True

    def test_default_output_format(self):
        cfg = Config()
        assert cfg.output.format == "markdown"

    def test_default_mic_settings(self):
        cfg = Config()
        assert cfg.audio.mic is False
        assert cfg.audio.mic_device == ""


class TestMergeToml:
    def test_full_override(self):
        cfg = Config()
        data = {
            "audio": {"backend": "sounddevice", "device": "USB Mic"},
            "transcription": {"model": "large-v3", "language": "de"},
        }
        merged = _merge_toml(cfg, data)
        assert merged.audio.backend == "sounddevice"
        assert merged.audio.device == "USB Mic"
        assert merged.transcription.model == "large-v3"
        assert merged.transcription.language == "de"

    def test_partial_toml_keeps_defaults(self):
        cfg = Config()
        data = {"transcription": {"model": "small"}}
        merged = _merge_toml(cfg, data)
        assert merged.transcription.model == "small"
        # Other sections unchanged
        assert merged.audio.backend == "coreaudio"
        assert merged.summarization.backend == "ollama"

    def test_mic_settings_from_toml(self):
        cfg = Config()
        data = {"audio": {"mic": True, "mic_device": "MacBook Pro Microphone"}}
        merged = _merge_toml(cfg, data)
        assert merged.audio.mic is True
        assert merged.audio.mic_device == "MacBook Pro Microphone"

    def test_unknown_keys_ignored(self):
        cfg = Config()
        data = {"audio": {"nonexistent_key": 42}}
        merged = _merge_toml(cfg, data)
        assert not hasattr(merged.audio, "nonexistent_key")


class TestEnvOverrides:
    def test_hf_token_from_env(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}):
            with mock.patch("notetaker.config.CONFIG_PATH") as mock_path:
                mock_path.exists.return_value = False
                cfg = Config.load()
        assert cfg.diarization.hf_token == "hf_test123"

    def test_ollama_host_from_env(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "http://remote:11434"}):
            with mock.patch("notetaker.config.CONFIG_PATH") as mock_path:
                mock_path.exists.return_value = False
                cfg = Config.load()
        assert cfg.summarization.host == "http://remote:11434"


class TestEnsureConfigFile:
    def test_creates_file_when_missing(self, tmp_path):
        config_dir = tmp_path / "notetaker"
        config_path = config_dir / "config.toml"
        with mock.patch("notetaker.config.CONFIG_DIR", config_dir), \
             mock.patch("notetaker.config.CONFIG_PATH", config_path):
            result = ensure_config_file()
        assert result.exists()
        assert "[audio]" in result.read_text()

    def test_does_not_overwrite_existing(self, tmp_path):
        config_dir = tmp_path / "notetaker"
        config_dir.mkdir()
        config_path = config_dir / "config.toml"
        config_path.write_text("# custom config\n")
        with mock.patch("notetaker.config.CONFIG_DIR", config_dir), \
             mock.patch("notetaker.config.CONFIG_PATH", config_path):
            ensure_config_file()
        assert config_path.read_text() == "# custom config\n"


class TestResolvedDir:
    def test_expands_tilde(self):
        cfg = OutputConfig(dir="~/notetaker")
        resolved = cfg.resolved_dir
        assert "~" not in str(resolved)
        assert str(resolved).endswith("notetaker")

    def test_absolute_path_unchanged(self):
        cfg = OutputConfig(dir="/tmp/notes")
        assert cfg.resolved_dir == Path("/tmp/notes")
