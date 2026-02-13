"""Tests for CLI command parsing."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from ownscribe.cli import cli
from ownscribe.config import Config


def _mock_config():
    """Return a mock that makes Config.load() return a default Config."""
    return mock.patch("ownscribe.cli.Config.load", return_value=Config())


class TestMainCommand:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fully local meeting transcription and summarization" in result.output

    def test_no_summarize_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--no-summarize"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.summarization.enabled is False

    def test_mic_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--mic"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.audio.mic is True

    def test_device_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--device", "USB Mic"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.audio.device == "USB Mic"
            assert config.audio.backend == "sounddevice"

    def test_model_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--model", "large-v3"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.transcription.model == "large-v3"


class TestSubcommandHelp:
    def test_transcribe_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcribe an audio file" in result.output

    def test_summarize_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--help"])
        assert result.exit_code == 0
        assert "Summarize a transcript file" in result.output

    def test_devices_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["devices", "--help"])
        assert result.exit_code == 0
        assert "List available audio input devices" in result.output

    def test_config_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "Open the configuration file" in result.output
