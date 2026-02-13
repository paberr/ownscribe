"""Tests for CLI command parsing."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from ownscribe.cli import cli
from ownscribe.config import Config


def _mock_config(config: Config | None = None):
    """Return a mock that makes Config.load() return a default Config."""
    return mock.patch("ownscribe.cli.Config.load", return_value=config or Config())


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

    def test_language_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--language", "de"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.transcription.language == "de"


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

    def test_cleanup_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cleanup", "--help"])
        assert result.exit_code == 0
        assert "Remove ownscribe data from disk" in result.output


class TestKeepRecordingFlag:
    def test_keep_recording_flag(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, ["--no-keep-recording"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.output.keep_recording is False

    def test_keep_recording_default_is_true(self):
        runner = CliRunner()
        with _mock_config(), mock.patch("ownscribe.pipeline.run_pipeline") as mock_run:
            result = runner.invoke(cli, [])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.output.keep_recording is True


class TestCleanup:
    def test_all_yes_removes_dirs(self, tmp_path):
        config_dir = tmp_path / "config"
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "output"
        for d in (config_dir, cache_dir, output_dir):
            d.mkdir()
            (d / "file.txt").write_text("data")

        cfg = Config()
        cfg.output.dir = str(output_dir)

        runner = CliRunner()
        with (
            _mock_config(cfg),
            mock.patch("ownscribe.cli._CONFIG_DIR", str(config_dir)),
            mock.patch("ownscribe.cli._CACHE_DIR", str(cache_dir)),
        ):
            result = runner.invoke(cli, ["cleanup", "--all", "--yes"])

        assert result.exit_code == 0
        assert not config_dir.exists()
        assert not cache_dir.exists()
        assert not output_dir.exists()
        assert "Removed Config" in result.output
        assert "Removed Cache" in result.output
        assert "Removed Output" in result.output

    def test_config_only(self, tmp_path):
        config_dir = tmp_path / "config"
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "output"
        config_dir.mkdir()
        cache_dir.mkdir()
        output_dir.mkdir()

        cfg = Config()
        cfg.output.dir = str(output_dir)

        runner = CliRunner()
        with (
            _mock_config(cfg),
            mock.patch("ownscribe.cli._CONFIG_DIR", str(config_dir)),
            mock.patch("ownscribe.cli._CACHE_DIR", str(cache_dir)),
        ):
            result = runner.invoke(cli, ["cleanup", "--config", "--yes"])

        assert result.exit_code == 0
        assert not config_dir.exists()
        assert cache_dir.exists()
        assert output_dir.exists()

    def test_skips_missing_dirs(self, tmp_path):
        cfg = Config()
        cfg.output.dir = str(tmp_path / "nonexistent")

        runner = CliRunner()
        with (
            _mock_config(cfg),
            mock.patch("ownscribe.cli._CONFIG_DIR", str(tmp_path / "no-config")),
            mock.patch("ownscribe.cli._CACHE_DIR", str(tmp_path / "no-cache")),
        ):
            result = runner.invoke(cli, ["cleanup", "--all", "--yes"])

        assert result.exit_code == 0
        assert "not found, skipping" in result.output
