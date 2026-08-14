"""Tests for the Core Audio helper command line."""

from __future__ import annotations

from pathlib import Path
from unittest import mock


def _make_recorder(**kwargs):
    from ownscribe.audio.coreaudio import CoreAudioRecorder

    binary = Path("/usr/local/bin/ownscribe-audio")
    with mock.patch("ownscribe.audio.coreaudio._find_binary", return_value=binary):
        return CoreAudioRecorder(**kwargs)


def _capture_cmd(recorder, tmp_path: Path) -> list[str]:
    with mock.patch("ownscribe.audio.coreaudio.subprocess.Popen") as mock_popen:
        recorder.start(tmp_path / "recording.wav")
    return mock_popen.call_args[0][0]


class TestDownloadBinaryArchitecture:
    """Releases only publish ownscribe-audio-arm64; anything else must say so."""

    @staticmethod
    def _download(platform_name: str, machine: str, capsys):
        from ownscribe.audio import coreaudio

        with (
            mock.patch.object(coreaudio.sys, "platform", platform_name),
            mock.patch.object(coreaudio.platform, "machine", return_value=machine),
            mock.patch.object(coreaudio.urllib.request, "urlretrieve") as urlretrieve,
        ):
            result = coreaudio._download_binary()
        return result, urlretrieve, capsys.readouterr()

    def test_intel_mac_explains_instead_of_downloading(self, capsys):
        result, urlretrieve, captured = self._download("darwin", "x86_64", capsys)

        assert result is None
        # A download would 404; the failure used to be swallowed silently.
        urlretrieve.assert_not_called()
        assert "Apple Silicon" in captured.err
        assert "x86_64" in captured.err

    def test_non_macos_stays_quiet(self, capsys):
        result, urlretrieve, captured = self._download("linux", "x86_64", capsys)

        assert result is None
        urlretrieve.assert_not_called()
        assert captured.err == ""

    def test_apple_silicon_still_downloads(self, tmp_path):
        from ownscribe.audio import coreaudio

        with (
            mock.patch.object(coreaudio.sys, "platform", "darwin"),
            mock.patch.object(coreaudio.platform, "machine", return_value="arm64"),
            mock.patch.object(coreaudio, "_CACHE_DIR", tmp_path),
            mock.patch.object(coreaudio.urllib.request, "urlretrieve") as urlretrieve,
            mock.patch.object(Path, "chmod"),
        ):
            result = coreaudio._download_binary()

        assert result == tmp_path / "ownscribe-audio"
        assert "ownscribe-audio-arm64" in urlretrieve.call_args[0][0]


class TestCoreAudioRecorderCommand:
    def test_mic_and_device_passed_when_mic_enabled(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(mic=True, mic_device="USB Mic"), tmp_path)

        assert "--mic" in cmd
        assert cmd[cmd.index("--mic-device") + 1] == "USB Mic"

    def test_mic_device_ignored_when_mic_disabled(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(mic=False, mic_device="USB Mic"), tmp_path)

        assert "--mic" not in cmd
        assert "--mic-device" not in cmd

    def test_capture_mode_all(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(capture_mode="all"), tmp_path)

        assert "--capture-mode-all" in cmd

    def test_capture_mode_picker(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(capture_mode="picker"), tmp_path)

        assert "--capture-mode-all" not in cmd

    def test_silence_timeout_passed(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(silence_timeout=120), tmp_path)

        assert cmd[cmd.index("--silence-timeout") + 1] == "120"

    def test_silence_timeout_omitted_when_disabled(self, tmp_path):
        cmd = _capture_cmd(_make_recorder(silence_timeout=0), tmp_path)

        assert "--silence-timeout" not in cmd
