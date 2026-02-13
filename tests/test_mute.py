"""Tests for mic mute toggle functionality."""

from __future__ import annotations

import signal
from unittest import mock

from ownscribe.audio.base import AudioRecorder


class TestBaseClassMuteDefaults:
    """Base class provides no-op mute methods."""

    def test_toggle_mute_is_noop(self):
        # Create a concrete subclass to test the base defaults
        class DummyRecorder(AudioRecorder):
            def start(self, output_path):
                pass

            def stop(self):
                pass

            def is_available(self):
                return True

        recorder = DummyRecorder()
        recorder.toggle_mute()  # should not raise

    def test_is_muted_returns_false(self):
        class DummyRecorder(AudioRecorder):
            def start(self, output_path):
                pass

            def stop(self):
                pass

            def is_available(self):
                return True

        recorder = DummyRecorder()
        assert recorder.is_muted is False


class TestCoreAudioRecorderMute:
    """CoreAudioRecorder sends SIGUSR1 and tracks mute state."""

    def _make_recorder(self, mic: bool = True) -> object:
        from ownscribe.audio.coreaudio import CoreAudioRecorder

        with mock.patch("ownscribe.audio.coreaudio._find_binary", return_value=None):
            return CoreAudioRecorder(mic=mic)

    def test_toggle_mute_sends_sigusr1(self):
        recorder = self._make_recorder(mic=True)
        proc = mock.MagicMock()
        proc.poll.return_value = None  # process running
        recorder._process = proc

        recorder.toggle_mute()

        proc.send_signal.assert_called_once_with(signal.SIGUSR1)
        assert recorder.is_muted is True

    def test_toggle_mute_twice_unmutes(self):
        recorder = self._make_recorder(mic=True)
        proc = mock.MagicMock()
        proc.poll.return_value = None
        recorder._process = proc

        recorder.toggle_mute()
        assert recorder.is_muted is True

        recorder.toggle_mute()
        assert recorder.is_muted is False
        assert proc.send_signal.call_count == 2

    def test_toggle_mute_noop_without_mic(self):
        recorder = self._make_recorder(mic=False)
        proc = mock.MagicMock()
        proc.poll.return_value = None
        recorder._process = proc

        recorder.toggle_mute()

        proc.send_signal.assert_not_called()
        assert recorder.is_muted is False

    def test_toggle_mute_noop_without_process(self):
        recorder = self._make_recorder(mic=True)
        # _process is None (not started)

        recorder.toggle_mute()  # should not raise
        assert recorder.is_muted is False

    def test_toggle_mute_noop_when_process_exited(self):
        recorder = self._make_recorder(mic=True)
        proc = mock.MagicMock()
        proc.poll.return_value = 0  # process exited
        recorder._process = proc

        recorder.toggle_mute()

        proc.send_signal.assert_not_called()
        assert recorder.is_muted is False

    def test_is_muted_default_false(self):
        recorder = self._make_recorder(mic=True)
        assert recorder.is_muted is False
