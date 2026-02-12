"""Shared fixtures for notetaker tests."""

from __future__ import annotations

import pytest

from notetaker.transcription.models import Segment, TranscriptResult


@pytest.fixture
def sample_transcript() -> TranscriptResult:
    """A basic transcript with two segments, no speakers."""
    return TranscriptResult(
        segments=[
            Segment(text="Hello world.", start=0.0, end=1.5),
            Segment(text="How are you?", start=1.5, end=3.0),
        ],
        language="en",
        duration=3.0,
    )


@pytest.fixture
def diarized_transcript() -> TranscriptResult:
    """A transcript with speaker labels."""
    return TranscriptResult(
        segments=[
            Segment(text="Hi, let's get started.", start=0.0, end=2.0, speaker="SPEAKER_00"),
            Segment(text="Sounds good.", start=2.0, end=3.5, speaker="SPEAKER_01"),
            Segment(text="First topic is the budget.", start=3.5, end=6.0, speaker="SPEAKER_00"),
        ],
        language="en",
        duration=6.0,
    )


@pytest.fixture
def tmp_config_file(tmp_path):
    """Write a minimal TOML config to a temp directory and return its path."""
    config_toml = tmp_path / "config.toml"
    config_toml.write_text(
        '[transcription]\nmodel = "large-v3"\n\n[output]\ndir = "/tmp/test-notes"\n'
    )
    return config_toml
