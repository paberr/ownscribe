from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("fastapi")
from fastapi import HTTPException

from webui import app as webapp


def test_markdown_transcript_skips_whisper_and_goes_to_the_summary_step(monkeypatch):
    seen = {}

    def fake_summarize(transcript, cancel, keys, detail):
        seen.update(transcript=transcript, keys=keys, detail=detail)
        return {"transcript": transcript, "summary": "## Key Points\n- x"}

    monkeypatch.setattr(webapp, "summarize", fake_summarize)
    monkeypatch.setattr(webapp, "transcribe", lambda *a: pytest.fail("Whisper must not run for a transcript"))
    out = webapp.work("﻿We ship on Friday.\n".encode(), ".MD", threading.Event(), ["key_points"], "detailed")
    assert seen == {"transcript": "We ship on Friday.", "keys": ["key_points"], "detail": "detailed"}
    assert out["summary"].startswith("## Key Points")


def test_empty_transcript_file_is_rejected():
    with pytest.raises(HTTPException) as e:
        webapp.work(b"  \n", ".txt", threading.Event(), ["key_points"], "concise")
    assert e.value.status_code == 400
