"""Minimal web UI for ownscribe: upload audio -> transcript + summary. Run: uvicorn webui.app:app"""
import asyncio
import dataclasses
import glob
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

import openai
from ownscribe.config import Config, TemplateConfig
from ownscribe.summarization.openai_summarizer import OpenAISummarizer
from ownscribe.summarization.prompts import clean_response
from webui import sections

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    k, _, v = line.partition("=")
    if k.strip() and not line.startswith("#"):
        os.environ.setdefault(k.strip(), v.strip())

config = Config.load()
config.summarization.host = os.environ.get("OPENAI_BASE_URL", config.summarization.host)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")  # served by the remote endpoint, nothing runs locally
SUMMARY_CONTEXT = int(os.environ.get("SUMMARY_CONTEXT", "32768"))  # tokens; bigger = fewer chunks, lower it if the model rejects long prompts
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "16384"))  # output cap; without one the endpoint may cut long notes short
# This endpoint's model "thinks" by default and the hidden reasoning tokens count against the output cap, which cut long
# notes off. "none" is fastest but undershoots the word targets; "low" meets them in ~25 s. Raise it for more depth.
SUMMARY_REASONING = os.environ.get("SUMMARY_REASONING", "low")
client = openai.OpenAI(base_url=config.summarization.host, api_key=config.summarization.api_key or "not-needed")

app = FastAPI(title="meetingnotes")


class WebSummarizer(OpenAISummarizer):
    """OpenAI summarizer that sets an output limit and refuses answers the model cut off."""

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        # Hidden reasoning shares the output cap, so a long run can be cut off. Retry once with reasoning off,
        # which always fits; only if that is cut off too do we fail (raising keeps half-written notes out of the cache).
        for effort in dict.fromkeys([SUMMARY_REASONING, "none"]):
            r = self._client.chat.completions.create(
                model=self._config.model, max_tokens=SUMMARY_MAX_TOKENS, reasoning_effort=effort,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            if r.choices[0].finish_reason != "length":
                return clean_response(r.choices[0].message.content or "")
            log.warning("notes cut off at %d tokens with reasoning=%s", SUMMARY_MAX_TOKENS, effort)
        raise HTTPException(502, "The notes were cut off by the model's output limit. Try Concise or fewer sections.")
log = logging.getLogger("uvicorn.error")

CACHE = Path(__file__).parent / ".cache"  # ponytail: plain files, no eviction; delete the folder to reset
CACHE.mkdir(exist_ok=True)


def cached(name: str, fn) -> str:
    """Return the stored result for `name`, computing and storing it on first use."""
    f = CACHE / name
    if f.exists():
        log.info("cache hit: %s", name)
        return f.read_text(encoding="utf-8")
    value = fn()
    f.write_text(value, encoding="utf-8")
    return value


class Cancelled(Exception):
    pass


def transcribe(path: str, cancel: threading.Event, key: str) -> str:
    """Downmix to 16 kHz mono mp3 in 10-min chunks so any recording fits the endpoint's size limit."""
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                            "-f", "segment", "-segment_time", "600", f"{d}/c%04d.mp3"], capture_output=True, text=True)
        chunks = sorted(glob.glob(f"{d}/c*.mp3"))
        if r.returncode or not chunks:
            raise HTTPException(400, f"Could not read audio: {r.stderr.strip()[-200:]}")
        parts = []
        for i, c in enumerate(chunks):  # finished chunks stay cached, so a cancelled run resumes where it stopped
            if cancel.is_set():
                raise Cancelled

            def send(c=c):
                with open(c, "rb") as audio:
                    return client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio).text.strip()
            parts.append(cached(f"{key}-{i}.chunk.txt", send))
        return " ".join(parts)  # ponytail: no overlap between chunks, a word on a boundary may split


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/sections")
def api_sections():
    """The selectable summary sections and the profile presets, so the page never hard-codes them."""
    return {"sections": sections.SECTIONS, "profiles": sections.PROFILES}


TEXT_SUFFIXES = {".md", ".txt"}  # an uploaded transcript skips Whisper and goes straight to the summary step


def summarize(transcript: str, cancel: threading.Event, keys: list, detail: str) -> dict:
    if cancel.is_set():
        raise Cancelled
    tpl = sections.build_template(keys, detail)
    cfg = dataclasses.replace(
        config,
        summarization=dataclasses.replace(config.summarization, template="web", context_size=SUMMARY_CONTEXT),
        templates={**config.templates, "web": TemplateConfig(**tpl)},
    )
    summarizer = WebSummarizer(cfg.summarization, cfg.templates)
    if not summarizer.is_available():
        raise HTTPException(502, f"Summarizer not reachable at {config.summarization.host}")
    try:
        cleaned = sections.clean_transcript(transcript)  # the model sees it tighter; the download stays raw
        skey = hashlib.sha256(
            f"{config.summarization.model}|{tpl['system_prompt']}|{tpl['prompt']}|{cleaned}".encode()).hexdigest()
        summary = cached(f"{skey}.summary.md", lambda: sections.normalize_lists(summarizer.summarize(cleaned)))
    finally:
        summarizer.close()
    return {"transcript": transcript, "summary": summary}


def work(data: bytes, suffix: str, cancel: threading.Event, keys: list, detail: str) -> dict:
    if suffix.lower() in TEXT_SUFFIXES:
        transcript = data.decode("utf-8-sig", errors="replace").strip()
        if not transcript:
            raise HTTPException(400, "The transcript file is empty.")
        return summarize(transcript, cancel, keys, detail)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
    try:
        try:
            key = hashlib.sha256(data).hexdigest() + "-" + WHISPER_MODEL
            transcript = cached(f"{key}.transcript.txt", lambda: transcribe(tmp.name, cancel, key))
        except openai.APIError as e:
            raise HTTPException(502, f"Whisper endpoint error: {e}")
        return summarize(transcript, cancel, keys, detail)
    finally:
        os.unlink(tmp.name)


@app.post("/process")
async def process(request: Request, file: UploadFile = File(...),
                  selected: str | None = Form(None, alias="sections"), detail: str = Form("concise")):
    """Runs the job in a thread; if the browser disconnects (Cancel), stop before the next step."""
    try:  # no field at all means "everything"; an explicit empty list is an error
        wanted = sections.PROFILES["complete"] if selected is None else json.loads(selected)
        keys = [k for k in wanted if k in {s["key"] for s in sections.SECTIONS}]
    except (ValueError, TypeError):
        keys = []
    if not keys:
        raise HTTPException(400, "Select at least one summary section.")
    cancel = threading.Event()
    job = asyncio.create_task(asyncio.to_thread(
        work, await file.read(), Path(file.filename or "audio").suffix or ".wav", cancel, keys,
        detail if detail in ("concise", "detailed") else "concise"))
    while not job.done():
        if await request.is_disconnected():
            cancel.set()  # ponytail: the in-flight Whisper/LLM call finishes first, then work stops
            break
        await asyncio.sleep(0.5)
    try:
        return await job
    except Cancelled:
        log.info("job cancelled by client")
        return {"cancelled": True}
