"""Tests for the semantic search module."""

from __future__ import annotations

from pathlib import Path

import click

from ownscribe.search import (
    _answer_from_transcripts,
    _build_summary_chunks,
    _discover_meetings,
    _estimate_tokens,
    _extract_keywords,
    _extract_quotes,
    _find_relevant_meetings,
    _key_phrases,
    _keyword_fallback,
    _parse_relevant_ids,
    _rank_meetings,
    _verify_quotes,
)

# -- Helpers --


def _make_meeting_dir(base: Path, folder_name: str, summary: str, transcript: str | None = None) -> None:
    folder = base / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.md").write_text(summary)
    if transcript is not None:
        (folder / "transcript.md").write_text(transcript)


class FakeSummarizer:
    """A fake summarizer that returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[tuple[str, str, bool]] = []
        self._responses = list(responses or [])
        self._call_idx = 0

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.calls.append((system_prompt, user_prompt, json_mode))
        if self._responses:
            resp = self._responses[self._call_idx % len(self._responses)]
            self._call_idx += 1
            return resp
        return '{"relevant": []}'


# -- Discovery tests --


class TestDiscoverMeetings:
    def test_discover_meetings(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary of Q1 planning")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "Daily standup notes")

        meetings, skipped = _discover_meetings(tmp_path, since=None, limit=None)
        assert len(meetings) == 2
        assert skipped == 0
        # newest first
        assert meetings[0].folder_name == "2026-02-13_1501_quarterly-planning"
        assert meetings[1].folder_name == "2026-02-12_0930_team-standup"
        assert "Quarterly Planning" in meetings[0].display_name
        assert "2026-02-13 15:01" in meetings[0].display_name

    def test_discover_meetings_no_slug(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-16_1433", "Summary without title")
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary of Q1 planning")

        meetings, skipped = _discover_meetings(tmp_path, since=None, limit=None)
        assert len(meetings) == 2
        assert skipped == 0
        # newest first
        assert meetings[0].folder_name == "2026-02-16_1433"
        assert meetings[0].display_name == "2026-02-16 14:33"
        assert meetings[1].folder_name == "2026-02-13_1501_quarterly-planning"

    def test_discover_meetings_since_filter(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "Summary")
        _make_meeting_dir(tmp_path, "2026-01-10_1000_old-meeting", "Summary")

        meetings, _ = _discover_meetings(tmp_path, since="2026-02-01", limit=None)
        assert len(meetings) == 2
        assert all("2026-02" in m.folder_name for m in meetings)

    def test_discover_meetings_limit(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "Summary")
        _make_meeting_dir(tmp_path, "2026-02-11_1000_old-meeting", "Summary")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=2)
        assert len(meetings) == 2
        # Should be the 2 newest
        assert meetings[0].folder_name == "2026-02-13_1501_quarterly-planning"
        assert meetings[1].folder_name == "2026-02-12_0930_team-standup"

    def test_discover_meetings_skips_no_summary(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")
        # Folder with no summary file
        no_summary = tmp_path / "2026-02-12_0930_team-standup"
        no_summary.mkdir()
        (no_summary / "transcript.md").write_text("some transcript")

        meetings, skipped = _discover_meetings(tmp_path, since=None, limit=None)
        assert len(meetings) == 1
        assert skipped == 1
        assert meetings[0].folder_name == "2026-02-13_1501_quarterly-planning"

    def test_discover_meetings_empty_dir(self, tmp_path):
        meetings, skipped = _discover_meetings(tmp_path, since=None, limit=None)
        assert meetings == []
        assert skipped == 0

    def test_discover_meetings_nonexistent_dir(self, tmp_path):
        meetings, skipped = _discover_meetings(tmp_path / "nonexistent", since=None, limit=None)
        assert meetings == []
        assert skipped == 0


# -- Token estimation --


class TestEstimateTokens:
    def test_estimate_tokens(self):
        assert _estimate_tokens("a" * 100) == 25
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("hello world") == 2


# -- Chunking --


class TestBuildSummaryChunks:
    def test_build_summary_chunks(self, tmp_path):
        # Create meetings with summaries of known size
        for i in range(5):
            _make_meeting_dir(
                tmp_path,
                f"2026-02-{10+i:02d}_1000_meeting-{i}",
                "x" * 2000,  # 500 tokens each
            )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # Small budget that forces multiple chunks
        chunks = _build_summary_chunks(meetings, context_budget=2000)
        assert len(chunks) > 1
        # All meetings accounted for
        all_ids = {m.folder_name for chunk in chunks for m in chunk}
        assert all_ids == {m.folder_name for m in meetings}

    def test_build_summary_chunks_single(self, tmp_path):
        for i in range(3):
            _make_meeting_dir(
                tmp_path,
                f"2026-02-{10+i:02d}_1000_meeting-{i}",
                "Short summary",
            )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # Large budget - everything fits in one chunk
        chunks = _build_summary_chunks(meetings, context_budget=100000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3


# -- Parse relevant IDs --


class TestParseRelevantIds:
    def test_valid_json(self):
        assert _parse_relevant_ids('{"relevant": ["id1", "id2"]}') == ["id1", "id2"]

    def test_empty_relevant(self):
        assert _parse_relevant_ids('{"relevant": []}') == []

    def test_json_with_extra_text(self):
        response = 'Here are the results: {"relevant": ["id1"]} That is all.'
        assert _parse_relevant_ids(response) == ["id1"]

    def test_bare_array(self):
        assert _parse_relevant_ids('["id1", "id2"]') == ["id1", "id2"]

    def test_unparseable(self):
        assert _parse_relevant_ids("I don't know") is None


# -- Keyword extraction --


class TestExtractKeywords:
    def test_filters_stop_words(self):
        kw = _extract_keywords("What is the budget for the project?")
        assert "budget" in kw
        assert "project" in kw
        assert "the" not in kw
        assert "is" not in kw

    def test_strips_punctuation(self):
        kw = _extract_keywords("Hello, world! How's it going?")
        assert "hello" in kw
        assert "world" in kw

    def test_all_stop_words_fallback(self):
        kw = _extract_keywords("is the a")
        # Falls back to returning all words when only stop words remain
        assert len(kw) > 0
        assert "is" in kw

    def test_empty_string(self):
        kw = _extract_keywords("")
        assert kw == set()


# -- Keyword fallback --


class TestKeywordFallback:
    def test_match_on_summary(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "budget review fiscal quarter")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "standup tasks blockers")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        matches = _keyword_fallback("What about the budget?", meetings)
        assert len(matches) == 1
        assert matches[0].folder_name == "2026-02-13_1501_quarterly-planning"

    def test_match_on_transcript_only(self, tmp_path):
        _make_meeting_dir(
            tmp_path, "2026-02-13_1501_quarterly-planning",
            "Generic meeting notes",  # summary has no useful keywords
            "Alice discussed the infrastructure migration timeline",  # transcript does
        )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        matches = _keyword_fallback("infrastructure migration", meetings)
        assert len(matches) == 1
        assert matches[0].folder_name == "2026-02-13_1501_quarterly-planning"

    def test_no_match(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "budget review fiscal quarter")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        matches = _keyword_fallback("kubernetes deployment", meetings)
        assert len(matches) == 0


# -- Find relevant meetings --


class TestFindRelevantMeetings:
    def test_single_batch(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary about deadlines and Q1")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "Daily standup notes")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        fake = FakeSummarizer(['{"relevant": ["2026-02-13_1501_quarterly-planning"]}'])

        result = _find_relevant_meetings(fake, "What about the deadline?", meetings, 100000)
        assert len(result) == 1
        assert result[0].folder_name == "2026-02-13_1501_quarterly-planning"
        assert fake.calls[0][2] is True  # json_mode

    def test_drops_unknown_ids(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        fake = FakeSummarizer(['{"relevant": ["2026-02-13_1501_quarterly-planning", "nonexistent-id"]}'])

        result = _find_relevant_meetings(fake, "question", meetings, 100000)
        assert len(result) == 1
        assert result[0].folder_name == "2026-02-13_1501_quarterly-planning"

    def test_multi_batch(self, tmp_path):
        for i in range(5):
            _make_meeting_dir(
                tmp_path,
                f"2026-02-{10+i:02d}_1000_meeting-{i}",
                "x" * 2000,
            )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # Meetings are newest-first, each gets its own batch:
        # batch 0=meeting-4, 1=meeting-3, 2=meeting-2, 3=meeting-1, 4=meeting-0
        responses = [
            '{"relevant": ["2026-02-14_1000_meeting-4"]}',
            '{"relevant": []}',
            '{"relevant": []}',
            '{"relevant": ["2026-02-11_1000_meeting-1"]}',
            '{"relevant": []}',
        ]
        fake = FakeSummarizer(responses)

        result = _find_relevant_meetings(fake, "question", meetings, context_size=2000)
        result_ids = {m.folder_name for m in result}
        assert "2026-02-14_1000_meeting-4" in result_ids
        assert "2026-02-11_1000_meeting-1" in result_ids

    def test_json_fallback_regex(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # Malformed but extractable JSON
        fake = FakeSummarizer(['Sure! {"relevant": ["2026-02-13_1501_quarterly-planning"]}'])

        result = _find_relevant_meetings(fake, "question", meetings, 100000)
        assert len(result) == 1

    def test_total_fallback(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_quarterly-planning", "Summary")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "Summary")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # Completely unparseable
        fake = FakeSummarizer(["I have no idea what to return"])

        result = _find_relevant_meetings(fake, "question", meetings, 100000)
        # All meetings should be included as fallback
        assert len(result) == 2


class TestFindRelevantMeetingsKeywordFallback:
    def test_llm_empty_falls_back_to_keywords(self, tmp_path):
        """When the LLM returns no relevant IDs, keyword fallback finds the meeting."""
        _make_meeting_dir(
            tmp_path,
            "2026-02-13_1501_quarterly-planning",
            "Generic meeting summary",  # summary is too vague for LLM
            "Alice discussed the infrastructure migration timeline in detail",
        )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        # LLM returns empty relevant list
        fake = FakeSummarizer(['{"relevant": []}'])

        result = _find_relevant_meetings(
            fake, "infrastructure migration", meetings, 100000,
        )
        assert len(result) == 1
        assert result[0].folder_name == "2026-02-13_1501_quarterly-planning"


# -- Ranking --


class TestRankMeetings:
    def test_keyword_overlap(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_budget-review", "budget review fiscal quarter spending")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_team-standup", "standup tasks blockers")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        ranked = _rank_meetings("What was discussed about the budget?", meetings)
        assert ranked[0].folder_name == "2026-02-13_1501_budget-review"

    def test_speaker_boost(self, tmp_path):
        _make_meeting_dir(tmp_path, "2026-02-13_1501_planning", "Bob discussed the timeline")
        _make_meeting_dir(tmp_path, "2026-02-12_0930_standup", "Anna mentioned the deadline and blockers")

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        ranked = _rank_meetings("What did Anna say about the deadline?", meetings)
        assert ranked[0].folder_name == "2026-02-12_0930_standup"


# -- Answer from transcripts --


class TestAnswerFromTranscripts:
    def test_answer(self, tmp_path):
        _make_meeting_dir(
            tmp_path,
            "2026-02-13_1501_quarterly-planning",
            "Summary",
            "Alice: The deadline is March 15th.\nBob: Got it.",
        )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        fake = FakeSummarizer(["Anna said the deadline is March 15th."])

        answer, skipped = _answer_from_transcripts(fake, "What about the deadline?", meetings, 100000)
        assert "March 15th" in answer
        assert skipped == 0

    def test_budget_overflow(self, tmp_path):
        # Create a meeting with a very large transcript
        _make_meeting_dir(
            tmp_path,
            "2026-02-13_1501_big-meeting",
            "Summary",
            "x" * 100000,
        )
        _make_meeting_dir(
            tmp_path,
            "2026-02-12_0930_small-meeting",
            "Summary",
            "Alice: Short transcript.",
        )

        meetings, _ = _discover_meetings(tmp_path, since=None, limit=None)
        fake = FakeSummarizer(["Answer based on available transcripts."])

        # Tiny budget: only the small one should fit
        answer, skipped = _answer_from_transcripts(fake, "question", meetings, context_size=2000)
        # The big one should be skipped, function should still return an answer
        assert "Answer based on available transcripts." in answer
        assert skipped > 0


# -- Quote verification --


class TestVerifyQuotes:
    def test_phrase_match(self):
        answer = '> The deadline for Q1 deliverables is March 15th but I think we should aim for March 10th'
        transcripts = {
            "meeting-1": (
                "The deadline for Q1 deliverables is March 15th but I think"
                " we should aim for March 10th to have buffer time."
            ),
        }
        result = _verify_quotes(answer, transcripts)
        assert "[unverified]" not in result

    def test_not_found(self):
        answer = '> The completely fabricated quote that does not exist in any transcript at all'
        transcripts = {
            "meeting-1": "Alice: Let's discuss the budget.\nBob: Sure, sounds good.",
        }
        result = _verify_quotes(answer, transcripts)
        assert "[unverified]" in result

    def test_empty_transcripts(self):
        answer = "> Some quote here"
        result = _verify_quotes(answer, {})
        assert result == answer


class TestExtractQuotes:
    def test_blockquotes(self):
        text = "Text before\n> This is a quote\n> continued here\nText after"
        quotes = _extract_quotes(text)
        assert any("This is a quote" in q for q in quotes)

    def test_inline_quotes(self):
        text = 'She said "The deadline is definitely March 15th for all deliverables" in the meeting.'
        quotes = _extract_quotes(text)
        assert any("deadline" in q for q in quotes)


class TestKeyPhrases:
    def test_short_quote(self):
        phrases = _key_phrases("too short")
        assert phrases == []

    def test_medium_quote(self):
        phrases = _key_phrases("one two three four five six seven eight nine ten")
        assert len(phrases) >= 1
        assert all(len(p.split()) >= 8 for p in phrases)


# -- Integration test --


class TestAskIntegration:
    def test_end_to_end(self, httpserver, tmp_path, monkeypatch):
        # Set up mock meetings
        _make_meeting_dir(
            tmp_path,
            "2026-02-13_1501_quarterly-planning",
            "Discussed Q1 deadlines. Anna mentioned March 15th target.",
            "[00:01:00] Anna: The deadline is March 15th.\n[00:01:30] Bob: Acknowledged.",
        )
        _make_meeting_dir(
            tmp_path,
            "2026-02-12_0930_team-standup",
            "Daily standup. Discussed blockers.",
            "[00:00:30] Charlie: No updates.\n[00:01:00] Dave: Still blocked.",
        )

        # Stage 1 response (find)
        find_response = {
            "message": {
                "role": "assistant",
                "content": '{"relevant": ["2026-02-13_1501_quarterly-planning"]}',
            },
            "done": True,
        }
        # Stage 2 response (answer)
        answer_response = {
            "message": {
                "role": "assistant",
                "content": (
                    'The deadline is March 15th.\n\n'
                    '**2026-02-13 15:01 — Quarterly Planning**\n'
                    '- **Anna** [00:01:00]: "The deadline is March 15th."'
                ),
            },
            "done": True,
        }
        # Auto-detect context size calls /api/show
        show_response = {
            "model_info": {"general.context_length": 8192},
        }
        httpserver.expect_ordered_request("/api/show", method="POST").respond_with_json(show_response)
        httpserver.expect_ordered_request("/api/chat", method="POST").respond_with_json(find_response)
        httpserver.expect_ordered_request("/api/chat", method="POST").respond_with_json(answer_response)

        from ownscribe.config import Config

        config = Config()
        config.output.dir = str(tmp_path)
        config.summarization.host = httpserver.url_for("")
        config.summarization.backend = "ollama"
        config.summarization.model = "test-model"

        from ownscribe.search import ask

        output_lines: list[str] = []
        monkeypatch.setattr(click, "echo", lambda msg="": output_lines.append(str(msg)))

        ask(config, "What did Anna say about the deadline?", since=None, limit=None)

        output = "\n".join(output_lines)
        assert "Found 1 relevant" in output
        assert "Quarterly Planning" in output
        assert "March 15th" in output


# -- OpenAI json_mode fallback tests --


def _openai_ok_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _openai_400_response() -> tuple[dict, int]:
    return (
        {"error": {
            "message": "'response_format.type' must be 'json_schema' or 'text'",
            "type": "invalid_request_error",
        }},
        400,
    )


class TestOpenAIChatJsonModeFallback:
    def test_openai_chat_json_mode_fallback(self, httpserver):
        """json_object and json_schema both fail → falls back to no response_format."""
        from ownscribe.config import SummarizationConfig
        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        body_400, status_400 = _openai_400_response()
        ep = "/v1/chat/completions"
        # 1st attempt (json_object) → 400
        httpserver.expect_ordered_request(ep, method="POST").respond_with_json(
            body_400, status=status_400,
        )
        # 2nd attempt (json_schema) → 400
        httpserver.expect_ordered_request(ep, method="POST").respond_with_json(
            body_400, status=status_400,
        )
        # 3rd attempt (no response_format) → 200
        httpserver.expect_ordered_request(ep, method="POST").respond_with_json(
            _openai_ok_response('{"relevant": ["meeting-1"]}'),
        )

        cfg = SummarizationConfig()
        cfg.host = httpserver.url_for("")
        cfg.model = "test-model"
        summarizer = OpenAISummarizer(cfg)

        result = summarizer.chat("system", "user", json_mode=True)
        assert '"relevant"' in result or "meeting-1" in result

    def test_openai_chat_json_schema_fallback(self, httpserver):
        """json_object fails → falls back to json_schema which succeeds."""
        from ownscribe.config import SummarizationConfig
        from ownscribe.summarization.openai_summarizer import OpenAISummarizer

        body_400, status_400 = _openai_400_response()
        ep = "/v1/chat/completions"
        # 1st attempt (json_object) → 400
        httpserver.expect_ordered_request(ep, method="POST").respond_with_json(
            body_400, status=status_400,
        )
        # 2nd attempt (json_schema) → 200
        httpserver.expect_ordered_request(ep, method="POST").respond_with_json(
            _openai_ok_response('{"relevant": ["meeting-2"]}'),
        )

        cfg = SummarizationConfig()
        cfg.host = httpserver.url_for("")
        cfg.model = "test-model"
        summarizer = OpenAISummarizer(cfg)

        result = summarizer.chat("system", "user", json_mode=True)
        assert "meeting-2" in result
