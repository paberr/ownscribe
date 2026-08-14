"""Tests for splitting transcripts into context-sized chunks."""

from __future__ import annotations

from ownscribe.summarization.chunking import (
    MIN_CHUNK_TOKENS,
    chunk_text,
    pack_chunks,
    split_units,
    truncate_to_tokens,
)


def _words(text: str) -> int:
    """Token counter stand-in: one token per word, so budgets read plainly."""
    return len(text.split())


class TestSplitUnits:
    def test_splits_on_segment_boundaries(self):
        text = "SPEAKER_00: Hello there.\nSPEAKER_01: Hi back."

        assert split_units(text, budget=100, count_tokens=_words) == [
            "SPEAKER_00: Hello there.",
            "SPEAKER_01: Hi back.",
        ]

    def test_drops_blank_lines(self):
        assert split_units("one\n\n  \ntwo", budget=100, count_tokens=_words) == ["one", "two"]

    def test_falls_back_to_sentences_for_an_oversized_segment(self):
        # A single line of four 3-word sentences, with room for only two at a time.
        text = "Aaa bbb ccc. Ddd eee fff. Ggg hhh iii. Jjj kkk lll."

        units = split_units(text, budget=6, count_tokens=_words)

        assert units == ["Aaa bbb ccc.", "Ddd eee fff.", "Ggg hhh iii.", "Jjj kkk lll."]

    def test_never_cuts_a_sentence_that_fits(self):
        text = " ".join(f"Sentence number {i} here." for i in range(20))

        units = split_units(text, budget=8, count_tokens=_words)

        # Every unit is a whole sentence, so each still ends in its full stop.
        assert all(unit.endswith(".") for unit in units)
        assert all(_words(unit) <= 8 for unit in units)

    def test_word_split_is_the_last_resort_for_one_long_sentence(self):
        text = " ".join(["word"] * 100)  # no sentence punctuation at all

        units = split_units(text, budget=10, count_tokens=_words)

        assert len(units) > 1
        assert all(_words(unit) <= 10 for unit in units)
        assert " ".join(units).split() == text.split()


class TestPackChunks:
    def test_packs_within_budget(self):
        units = [f"line {i} here" for i in range(10)]  # 3 tokens each

        chunks = pack_chunks(units, budget=9, overlap_budget=0, count_tokens=_words)

        assert len(chunks) > 1
        assert all(_words(chunk) <= 9 for chunk in chunks)

    def test_overlap_repeats_the_previous_tail(self):
        units = [f"unit{i} aa bb" for i in range(9)]  # 3 tokens each

        chunks = pack_chunks(units, budget=9, overlap_budget=3, count_tokens=_words)

        assert len(chunks) > 1
        # The last unit of chunk 0 opens chunk 1.
        assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]

    def test_overlap_never_pushes_a_chunk_over_budget(self):
        units = [f"unit{i} aa bb cc dd" for i in range(12)]  # 5 tokens each

        chunks = pack_chunks(units, budget=10, overlap_budget=8, count_tokens=_words)

        assert all(_words(chunk) <= 10 for chunk in chunks)

    def test_single_unit_stays_one_chunk(self):
        assert pack_chunks(["only line"], budget=100, overlap_budget=5, count_tokens=_words) == ["only line"]


class TestChunkText:
    def test_keeps_every_word(self):
        text = "\n".join(f"SPEAKER_0{i % 2}: Sentence {i} of the meeting." for i in range(400))

        chunks = chunk_text(text, budget=300, count_tokens=_words)

        assert len(chunks) > 1
        joined = " ".join(chunks)
        for i in range(400):
            assert f"Sentence {i} of" in joined

    def test_every_chunk_fits_the_budget(self):
        text = "\n".join(f"SPEAKER_00: Sentence {i} here now." for i in range(600))

        for chunk in chunk_text(text, budget=300, count_tokens=_words):
            assert _words(chunk) <= 300

    def test_budget_has_a_floor(self):
        # An absurdly small budget must not word-split the transcript to shreds.
        text = "\n".join(f"Sentence {i} here." for i in range(200))

        chunks = chunk_text(text, budget=1, count_tokens=_words)

        assert all(_words(chunk) <= MIN_CHUNK_TOKENS for chunk in chunks)

    def test_empty_text(self):
        assert chunk_text("", budget=1000, count_tokens=_words) == []


class TestTruncateToTokens:
    def test_leaves_short_text_alone(self):
        assert truncate_to_tokens("a few words", 100, _words) == "a few words"

    def test_cuts_long_text_down(self):
        text = " ".join(["word"] * 200)

        assert _words(truncate_to_tokens(text, 20, _words)) <= 20

    def test_zero_budget(self):
        assert truncate_to_tokens("anything", 0, _words) == ""
