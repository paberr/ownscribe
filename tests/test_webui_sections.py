from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webui"))

import sections


def test_profiles_match_the_agreed_sets():
    assert sections.PROFILES["action"] == [
        "exec_summary", "main_discussions", "action_plan", "decisions", "next_steps",
    ]
    assert sections.PROFILES["information"] == [
        "exec_summary", "main_topics", "key_points", "questions", "follow_up",
    ]
    assert sections.PROFILES["complete"] == [s["key"] for s in sections.SECTIONS]


def test_prompt_contains_only_selected_sections_in_canonical_order():
    prompt = sections.build_template(["key_points", "exec_summary"])["prompt"]
    assert "## Executive Summary" in prompt and "## Key Points" in prompt
    assert prompt.index("## Executive Summary") < prompt.index("## Key Points")
    assert "## Action Plan" not in prompt


def test_prompt_only_uses_braces_for_the_transcript_placeholder():
    prompt = sections.build_template(sections.PROFILES["complete"])["prompt"]
    assert prompt.count("{") == 1
    assert "{transcript}" in prompt.format(transcript="{transcript}")  # str.format must not raise


def test_detailed_raises_the_caps():
    concise = sections.build_template(["key_points"], "concise")["prompt"]
    detailed = sections.build_template(["key_points"], "detailed")["prompt"]
    assert "At most 6 bullets" in concise and "30-50 words" in concise
    assert "At most 10 bullets" in detailed and "45-70 words" in detailed


def test_key_points_are_complete_explanatory_sentences():
    concise = sections.build_template(["key_points"], "concise")["prompt"]
    detailed = sections.build_template(["key_points"], "detailed")["prompt"]
    assert "complete, self-contained sentence" in concise and "30-50" in concise
    assert "complete, self-contained sentence" in detailed and "45-70" in detailed
    assert "20 words" not in concise


def test_priority_sections_get_longer_items_than_secondary_ones():
    prompt = sections.build_template(["main_discussions", "decisions"], "concise")["prompt"]
    assert "40-60" in prompt.split("## Decisions Made")[0]
    assert "15-25" in prompt.split("## Decisions Made")[1]


def test_system_prompt_names_the_four_priority_sections():
    system = sections.build_template(["key_points"])["system_prompt"]
    for label in ("Main Discussions", "Key Points", "Action Plan", "Questions & Discussions"):
        assert label in system


def test_action_plan_word_ranges():
    assert "15-30" in sections.build_template(["action_plan"], "concise")["prompt"]
    assert "20-40" in sections.build_template(["action_plan"], "detailed")["prompt"]


def test_complete_profile_formats_with_one_placeholder():
    prompt = sections.build_template(sections.PROFILES["complete"])["prompt"]
    assert prompt.count("{") == 1 and prompt.count("}") == 1
    assert "hello" in prompt.format(transcript="hello")


def test_unknown_detail_falls_back_to_concise():
    keys = sections.PROFILES["complete"]
    assert sections.build_template(keys, "bogus") == sections.build_template(keys, "concise")


def test_clean_transcript_drops_fillers_and_repeats_but_keeps_content():
    text = (
        "Okay. Yeah okay. We launch the project in November. Alice will send the budget report on Friday. "
        "Alice will send the budget report on Friday. Alice will send the budget reports on Friday. "
        "Bob books the room. Thank you."
    )
    assert sections.clean_transcript(text) == (
        "We launch the project in November. Alice will send the budget report on Friday. Bob books the room."
    )


def test_clean_transcript_collapses_immediate_repeats_but_keeps_distant_short_sentences():
    far = " ".join(f"Item {hashlib.md5(str(i).encode()).hexdigest()} was reviewed today." for i in range(25))
    text = f"We agreed. We agreed. {far} We agreed."
    out = sections.clean_transcript(text)
    assert out.startswith("We agreed. Item ")  # immediate repeat collapsed
    assert out.endswith("We agreed.")  # same short sentence much later is kept


def test_prompt_requires_list_format_for_every_section_but_the_summary():
    prompt = sections.build_template(["key_points", "exec_summary"], "detailed")["prompt"]
    assert "Every section except Executive Summary is a markdown list" in prompt
    assert "starting with `- `" in prompt


def test_normalize_lists_turns_paragraph_items_into_bullets():
    raw = (
        "## Key Points\nFirst full sentence here.\n\nSecond one,\nwrapped over two lines.\n\n"
        "## Action Plan\n**Ann** - send it.\n**Bob** - book it."
    )
    assert sections.normalize_lists(raw) == (
        "## Key Points\n- First full sentence here.\n- Second one, wrapped over two lines.\n\n"
        "## Action Plan\n- **Ann** - send it.\n- **Bob** - book it."
    )


def test_normalize_lists_keeps_existing_lists_summary_and_empty_sections():
    raw = (
        "## Executive Summary\nProse stays as prose.\n\nSecond paragraph.\n\n"
        "## Decisions Made\n* one\n1. two\n  - nested\n\n## Follow-up Elements\nNone mentioned."
    )
    assert sections.normalize_lists(raw) == (
        "## Executive Summary\nProse stays as prose.\n\nSecond paragraph.\n\n"
        "## Decisions Made\n- one\n- two\n  - nested\n\n## Follow-up Elements\nNone mentioned."
    )


def test_normalize_lists_is_idempotent():
    once = sections.normalize_lists("## Key Points\nA.\n\nB.")
    assert sections.normalize_lists(once) == once
