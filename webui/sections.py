"""Summary sections the user can pick from, grouped as in the UI (action-oriented vs information-oriented)."""
import difflib
import re

# Order here is the order sections appear in the prompt and in the UI.
SECTIONS = [
    {"key": "exec_summary", "label": "Executive Summary", "group": "action"},
    {"key": "main_discussions", "label": "Main Discussions", "group": "action"},
    {"key": "action_plan", "label": "Action Plan", "group": "action"},
    {"key": "decisions", "label": "Decisions Made", "group": "action"},
    {"key": "next_steps", "label": "Next Steps", "group": "action"},
    {"key": "main_topics", "label": "Main Topics", "group": "info"},
    {"key": "key_points", "label": "Key Points", "group": "info"},
    {"key": "questions", "label": "Questions & Discussions", "group": "info"},
    {"key": "follow_up", "label": "Follow-up Elements", "group": "info"},
]

_ACTION = [s["key"] for s in SECTIONS if s["group"] == "action"]
_INFO = [s["key"] for s in SECTIONS if s["group"] == "info"]

PROFILES = {
    "action": _ACTION,
    "information": ["exec_summary", *_INFO],
    "complete": [s["key"] for s in SECTIONS],
}

# Per-section instruction; {n} is the item cap, {words} the words-per-item range, both from _LIMITS.
_RULES = {
    "exec_summary": "{n} sentences of plain prose: what the meeting was for and what came out of it.",
    "main_discussions": (
        "At most {n} bullets, each `**Topic** - ` followed by 2-3 complete sentences ({words} words in total) "
        "covering what was discussed, the positions or reasons given, and the outcome."
    ),
    "action_plan": (
        "At most {n} bullets, each `**Owner** - ` followed by the task as a full sentence ({words} words), "
        "with the context if stated and the deadline if stated. Explicit commitments only."
    ),
    "decisions": (
        "At most {n} bullets ({words} words each). Only decisions that were explicitly made, "
        "not ideas that were floated."
    ),
    "next_steps": "At most {n} bullets ({words} words each): the upcoming steps in chronological order.",
    "main_topics": "At most {n} short labels of 2-5 words each. No full sentences.",
    "key_points": (
        "At most {n} bullets. Each bullet is one complete, self-contained sentence (two at most) of {words} words "
        "that explains the full scenario: who or what is involved, the context, and why it matters or what follows. "
        "A reader must understand it without the transcript. No fragments, labels or telegraphic phrasing. "
        "Only the major points."
    ),
    "questions": (
        "At most {n} bullets ({words} words each): a question left unanswered or a point still being debated, "
        "with one clause on why it is still open."
    ),
    "follow_up": "At most {n} bullets ({words} words each): documents, data or people that need follow-up later.",
}
# (max items, words per item) per section; main_topics/exec_summary use words=None (their rules fix the wording).
_LIMITS = {
    "concise": {
        "main_discussions": (6, "40-60"), "key_points": (6, "30-50"), "action_plan": (8, "15-30"),
        "questions": (6, "20-40"), "decisions": (5, "15-25"), "next_steps": (4, "12-20"),
        "follow_up": (4, "10-20"), "main_topics": (6, None), "exec_summary": ("3-4", None),
    },
    "detailed": {
        "main_discussions": (10, "60-90"), "key_points": (10, "45-70"), "action_plan": (14, "20-40"),
        "questions": (10, "30-55"), "decisions": (8, "20-35"), "next_steps": (6, "15-30"),
        "follow_up": (6, "15-30"), "main_topics": (8, None), "exec_summary": ("4-6", None),
    },
}

_SYSTEM = (
    "You turn meeting transcripts into clear, well-organised notes. Be selective and never pad: report what "
    "matters, not everything that was said. Four sections deserve the most depth and the fullest sentences: "
    "Main Discussions, Key Points, Action Plan and Questions & Discussions. Cover those thoroughly and "
    "completely. The other sections (Executive Summary, Decisions Made, Next Steps, Main Topics, "
    "Follow-up Elements) stay shorter but must still be accurate and complete."
)

_PROMPT = """Write notes for the transcript below.

Output ONLY these sections, in this order, each starting with exactly the heading shown:

{sections}

Rules:
- No introduction, no closing remarks, no sections other than the ones above.
- Every section except Executive Summary is a markdown list: one item per line, each line starting with `- `. Never merge several items into a paragraph, however long the items are.
- The transcript may repeat itself. Mention each fact once, and never repeat an item across sections.
- Ignore greetings, filler and off-topic chatter. Do not invent owners, dates or facts.
- If nothing in the transcript fits a section, write exactly `None mentioned.` under its heading.
- Write in the language of the transcript.

---

Transcript:
{{transcript}}"""


def build_template(keys, detail="concise"):
    """Return {"system_prompt", "prompt"} asking for only the selected sections, in canonical order.

    The prompt keeps a literal {transcript} placeholder (ownscribe fills it with str.format).
    """
    limits = _LIMITS.get(detail, _LIMITS["concise"])
    chosen = [s for s in SECTIONS if s["key"] in set(keys)]
    block = "\n\n".join(
        f"## {s['label']}\n{_RULES[s['key']].format(n=limits[s['key']][0], words=limits[s['key']][1])}"
        for s in chosen
    )
    return {"system_prompt": _SYSTEM, "prompt": _PROMPT.format(sections=block)}


_FILLERS = {"okay", "ok", "yeah", "yes", "yep", "right", "sure", "correct", "exactly", "alright",
            "uh", "um", "hmm", "mm", "thanks", "thank", "you", "bye", "so", "well"}


def _norm(sentence):
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


def clean_transcript(text):
    """Drop filler-only sentences and repeats so the model sees each statement once."""
    seen, window, kept = set(), [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        n = _norm(sentence)
        if not n or all(w in _FILLERS for w in n.split()):
            continue
        if len(n.split()) >= 6 and n in seen:  # far-apart repeats only count for longer sentences; short ones can recur naturally
            continue
        if any(difflib.SequenceMatcher(None, n, p).ratio() >= 0.9 for p in window):
            continue
        seen.add(n)
        window = (window + [n])[-20:]  # ponytail: only recent sentences are fuzzy-compared; O(n*20)
        kept.append(sentence.strip())
    return " ".join(kept)


_LIST_ITEM = re.compile(r"^(\s*)(?:[-*]|\d+[.)])\s+")


def normalize_lists(markdown):
    """Make every section except the Executive Summary a '- ' bullet list.

    With long items some models skip the list instruction and write one paragraph per item; this restores the list.
    """
    out = []
    for block in re.split(r"(?m)^(?=## )", markdown.strip()):
        if not block.strip():  # the split yields an empty first piece when the text starts with a heading
            continue
        head, _, body = block.partition("\n")
        text = body.strip()
        if not head.startswith("## ") or head[3:].strip().lower() == "executive summary" or not text \
                or re.fullmatch(r"(?i)none mentioned\.?", text):
            out.append(block.strip())
            continue
        lines = text.splitlines()
        if any(_LIST_ITEM.match(line) for line in lines):  # already a list: only unify the markers
            text = "\n".join(_LIST_ITEM.sub(r"\1- ", line) for line in lines)
        else:
            items = re.split(r"\n\s*\n", text)
            if len(items) == 1:  # no blank lines: one item per line
                items = lines
            text = "\n".join(f"- {' '.join(item.split())}" for item in items if item.strip())
        out.append(f"{head}\n{text}")
    return "\n\n".join(out)
