"""
Stateless helper functions: parsing, formatting, exporting.
No API calls, no Streamlit imports — pure logic.
"""

import re
from datetime import datetime


GENRE_EMOJI = {
    "Fantasy": "🧙",
    "Sci-Fi": "🚀",
    "Mystery": "🔍",
    "Romance": "💕",
    "Horror": "👻",
    "Comedy": "😄",
}


def genre_badge(genre: str) -> str:
    return f"{GENRE_EMOJI.get(genre, '📖')} {genre}"


def parse_choices(raw_text: str) -> tuple[str, list[str]]:
    """
    Splits an AI response into (story_continuation, [choice_a, choice_b, choice_c]).

    Expected format from the model:
        ...story text...

        **Choice A — Title:** Description sentence.
        **Choice B — Title:** Description sentence.
        **Choice C — Title:** Description sentence.

    Returns (full_text, []) if parsing fails — caller shows the full text as
    a regular continuation (graceful degradation).
    """
    pattern = r"\*\*Choice\s+[ABC]\s*(?:—|-)[^:]*:\*\*\s*(.+?)(?=\n\*\*Choice\s+[ABC]|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    choices = [c.strip() for c in matches if c.strip()]

    if len(choices) == 3:
        # Everything before the first "**Choice A" is the story continuation
        split = re.search(r"\*\*Choice\s+A", raw_text)
        story_part = raw_text[: split.start()].strip() if split else raw_text
        return story_part, choices

    return raw_text, []


def export_to_markdown(title: str, genre: str, segments: list[dict]) -> str:
    """Formats the full story as a clean Markdown document for download."""
    lines = [
        f"# {title}",
        f"*Genre: {genre}*  ",
        f"*Exported: {datetime.now().strftime('%B %d, %Y at %H:%M')}*",
        "",
        "---",
        "",
    ]

    for seg in segments:
        if seg["role"] == "user" and seg["content"] != "Continue the story.":
            lines.append(f"> **You:** {seg['content']}")
        elif seg["role"] == "assistant":
            lines.append(seg["content"])
        lines.append("")

    return "\n".join(lines)


def safe_filename(title: str) -> str:
    """Converts a story title to a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:60] or "story"
