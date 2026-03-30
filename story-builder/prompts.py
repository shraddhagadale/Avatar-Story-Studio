"""
All prompt templates live here.
Keeping prompts isolated means you can tune them without touching API or UI logic.
"""

from config import GENRE_RULES


def build_system_prompt(title: str, genre: str, characters: list[dict]) -> str:
    """
    Rebuilt on every API call so it always reflects the latest character list.
    This is the core consistency mechanism — the model sees genre rules and
    known characters on every single turn.
    """
    char_section = ""
    if characters:
        char_lines = "\n".join(f"  - {c['name']}: {c['description']}" for c in characters)
        char_section = f"\n\nKNOWN CHARACTERS (maintain these exactly):\n{char_lines}"

    return f"""You are a masterful collaborative storyteller co-writing a {genre} story titled "{title}".

GENRE: {genre.upper()}
{GENRE_RULES[genre]}

CONSISTENCY RULES (follow these strictly):
1. Never contradict any established fact, character name, ability, or event from the story so far.
2. Maintain the narrative voice and tone set in the opening paragraph.
3. Every continuation must flow naturally — no abrupt scene jumps without transition.
4. When the user contributes text, treat it as story canon and build upon it.
5. Respect cause-and-effect: consequences of earlier choices must persist.
6. Do not introduce new characters, locations, or abilities without narrative grounding.{char_section}

WRITING STYLE:
- Third-person past tense narrative
- Vivid sensory details (sound, smell, texture — not just visuals)
- Show don't tell: reveal character through action and dialogue
- Each continuation: 1–2 paragraphs, approximately 150–250 words
- Maintain consistent pacing with what has been established"""


# ── Opening ──────────────────────────────────────────────────────────────────

def build_opening_prompt(title: str, genre: str, hook: str) -> str:
    return f"""You are starting a new {genre} story titled "{title}".

The writer has provided this opening hook:
\"\"\"{hook}\"\"\"

Write a strong, immersive opening paragraph (150–250 words) that:
- Establishes the world, atmosphere, and tone of a {genre} story
- Introduces at least one compelling character or situation grounded in the hook
- Uses vivid, sensory language
- Ends with a narrative hook that propels the reader forward

Write only the story paragraph. No title, no preamble, no meta-commentary."""


# ── Branching choices ─────────────────────────────────────────────────────────

CHOICES_INSTRUCTION = """Continue the story with 1–2 paragraphs (150–200 words), then present exactly 3 branching paths.

After your story continuation, add:

**Choice A — [short title]:** [one sentence describing this path]
**Choice B — [short title]:** [one sentence describing this path]
**Choice C — [short title]:** [one sentence describing this path]

Guidelines for the choices:
- Choice A: the cautious / expected path
- Choice B: a balanced middle ground
- Choice C: the bold, risky, or unexpected path
- Each choice must be meaningfully different and consistent with the story world
- Do NOT write the continuation for any choice — only the one-sentence description"""


# ── Character extraction ──────────────────────────────────────────────────────

CHARACTER_EXTRACTION_SYSTEM = (
    "You are a character extraction assistant. "
    "You only output valid JSON — no markdown, no explanation, no code blocks."
)

def build_character_extraction_prompt(story_text: str) -> str:
    # Only send the last ~3000 chars to keep this call cheap
    excerpt = story_text[-3000:] if len(story_text) > 3000 else story_text
    return f"""Extract all named characters from this story excerpt.

Return ONLY this exact JSON format:
{{"characters": [{{"name": "Character Name", "description": "one sentence: role + defining trait"}}]}}

If there are no named characters, return: {{"characters": []}}

Story excerpt:
{excerpt}"""
