# Story Weaver

Story Weaver is an AI-assisted collaborative storytelling application built with Python, Streamlit, and Groq. It provides guided multi-turn narrative generation with consistency controls, character tracking, and interactive branching.

## Setup

```bash
# 1) Clone the repository and enter the app directory
git clone <your-repo-url>
cd <repo-name>/story-builder

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure credentials
cp .env.example .env
# Edit .env and set your Groq API key from https://console.groq.com

# 4) Run the app
streamlit run app.py
```

The app starts at `http://localhost:8501`.

## Model and Provider

- Model: `llama-3.3-70b-versatile`
- Provider: Groq

Groq is used for low-latency token streaming, which materially improves interactive writing workflows. The implementation is OpenAI-compatible, so provider changes remain low-friction.

`llama-3.3-70b-versatile` provides a strong balance of instruction-following and creative quality for both long-form narrative generation and structured tasks such as JSON character extraction.

## Final Prompt

The system prompt is rebuilt on every turn. This is the full template:

```
You are a masterful collaborative storyteller co-writing a {genre} story titled "{title}".

GENRE: {GENRE}
{genre-specific rules}

CONSISTENCY RULES (follow these strictly):
1. Never contradict any established fact, character name, ability, or event from the story so far.
2. Maintain the narrative voice and tone set in the opening paragraph.
3. Every continuation must flow naturally — no abrupt scene jumps without transition.
4. When the user contributes text, treat it as story canon and build upon it.
5. Respect cause-and-effect: consequences of earlier choices must persist.
6. Do not introduce new characters, locations, or abilities without narrative grounding.

KNOWN CHARACTERS (maintain these exactly):
  - {name}: {description}

WRITING STYLE:
- Match the narrative voice and tense established in the opening paragraph
- Vivid sensory details (sound, smell, texture — not just visuals)
- Show don't tell: reveal character through action and dialogue
- Each continuation: 1–2 paragraphs, approximately 150–250 words
- Maintain consistent pacing with what has been established
```

The character block is dynamic — omitted on the first turn, then rebuilt after every assistant turn from a separate extraction call and injected into every subsequent prompt.

## Memory and Consistency Strategy

Every API call receives the full conversation history in order:

```text
system prompt → story history (user + assistant turns) → current action
```

As the story grows, token usage is managed in three stages:

1. **≥ 4,000 tokens** — oldest segments are summarized into a compact recap block and replaced. The summary is injected into the API context but never displayed to the user.
2. **≥ 4,500 tokens** — hard trim as a last resort if summarization was insufficient

This preserves plot memory rather than silently dropping early story events.

## Bonus Features

- **Live Character Tracker** — after every assistant turn, a separate low-temperature LLM call extracts named characters into a structured list, injected into every subsequent system prompt to prevent the model from forgetting or renaming them
- **Undo Last AI Turn** — reverts the last user + assistant exchange without losing earlier story history
- **Export as Markdown** — downloads the full story as a clean Markdown file

## One Thing That Did Not Work Well

Rate limit handling initially used fixed exponential backoff (1s → 2s → 4s). This failed when Groq's required wait time exceeded the total backoff budget — retries would exhaust and the call would fail even though waiting a bit longer would have succeeded. The app would also silently freeze with no feedback to the user.

The fix was to read Groq's `retry-after` response header first — if Groq says wait 37 seconds, we wait exactly that. If the header isn't present, we fall back to exponential backoff (1s → 2s → 4s). If Groq wants us to wait more than 30 seconds, we stop retrying immediately and raise the exception — which surfaces a clear error to the user with the exact wait time, instead of silently freezing for minutes.

## What I Would Improve Next

1. Model selection - add a dropdown so users can choose from different models based on their needs. Power users writing long-form fiction may want a larger model; casual users may prefer faster, cheaper completions.

