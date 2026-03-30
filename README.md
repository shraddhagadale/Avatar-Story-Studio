# Story Builder

Story Builder is an AI-assisted collaborative storytelling application built with Python, Streamlit, and Groq. It provides guided multi-turn narrative generation with consistency controls, character tracking, and interactive branching.

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

`llama-3.3-70b-versatile` currently provides a strong quality and instruction-following balance for long-form creative writing plus structured tasks such as option generation and JSON extraction. For lower cost and faster responses, `llama-3.1-8b-instant` can be used as a direct replacement in `config.py`.

## Memory and Consistency Strategy

The app uses full conversation history injection on every completion call. Each request includes:

1. Current system prompt
2. Complete ordered story history (user and assistant turns)
3. Current user action

```text
API call = system prompt + story history + current action
```

### Rationale

- High continuity because prior events are always in context
- Deterministic reasoning about model inputs
- Appropriate for this project scale given Groq's context window

### Context Guard

`context_manager.py` estimates token usage (`len(text) // 4`) before generation. When usage approaches the configured limit, oldest segments are trimmed and the UI displays a warning.

### Prompt Rebuild Policy

The system prompt is rebuilt on each turn with genre constraints and the current extracted character set. This acts as a durable story contract even if older segments are trimmed for token control.

## Prompting Design

### System Prompt Composition

The system prompt is structured in this order:

1. Role and genre context
2. Genre-specific rule set
3. Core consistency directives
4. Known characters (dynamically injected)
5. Writing style guidance

### Branching Choices

The choices instruction is injected as an ephemeral user message and is not persisted in story history. This keeps narrative context clean while enforcing strict output formatting.

Choice generation follows a deliberate risk ladder (`A/B/C`: conservative, balanced, bold) to produce meaningfully different user paths.

### Character Extraction

After each assistant turn, a separate low-temperature extraction call (`temp=0.0`) runs with `response_format={"type": "json_object"}`. This returns structured character metadata and is more reliable than regex-based extraction for aliases and renamed entities.

## Features

- Live character tracker in the sidebar with generated descriptions
- Undo last turn to revert the latest user and assistant exchange
- Export full story as Markdown
- Exponential backoff handling for rate limits (1s, 2s, 4s)
- Runtime temperature control for generation style

## Lessons from Early Iterations

Initial prompts were too generic. On longer stories, output drift appeared as naming inconsistencies, forgotten attributes, and tone shifts.

This was mitigated by introducing explicit genre rule sets and injecting refreshed character context into every system prompt rebuild.

A second issue was partial continuation output during choice generation. This was resolved by adding explicit constraints that require one-line option descriptions only.

## Next Improvements

1. Add long-context summarization to replace hard trimming with structured recap blocks
2. Add genre remix capability for style transformations without plot loss
3. Improve streaming UX with an explicit stop-generation control
4. Persist stories in SQLite for session continuity
5. Extend character tracking to include relationship evolution over time
