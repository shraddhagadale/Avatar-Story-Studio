# Story Builder

An AI-powered collaborative storytelling app built with Python, Streamlit, and Groq.

## Setup

```bash
# 1. Clone / unzip the project
cd story-builder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# Edit .env and replace with your key from https://console.groq.com

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Model & Provider

**Model:** `llama-3.3-70b-versatile` via **Groq**

**Why Groq?**
Groq's inference speed (~500 tokens/sec) makes the streaming feel like a real co-writer thinking in real time. Batch responses from slower providers feel broken by comparison. The free tier is generous enough for demos and the SDK is OpenAI-compatible, making it trivial to swap providers.

**Why llama-3.3-70b-versatile?**
Strong creative writing quality with good instruction-following for structured tasks (choice generation, JSON extraction). For faster/cheaper operation, `llama-3.1-8b-instant` is a drop-in swap in `config.py`.

---

## Memory & Consistency Strategy

**Approach: Full history injection on every call.**

Every API call receives the complete story history as a structured `messages` array (system + alternating user/assistant turns). The model sees everything that has happened — no retrieval, no summarisation, no lossy compression.

```
API call = system_prompt + all segments as messages + current user action
```

**Why full history?**
- Perfect recall: the model cannot contradict earlier events because they're in context
- Simple to reason about: what you see is what the model gets
- Correct for this scale: Groq's 128K context window handles hundreds of story turns

**Context guard:**
`context_manager.py` estimates tokens before each call (`len(text) // 4`). If the history approaches the token budget, oldest segments are trimmed. A warning appears in the sidebar when the story is getting long.

**System prompt as consistency layer:**
The system prompt is rebuilt on every call with the current genre rules and extracted character list. This means the model always has an up-to-date "story bible" — even if older segments get trimmed, the character descriptions persist.

---

## Prompt Engineering

### Main system prompt structure
```
Role + genre context
↓
Genre-specific rules (6 distinct rule sets)
↓
5 core consistency directives
↓
Known characters (dynamically injected, updated each turn)
↓
Writing style guide
```

### Branching choices
The `CHOICES_INSTRUCTION` prompt is injected as a temporary user message — it never saves to the story segments. This keeps the story history clean while giving the model precise formatting instructions.

Choices follow a deliberate A/B/C risk gradient: safe → balanced → bold. This gives users meaningful agency rather than three equivalent options.

### Character extraction
A separate low-temperature (`temp=0.0`) call with `response_format={"type": "json_object"}` runs after each AI turn. The model extracts named characters and returns structured JSON. This is cheap (~50 tokens) and handles edge cases (nicknames, titles, introduced-then-renamed characters) that regex cannot.

## What I'd Improve With Another Day

1. **Summarisation for very long stories** — Instead of trimming oldest segments, use a second LLM call to compress old chapters into a structured "story recap" block. This preserves continuity while controlling token usage.

2. **Genre Remix** — A button to rewrite the latest paragraph in a completely different genre while preserving the plot. Interesting prompt engineering challenge.

3. **Better streaming UX** — Add a "Stop generating" button mid-stream using Streamlit's `st.stop()` pattern.

4. **Persistence** — Save stories to SQLite so users can resume across sessions. Streamlit's session state is ephemeral.

5. **Smarter character tracking** — Track relationship changes between characters turn-by-turn, not just their static descriptions.
