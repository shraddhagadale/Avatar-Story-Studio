"""
Manages the messages array sent to the Groq API on every call.

Core responsibility: build a valid, token-safe messages list from raw segments.

Why this matters:
- The Groq API requires strict user/assistant alternation
- Free tier is ~6K tokens/min — we must keep requests lean
- This is the "memory" layer: everything the AI knows comes through here
"""

from config import TOKEN_TRIM_THRESHOLD


def estimate_tokens(text: str) -> int:
    """
    Fast estimate: ~4 characters per token for English prose.
    Accuracy: ~85-90%. Good enough for budget decisions.

    """
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # per-message formatting overhead
    return total


def build_messages(system_prompt: str, segments: list[dict]) -> list[dict]:
    """
    Converts story segments into a valid chat completions messages array.

    Steps:
    1. System prompt always first
    2. Map segments → {role, content} messages
    3. Merge consecutive same-role messages (API requires strict alternation)
    4. If last message is assistant, append implicit "Continue" so the model
       knows it should generate next — not required by all models but safer
    5. Trim oldest story messages if token budget exceeded (never drop system)
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for seg in segments:
        messages.append({"role": seg["role"], "content": seg["content"]})

    # ── Step 3: Merge consecutive same-role messages ──────────────────────────
    # Example: two AI continuations with no user turn between them get merged.
    merged: list[dict] = [messages[0]]  # system always first
    for msg in messages[1:]:
        if merged[-1]["role"] == msg["role"]:
            merged[-1] = {
                "role": msg["role"],
                "content": merged[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            merged.append(msg)
    messages = merged

    # ── Step 4: Ensure final message is from user ─────────────────────────────
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Continue the story."})

    # ── Step 5: Trim oldest segments if over token budget ────────────────────
    # We never remove index 0 (system prompt) or the last message (current turn)
    while estimate_messages_tokens(messages) > TOKEN_TRIM_THRESHOLD and len(messages) > 3:
        messages.pop(1)  # drop oldest story segment after system prompt

    return messages


def split_segments_for_summary(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Splits segments by token count — removes oldest segments until remaining
    segments are under TOKEN_SUMMARIZE_THRESHOLD. More principled than splitting
    by count since segments vary widely in length.
    """
    recent = list(segments)
    oldest = []

    while recent and estimate_messages_tokens(
        [{"role": s["role"], "content": s["content"]} for s in recent]
    ) > TOKEN_TRIM_THRESHOLD // 2:
        oldest.insert(0, recent.pop(0))

    return oldest, recent


def get_full_story_text(segments: list[dict]) -> str:
    """
    Concatenates all segments into readable prose for display and extraction.
    User contributions are inlined with context markers.
    """
    parts = []
    for seg in segments:
        if seg["role"] == "user" and seg["content"] != "Continue the story.":
            parts.append(f'[You wrote: "{seg["content"]}"]')
        else:
            parts.append(seg["content"])
    return "\n\n".join(parts)
