"""
Single module for all Groq API calls.
Swap the provider here and nowhere else.
"""

import os
import json
import time

from groq import Groq
from groq import RateLimitError, APIConnectionError, APIStatusError
from dotenv import load_dotenv

from config import MODEL, MAX_TOKENS_OUTPUT
from prompts import (
    build_opening_prompt,
    CHARACTER_EXTRACTION_SYSTEM,
    build_character_extraction_prompt,
)

load_dotenv()

# Lazy singleton — initialised on first use
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. "
                "Create a .env file with GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 3):
    """
    Exponential backoff on rate limit errors.
    Raises the last exception if all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s → 2s → 4s
            time.sleep(wait)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_opening(title: str, genre: str, hook: str, temperature: float) -> str:
    """Non-streaming. Returns the opening paragraph as a string."""
    prompt = build_opening_prompt(title, genre, hook)

    def _call():
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=MAX_TOKENS_OUTPUT,
        )
        return response.choices[0].message.content.strip()

    return _with_retry(_call)


def stream_continuation(messages: list[dict], temperature: float):
    """
    Generator — yields text chunks as they arrive from Groq.
    Use with Streamlit's st.write_stream() or a manual placeholder loop.

    Why streaming? It makes the app feel alive. A 3-second wait for a batch
    response feels broken; watching text appear feels like a co-writer thinking.
    """
    def _call():
        return get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS_OUTPUT,
            stream=True,
        )

    stream = _with_retry(_call)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def get_choices_response(messages: list[dict], temperature: float) -> str:
    """
    Non-streaming. Returns the full text: story continuation + 3 choices.
    We use non-streaming here because we need to parse the full response
    before displaying anything.
    """
    def _call():
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS_OUTPUT,
        )
        return response.choices[0].message.content.strip()

    return _with_retry(_call)


def extract_characters(story_text: str) -> list[dict]:
    """
    Runs a cheap, low-temperature extraction call.
    Returns list of {"name": str, "description": str}.
    Fails silently — character tracking is non-critical.

    Why a separate call? Regex can't reliably extract characters from
    narrative prose. The LLM handles edge cases (nicknames, titles, etc.)
    for ~50 tokens cost per turn.
    """
    prompt = build_character_extraction_prompt(story_text)

    try:
        def _call():
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": CHARACTER_EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # deterministic for extraction
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        raw = _with_retry(_call)
        data = json.loads(raw)
        return data.get("characters", [])
    except Exception:
        # Character extraction failing should never crash the story
        return []
