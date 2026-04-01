"""
Story Builder — AI-powered collaborative storytelling app.
UI layer only: reads from session_state, calls helpers, updates state.
"""

import streamlit as st
from groq import RateLimitError, APIConnectionError

from config import GENRES, GENRE_RULES, TOKEN_WARN_THRESHOLD, TOKEN_SUMMARIZE_THRESHOLD
from context_manager import build_messages, get_full_story_text, estimate_tokens, split_segments_for_summary
from llm import generate_opening, stream_continuation, get_choices_response, extract_characters, summarize_segments
from prompts import build_system_prompt, CHOICES_INSTRUCTION
from utils import genre_badge, parse_choices, export_to_markdown, safe_filename

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Story Weaver · myAvatar",
    page_icon="📖",
    layout="wide",

)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
  /* ── Design tokens ───────────────────────────────────────────── */
  :root {
    --navy:           #000B3B;
    --navy-hover:     #001166;
    --text-primary:   #0a0a0a;
    --text-muted:     #64748B;
    --border:         rgba(226,232,240,0.6);
    --radius-card:    15px;
    --radius-input:   14px;
    --radius-btn:     8px;
    --orb-purple:     rgba(167,139,250,0.22);
    --orb-pink:       rgba(251,182,206,0.20);
    --orb-blue:       rgba(147,197,253,0.15);
    --glass-bg:       rgba(255,255,255,0.65);
    --glass-border:   rgba(255,255,255,0.80);
    --glass-shadow:   0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
    --glass-hover:    0 8px 32px rgba(0,0,0,0.10);
    --grad-heading:   linear-gradient(90deg, #FF6B54 0%, #A855F7 60%, #6366F1 100%);
    --transition:     all 0.2s ease;
    --bg-card:        rgba(255,255,255,0.65);
    --bg-section:     rgba(237,233,251,0.45);
    --shadow-card:    0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-hover:   0 8px 32px rgba(0,0,0,0.10);
    --shadow-input:   0 1px 4px rgba(0,0,0,0.04);
  }

  /* ── Base + gradient orb background ─────────────────────────── */
  html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
  }
  .stApp {
    background-color: #FFFFFF !important;
    background-image:
      radial-gradient(ellipse 70% 60% at 0% 50%,   var(--orb-purple) 0%, transparent 65%),
      radial-gradient(ellipse 60% 55% at 100% 65%,  var(--orb-pink)   0%, transparent 65%),
      radial-gradient(ellipse 45% 40% at 85% 5%,    var(--orb-blue)   0%, transparent 60%),
      radial-gradient(ellipse 35% 30% at 15% 90%,   var(--orb-pink)   0%, transparent 55%) !important;
    min-height: 100vh;
  }
  .main, section[data-testid="stSidebar"] ~ div,
  [data-testid="stAppViewContainer"],
  [data-testid="stVerticalBlock"] {
    background: transparent !important;
  }

  /* ── Glassmorphism fallback for devices without backdrop-filter  */
  @supports not (backdrop-filter: blur(1px)) {
    .ma-card, .story-block, .user-block,
    [data-testid="stSidebar"],
    .stExpander, .ma-badge, .genre-selector .stButton > button {
      background: rgba(255,255,255,0.92) !important;
    }
  }

  /* ── Hide Streamlit chrome ───────────────────────────────────── */
  [data-testid="stHeader"]     { display: none !important; }
  [data-testid="stToolbar"]    { display: none !important; }
  [data-testid="stDecoration"] { display: none !important; }
  footer                       { display: none !important; }
  #MainMenu                    { display: none !important; }

  /* ── Layout ──────────────────────────────────────────────────── */
  .block-container {
    max-width: 900px !important;
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    background: transparent !important;
  }
  /* Tighten Streamlit's default inter-element gap in the main content area */
  [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] > div:has(.story-block),
  [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] > div:has(.user-block) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
  }

  /* ── Typography ──────────────────────────────────────────────── */
  h1, h2, h3, h4,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2 {
    font-family: 'Inter', sans-serif !important;
  }
  h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
  }
  h2 { font-size: 1.4rem !important; font-weight: 700 !important; letter-spacing: -0.3px !important; }
  h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
  p  { font-size: 15px; line-height: 1.85; }

  /* ── Gradient accent text ────────────────────────────────────── */
  .avatar-gradient {
    background: var(--grad-heading);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
  }

  /* ── Hero ────────────────────────────────────────────────────── */
  .ma-hero {
    padding: 0.6rem 0 0.5rem 0;
    border-bottom: 1px solid rgba(226,232,240,0.5);
    margin-bottom: 0.5rem;
    background: transparent;
  }
  .ma-hero h1 {
    font-size: 2rem !important;
    line-height: 1.15 !important;
    margin-bottom: 0.35rem;
  }
  .ma-hero p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
    max-width: 520px;
  }

  /* ── Cards ───────────────────────────────────────────────────── */
  .ma-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: var(--radius-card);
    padding: 1.25rem 1.5rem;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-card);
    margin-bottom: 0.6rem;
    transition: var(--transition);
  }
  .ma-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
  }
  .ma-card-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
  }
  .ma-card-value {
    font-size: 14px;
    color: var(--text-primary);
    line-height: 1.6;
  }

  /* ── Info column wrapper — subtle lavender tint ─────────────── */
  .info-col {
    background: rgba(237,233,251,0.25);
    border-radius: 16px;
    padding: 4px 4px;
  }

  /* ── Selected Genre card — high visual weight ────────────────── */
  .genre-card-info {
    border-left: 3px solid rgba(168,85,247,0.55) !important;
    background: rgba(168,85,247,0.05) !important;
  }

  /* ── How It Works card — receded, supporting context ─────────── */
  .how-it-works-card {
    background: rgba(255,255,255,0.45) !important;
    box-shadow: none !important;
  }
  .how-it-works-card .ma-card-label {
    color: #94A3B8 !important;
  }

  /* ── Story display — AI paragraphs ──────────────────────────── */
  .story-block {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: var(--radius-card);
    padding: 1.1rem 1.4rem;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-card);
    margin-bottom: 0.5rem;
    font-size: 16px;
    line-height: 1.65;
    color: var(--text-primary);
  }
  /* Override global p rule — child paragraphs inside story block */
  .story-block p {
    line-height: 1.65 !important;
    font-size: 16px !important;
    margin-bottom: 0.6rem;
  }
  .story-block p:last-child { margin-bottom: 0; }

  /* ── Story display — user contributions ─────────────────────── */
  .user-block {
    background: rgba(237,233,251,0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-left: 2px solid rgba(167,139,250,0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.4rem 1rem;
    margin-bottom: 0.4rem;
    color: var(--text-muted);
    font-style: italic;
    font-size: 13px;
    line-height: 1.6;
  }

  /* ── Streaming cursor blink ──────────────────────────────────── */
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }
  .cursor-blink {
    animation: blink 0.9s step-start infinite;
    font-weight: 400;
    color: #A855F7;
  }

  /* ── "What happens next?" section header ─────────────────────── */
  .choices-header {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.2px;
  }
  .choices-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 14px;
  }

  /* ── Choice buttons — risk badge pills ──────────────────────── */
  .choice-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 20px;
    margin-right: 6px;
    vertical-align: middle;
  }
  .badge-safe     { background: rgba(34,197,94,0.12);  color: #16a34a; }
  .badge-balanced { background: rgba(168,85,247,0.12); color: #7C3AED; }
  .badge-bold     { background: rgba(239,68,68,0.10);  color: #DC2626; }

  /* ── Primary button ──────────────────────────────────────────── */
  .stButton > button[kind="primary"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-btn) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
    box-shadow: 0 1px 3px rgba(0,11,59,0.2) !important;
    transition: var(--transition) !important;
    padding: 0.6rem 1.25rem !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: var(--navy-hover) !important;
    box-shadow: 0 4px 12px rgba(0,11,59,0.3) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button[kind="primary"]:active { transform: scale(0.98) !important; }

  /* ── Secondary buttons ───────────────────────────────────────── */
  .stButton > button:not([kind="primary"]) {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-btn) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
  }
  .stButton > button:not([kind="primary"]):hover {
    border-color: #94A3B8 !important;
    background: rgba(237,233,251,0.6) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:not([kind="primary"]):active { transform: scale(0.98) !important; }

  /* ── Download button ─────────────────────────────────────────── */
  .stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-btn) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
  }
  .stDownloadButton > button:hover {
    border-color: #94A3B8 !important;
    background: rgba(237,233,251,0.6) !important;
  }

  /* ── Inputs ──────────────────────────────────────────────────── */
  .stTextInput input,
  .stTextArea textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important; /* 16px minimum — prevents iOS Safari auto-zoom on focus */
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-input) !important;
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: var(--shadow-input) !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
  }
  /* Placeholder — consistent across all inputs and textareas */
  .stTextInput input::placeholder,
  .stTextArea textarea::placeholder {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    color: #9CA3AF !important;
    font-style: italic !important;
  }

  /* Re-enable resize handle — Streamlit strips it by default */
  .stTextArea textarea { resize: vertical !important; min-height: 80px !important; }

  .stTextInput input:focus,
  .stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(0,11,59,0.08) !important;
    outline: none !important;
  }

  /* ── Selectbox (kept as fallback) ────────────────────────────── */
  .stSelectbox > div > div {
    font-family: 'Inter', sans-serif !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-input) !important;
    background: rgba(255,255,255,0.8) !important;
    box-shadow: var(--shadow-input) !important;
    cursor: pointer !important;
  }

  /* ── Slider ──────────────────────────────────────────────────── */
  [data-baseweb="slider"] [role="slider"] { background: var(--navy) !important; }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: rgba(237,233,251,0.55) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(226,232,240,0.5) !important;
  }
  [data-testid="stSidebar"] .block-container {
    padding-top: 0.75rem !important;
    max-width: 100% !important;
  }

  /* ── Expander ────────────────────────────────────────────────── */
  .stExpander {
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-card) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
  }

  /* ── Divider ─────────────────────────────────────────────────── */
  hr { border-color: var(--border) !important; margin: 0.25rem 0 !important; }

  /* ── Alerts ──────────────────────────────────────────────────── */
  .stAlert { border-radius: var(--radius-card) !important; }

  /* ── Genre badge pill ────────────────────────────────────────── */
  .ma-badge {
    display: inline-block;
    background: rgba(237,233,251,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 11px;
    letter-spacing: 0.3px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
  }

  /* ── Form labels ─────────────────────────────────────────────── */
  .ma-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
    display: block;
  }

  /* ── Creativity slider range hint ───────────────────────────── */
  .slider-hint {
    display: flex;
    justify-content: space-between;
    margin-top: -28px;
    padding: 0 2px;
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.2px;
  }

  /* ── Caption / muted ─────────────────────────────────────────── */
  .stCaption, small {
    color: var(--text-muted) !important;
    font-size: 12px !important;
  }

  /* ── Left info panel — sticky, always visible ───────────────── */
  .info-panel {
    position: sticky;
    top: 1.5rem;
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
    padding: 1.25rem;
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: var(--radius-card);
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-card);
  }
  .info-panel-section { margin-bottom: 0.25rem; }
  .info-panel-divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
  }
  .info-panel-rules {
    font-size: 12px;
    color: #374151;
    line-height: 1.6;
    margin-top: 0.4rem;
  }
  .info-panel-char {
    margin-top: 0.75rem;
    font-size: 13px;
    color: var(--text-primary);
  }
  .info-panel-char-desc {
    font-size: 11px;
    color: #6B7280;
    margin-top: 0.2rem;
    line-height: 1.5;
  }
  .info-panel-empty {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 0.4rem;
  }

  /* ── Story screen header — title + genre + actions in one row ── */
  .story-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.4rem 0 0.3rem 0;
    margin-bottom: 0;
  }
  .story-header-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
    white-space: nowrap;
  }
  .story-header-genre {
    display: inline-block;
    background: rgba(237,233,251,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
  }
  .story-header-spacer { flex: 1; }
  .story-header-action {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    text-decoration: none;
    padding: 5px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }
  .story-header-action:hover {
    background: rgba(237,233,251,0.5);
    border-color: var(--navy);
    color: var(--text-primary);
  }

  /* ── Hide Streamlit input instruction hint ───────────────────── */
  [data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── State initialisation ──────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "story_started": False,
        "title": "",
        "genre": "Fantasy",
        "hook": "",
        "segments": [],
        "characters": [],
        "pending_choices": [],
        "temperature": 0.8,
        "error_msg": None,
        "undo_stack": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_and_clear_error():
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        st.session_state.error_msg = None


def _handle_llm_error(e: Exception):
    if isinstance(e, RateLimitError):
        retry_after = None
        if hasattr(e, "response") and e.response is not None:
            retry_after = e.response.headers.get("retry-after")
        wait_msg = f"Try again in ~{int(float(retry_after))}s." if retry_after else "Try again in a moment."
        st.session_state.error_msg = f"Rate limit reached for Groq free tier. {wait_msg}"
    elif isinstance(e, APIConnectionError):
        st.session_state.error_msg = "🔌 Connection error. Check your internet connection."
    else:
        st.session_state.error_msg = f"Something went wrong: {str(e)}"


def _maybe_summarize():
    """
    When story context approaches the token limit, summarize the oldest segments
    into a compact recap instead of hard-trimming. Triggered at TOKEN_SUMMARIZE_THRESHOLD.
    Splits by token count (not arbitrary half) for a principled reduction.
    Verifies token count after summarizing — hard trim in build_messages() is the fallback.
    """
    story_text = get_full_story_text(st.session_state.segments)
    if estimate_tokens(story_text) >= TOKEN_SUMMARIZE_THRESHOLD:
        old, recent = split_segments_for_summary(st.session_state.segments)
        if not old:
            return  # nothing to summarize — let hard trim handle it
        turn_count = len(st.session_state.segments)
        summary_text = summarize_segments(old)
        summary_segment = {
            "role": "assistant",
            "content": f"[Story so far — after {turn_count} turns: {summary_text}]"
        }
        st.session_state.segments = [summary_segment] + recent

        # Verify token count came down after summarizing
        after_text = get_full_story_text(st.session_state.segments)
        if estimate_tokens(after_text) >= TOKEN_SUMMARIZE_THRESHOLD:
            # Summary didn't reduce enough — hard trim in build_messages() handles the rest
            import logging
            logging.getLogger("llm").warning(
                f"Summarization insufficient — still at ~{estimate_tokens(after_text)} tokens after summary"
            )


# Genre metadata — used both for the card grid and for the guide panel
GENRE_META = {
    "Fantasy": {
        "emoji": "🧙",
        "desc":  "Magic, ancient lore, quests, mythical creatures, epic stakes.",
    },
    "Sci-Fi":  {
        "emoji": "🚀",
        "desc":  "Technology, space, the future, alternate timelines, big ideas.",
    },
    "Mystery": {
        "emoji": "🔍",
        "desc":  "Clues, suspects, hidden motives, reveals, fair-play twists.",
    },
    "Romance": {
        "emoji": "💕",
        "desc":  "Emotional depth, tension, near-misses, earned connection.",
    },
    "Horror":  {
        "emoji": "👻",
        "desc":  "Atmosphere, dread, ambiguity, psychological tension, fear.",
    },
    "Comedy":  {
        "emoji": "😄",
        "desc":  "Character-driven humor, escalating absurdity, callbacks, timing.",
    },
}


@st.dialog("Start a new story?")
def _confirm_new_story_dialog():
    st.warning("This will clear your current story and cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, start fresh", type="primary", use_container_width=True):
            for k in ["story_started", "title", "genre", "hook", "segments",
                      "characters", "pending_choices", "undo_stack", "error_msg",
                      "_confirm_new_story"]:
                st.session_state.pop(k, None)
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("_confirm_new_story", None)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SETUP SCREEN
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.story_started:

    st.markdown("""
    <div class="ma-hero" style="text-align:center">
        <h1 style="font-family:'Inter',sans-serif;font-weight:800;letter-spacing:-1px">
            <span class="avatar-gradient">Story Weaver</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_guide = st.columns([3, 2], gap="large")

    with col_form:
        # Story title
        st.markdown('<span class="ma-label">Story Title</span>', unsafe_allow_html=True)
        title = st.text_input(
            "Story Title",
            placeholder="Example: The Last Signal from Meridian Station",
            max_chars=100,
            label_visibility="collapsed",
        )

        # Genre — dropdown
        st.markdown('<span class="ma-label">Genre</span>', unsafe_allow_html=True)
        genre = st.selectbox(
            "Genre",
            options=GENRES,
            index=GENRES.index(st.session_state.genre),
            format_func=lambda g: f"{GENRE_META[g]['emoji']} {g}",
            label_visibility="collapsed",
        )
        st.session_state.genre = genre

        # Opening hook
        st.markdown('<span class="ma-label">Opening Hook</span>', unsafe_allow_html=True)
        hook = st.text_area(
            "Opening Hook",
            placeholder=(
                "Describe your world, characters, or the opening situation.\n\n"
                "Example: A lone communications officer at the edge of the solar system "
                "intercepts a distress signal in her own voice, from a frequency "
                "decommissioned 40 years ago."
            ),
            height=120,
            label_visibility="collapsed",
        )

        # Creativity slider — compact inline range hint always visible
        st.slider(
            "🎨 Creativity",
            min_value=0.1, max_value=1.5, value=st.session_state.temperature, step=0.1,
            help="Drag to control how adventurous the AI is. Balanced (0.7–0.9) is the sweet spot for most stories.",
            key="setup_creativity",
            on_change=lambda: setattr(st.session_state, 'temperature', st.session_state.setup_creativity),
        )
        st.markdown("""
        <div class="slider-hint">
          <span>Focused</span>
          <span>Balanced&nbsp;✦</span>
          <span>Wild</span>
        </div>
        """, unsafe_allow_html=True)

        _show_and_clear_error()

        if st.button("Start the Story →", type="primary", use_container_width=True):
            if not title.strip():
                st.error("Please enter a story title.")
            elif not hook.strip():
                st.error("Please provide an opening hook.")
            else:
                with st.spinner("Crafting your opening…"):
                    try:
                        opening = generate_opening(title.strip(), genre, hook.strip(), st.session_state.temperature)
                        st.session_state.title = title.strip()
                        st.session_state.genre = genre
                        st.session_state.hook = hook.strip()
                        st.session_state.segments = [{"role": "assistant", "content": opening}]
                        st.session_state.characters = extract_characters(opening)
                        st.session_state.story_started = True
                        st.rerun()
                    except Exception as e:
                        _handle_llm_error(e)
                        st.rerun()

    with col_guide:
        g = st.session_state.genre
        meta = GENRE_META[g]
        # Top offset: pushes cards down to align with the Genre picker row.
        # Story Title label + input ≈ 80px before Genre label appears.
        st.markdown(f"""
        <div class="info-col">
          <!-- Selected Genre — high visual weight, purple accent, reacts to picks -->
          <div class="ma-card genre-card-info">
            <div class="ma-card-label">Selected Genre</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
              <span style="font-size:1.5rem;line-height:1">{meta['emoji']}</span>
              <span style="font-size:15px;font-weight:700;color:var(--text-primary)">{g}</span>
            </div>
            <div class="ma-card-value">{meta['desc']}</div>
            <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
              <div class="ma-card-label" style="margin-bottom:6px">Story Rules</div>
              <div class="ma-card-value" style="font-size:12px;color:#374151">{GENRE_RULES[g]}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STORY SCREEN
# ─────────────────────────────────────────────────────────────────────────────

else:

    # ── Main content ──────────────────────────────────────────────────────────


    # Dialog trigger — fires when New Story button was clicked
    if st.session_state.get("_confirm_new_story"):
        _confirm_new_story_dialog()

    # ── Story header — title + genre + export + new story on one line ────────
    _genre_meta = GENRE_META[st.session_state.genre]
    _export_content = export_to_markdown(
        st.session_state.title,
        st.session_state.genre,
        st.session_state.segments,
    )
    _safe_fn = safe_filename(st.session_state.title)

    st.markdown("""
    <style>
      /* Keep header buttons on one line */
      [data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] button,
      [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-secondary"] {
        white-space: nowrap !important;
      }
    </style>
    """  , unsafe_allow_html=True)

    with st.container():
        hdr_title, hdr_export, hdr_new = st.columns([4, 1.5, 1.5])
        with hdr_title:
            st.markdown(f"""
            <div class="story-header">
              <span class="story-header-title">{st.session_state.title}</span>
              <span class="story-header-genre">{_genre_meta['emoji']} {st.session_state.genre}</span>
            </div>
            """, unsafe_allow_html=True)
        with hdr_export:
            st.download_button(
                "📥 Export",
                data=_export_content,
                file_name=f"{_safe_fn}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with hdr_new:
            if st.button("＋ New", use_container_width=True, key="new_story_btn"):
                st.session_state._confirm_new_story = True
                st.rerun()

    st.divider()

    # ── Story context — collapsible rules + characters ────────────────────────
    _char_items_md = "\n".join(
        f"- **{c.get('name', 'Unknown')}** — {c.get('description', '')}"
        for c in st.session_state.characters
    ) or "_None yet._"

    with st.expander("📋 Story Rules & Characters", expanded=False):
        rules_col, chars_col = st.columns(2)
        with rules_col:
            st.markdown(f"**Story Rules**\n\n{GENRE_RULES[st.session_state.genre]}")
        with chars_col:
            st.markdown(f"**Characters**\n\n{_char_items_md}")

    # ── Story segments — natural page flow ────────────────────────────────────
    for seg in st.session_state.segments:
        if seg["role"] == "assistant":
            if seg["content"].startswith("[Story so far:"):
                continue  # internal summary — API context only, not displayed
            st.markdown(f'<div class="story-block">{seg["content"]}</div>',
                        unsafe_allow_html=True)
        elif seg["content"] not in ("Continue the story.", ""):
            st.markdown(f'<div class="user-block">✏️ {seg["content"]}</div>',
                        unsafe_allow_html=True)

    # Placeholder — only visible while streaming, collapses when empty
    stream_placeholder = st.empty()

    # ── Choice mode ───────────────────────────────────────────────────────
    if st.session_state.pending_choices:
        st.markdown("""
        <div class="choices-header">What happens next?</div>
        <div class="choices-sub">Pick a path — each choice takes the story in a different direction.</div>
        """, unsafe_allow_html=True)

        risk_labels = [
            ("Safe",     "badge-safe"),
            ("Balanced", "badge-balanced"),
            ("Bold",     "badge-bold"),
        ]

        for i, choice_text in enumerate(st.session_state.pending_choices):
            letter = ["A", "B", "C"][i]
            risk, _ = risk_labels[i]
            if st.button(
                f"{letter}.  {choice_text}",
                key=f"choice_{i}",
                use_container_width=True,
                help=f"Risk level: {risk}",
            ):
                st.session_state.segments.append(
                    {"role": "user", "content": f"I choose: {choice_text}"}
                )
                st.session_state.pending_choices = []

                with st.spinner("Condensing story memory…"):
                    _maybe_summarize()
                system_prompt = build_system_prompt(
                    st.session_state.title,
                    st.session_state.genre,
                    st.session_state.characters,
                )
                messages = build_messages(system_prompt, st.session_state.segments)

                try:
                    full_text = ""
                    for chunk in stream_continuation(messages, st.session_state.temperature):
                        full_text += chunk
                        stream_placeholder.markdown(
                            f'<div class="story-block">{full_text}<span class="cursor-blink">▌</span></div>',
                            unsafe_allow_html=True,
                        )
                    stream_placeholder.empty()
                    st.session_state.segments.append({"role": "assistant", "content": full_text})
                    updated = extract_characters(get_full_story_text(st.session_state.segments), st.session_state.characters)
                    if updated:
                        st.session_state.characters = updated
                except Exception as e:
                    _handle_llm_error(e)

                st.rerun()

    # ── Normal input mode ─────────────────────────────────────────────────
    else:
        st.markdown('<span class="ma-label">Steer the story</span>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Steer the story",
            placeholder="Add a line or two to guide the next turn — or leave blank and let the AI decide…",
            height=80,
            label_visibility="collapsed",
            key="user_text_input",
        )

        # Creativity slider — above action buttons
        st.slider(
            "🎨 Creativity",
            min_value=0.1, max_value=1.5,
            value=st.session_state.temperature,
            step=0.1,
            help="Drag to control how adventurous the AI is. Balanced (0.7–0.9) is the sweet spot.",
            key="story_creativity",
            on_change=lambda: setattr(st.session_state, 'temperature', st.session_state.story_creativity),
        )
        st.markdown("""
        <div class="slider-hint">
          <span>Focused</span>
          <span>Balanced&nbsp;✦</span>
          <span>Wild</span>
        </div>
        """, unsafe_allow_html=True)

        story_text = get_full_story_text(st.session_state.segments)
        est_tokens = estimate_tokens(story_text)
        if est_tokens > TOKEN_WARN_THRESHOLD:
            st.warning(f"Story is getting long (~{est_tokens:,} tokens) — oldest turns may be summarized.")

        _show_and_clear_error()

        # Action row: Choices · Continue · Undo — all on one line
        col_choices, col_continue, col_undo = st.columns([3, 3, 2], gap="small")

        with col_choices:
            if st.button("🎭  Give Me Choices", use_container_width=True):
                if user_input.strip():
                    st.session_state.segments.append(
                        {"role": "user", "content": user_input.strip()}
                    )
                with st.spinner("Condensing story memory…"):
                    _maybe_summarize()
                system_prompt = build_system_prompt(
                    st.session_state.title,
                    st.session_state.genre,
                    st.session_state.characters,
                )
                messages = build_messages(system_prompt, st.session_state.segments)
                messages.append({"role": "user", "content": CHOICES_INSTRUCTION})

                with st.spinner("Generating story branches…"):
                    try:
                        raw = get_choices_response(messages, st.session_state.temperature)
                        story_part, choices = parse_choices(raw)

                        if choices:
                            if story_part.strip():
                                st.session_state.segments.append(
                                    {"role": "assistant", "content": story_part}
                                )
                            st.session_state.pending_choices = choices
                            updated = extract_characters(get_full_story_text(st.session_state.segments), st.session_state.characters)
                            if updated:
                                st.session_state.characters = updated
                        else:
                            st.session_state.error_msg = "Couldn't generate choices — try again."
                    except Exception as e:
                        _handle_llm_error(e)

                st.rerun()

        with col_continue:
            if st.button("✍️  Continue", type="primary", use_container_width=True):
                if user_input.strip():
                    st.session_state.segments.append(
                        {"role": "user", "content": user_input.strip()}
                    )
                with st.spinner("Condensing story memory…"):
                    _maybe_summarize()
                system_prompt = build_system_prompt(
                    st.session_state.title,
                    st.session_state.genre,
                    st.session_state.characters,
                )
                messages = build_messages(system_prompt, st.session_state.segments)

                try:
                    full_text = ""
                    for chunk in stream_continuation(messages, st.session_state.temperature):
                        full_text += chunk
                        stream_placeholder.markdown(
                            f'<div class="story-block">{full_text}<span class="cursor-blink">▌</span></div>',
                            unsafe_allow_html=True,
                        )
                    stream_placeholder.empty()
                    st.session_state.segments.append({"role": "assistant", "content": full_text})
                    updated = extract_characters(get_full_story_text(st.session_state.segments), st.session_state.characters)
                    if updated:
                        st.session_state.characters = updated
                except Exception as e:
                    _handle_llm_error(e)

                st.rerun()

        with col_undo:
            can_undo = len(st.session_state.segments) > 1
            if st.button("↩ Undo", use_container_width=True, disabled=not can_undo,
                         help="Remove the last AI turn and your prompt that triggered it."):
                popped = st.session_state.segments.pop()
                st.session_state.undo_stack.append(popped)
                if (st.session_state.segments
                        and st.session_state.segments[-1]["role"] == "user"):
                    st.session_state.undo_stack.append(st.session_state.segments.pop())
                st.session_state.pending_choices = []
                st.rerun()

