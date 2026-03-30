"""
Story Builder — AI-powered collaborative storytelling app.
UI layer only: reads from session_state, calls helpers, updates state.
"""

import streamlit as st
from groq import RateLimitError, APIConnectionError

from config import GENRES, TOKEN_WARN_THRESHOLD
from context_manager import build_messages, get_full_story_text, estimate_tokens
from llm import generate_opening, stream_continuation, get_choices_response, extract_characters
from prompts import build_system_prompt, CHOICES_INSTRUCTION
from utils import genre_badge, parse_choices, export_to_markdown, safe_filename

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Story Builder · myAvatar",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
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
  [data-testid="stToolbar"]    { display: none !important; }
  [data-testid="stDecoration"] { display: none !important; }
  footer                       { display: none !important; }
  #MainMenu                    { display: none !important; }

  /* ── Layout ──────────────────────────────────────────────────── */
  .block-container {
    max-width: 900px !important;
    padding-top: 0.5rem !important;
    padding-bottom: 1.5rem !important;
    background: transparent !important;
  }
  /* Tighten Streamlit's default inter-element gap */
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    gap: 0.25rem !important;
  }
  /* Remove extra margin Streamlit adds around element containers */
  .element-container { margin-bottom: 0 !important; }

  /* ── Typography ──────────────────────────────────────────────── */
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
    padding: 0.75rem 0 0.75rem 0;
    border-bottom: 1px solid rgba(226,232,240,0.5);
    margin-bottom: 1rem;
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
    margin-bottom: 1rem;
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

  /* ── Genre selector — styled buttons as cards ───────────────── */
  .genre-selector .stButton > button {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 18px 8px 14px 8px !important;
    text-align: center !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
    min-height: 80px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
  }
  .genre-selector .stButton > button:hover {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-2px) !important;
    background: rgba(255,255,255,0.85) !important;
  }
  /* Selected state — primary button style overridden for genre cards */
  .genre-selector .stButton > button[kind="primary"] {
    background: rgba(168,85,247,0.10) !important;
    border-color: #A855F7 !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
    color: var(--text-primary) !important;
  }

  /* ── Story display — AI paragraphs ──────────────────────────── */
  .story-block {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: var(--radius-card);
    padding: 1.75rem 2rem;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-card);
    margin-bottom: 0.75rem;
    font-size: 16px;
    line-height: 1.95;
    color: var(--text-primary);
  }

  /* ── Story display — user contributions ─────────────────────── */
  .user-block {
    background: rgba(237,233,251,0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-left: 2px solid rgba(167,139,250,0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1.2rem;
    margin-bottom: 0.75rem;
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
    padding-top: 1.5rem !important;
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
  hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

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

  /* ── Caption / muted ─────────────────────────────────────────── */
  .stCaption, small {
    color: var(--text-muted) !important;
    font-size: 12px !important;
  }
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
        st.session_state.error_msg = (
            "⏱️ Rate limit reached — Groq free tier allows ~30 requests/min. "
            "Wait a moment and try again."
        )
    elif isinstance(e, APIConnectionError):
        st.session_state.error_msg = "🔌 Connection error. Check your internet connection."
    else:
        st.session_state.error_msg = f"Something went wrong: {str(e)}"


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


# ─────────────────────────────────────────────────────────────────────────────
# SETUP SCREEN
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.story_started:

    # Hero — gradient heading, myavatar.ai style
    st.markdown("""
    <div class="ma-hero">
        <h1>📖 <span class="avatar-gradient">Story Builder</span></h1>
        <p>Collaborate with AI to craft stories that remember every character, plot twist, and world rule — turn after turn.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_guide = st.columns([3, 2], gap="large")

    with col_form:
        # Story title
        st.markdown('<span class="ma-label">Story Title</span>', unsafe_allow_html=True)
        title = st.text_input(
            "Story Title",
            placeholder="The Last Kingdom of Ember…",
            max_chars=100,
            label_visibility="collapsed",
        )

        # Genre — visual 3×2 card grid (all options scannable at once)
        st.markdown('<span class="ma-label" style="margin-bottom:10px;display:block">Genre</span>', unsafe_allow_html=True)
        st.markdown('<div class="genre-selector">', unsafe_allow_html=True)
        g_cols = st.columns(3, gap="small")
        for idx, g in enumerate(GENRES):
            meta = GENRE_META[g]
            is_selected = st.session_state.genre == g
            with g_cols[idx % 3]:
                # Emoji on first line, name on second — button renders multi-line
                btn_label = f"{meta['emoji']}\n{g}"
                if st.button(
                    btn_label,
                    key=f"genre_btn_{g}",
                    use_container_width=True,
                    help=meta["desc"],
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.genre = g
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Sync the local variable for this render
        genre = st.session_state.genre

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Opening hook
        st.markdown('<span class="ma-label">Opening Hook</span>', unsafe_allow_html=True)
        hook = st.text_area(
            "Opening Hook",
            placeholder=(
                "Describe your world, characters, or the opening situation.\n\n"
                "Example: A disgraced knight discovers a map leading to the tomb of the "
                "last dragon, hidden beneath the city she once swore to protect."
            ),
            height=130,
            label_visibility="collapsed",
        )

        # Creativity slider — inline with label
        temperature = st.slider(
            "🎨 Creativity",
            min_value=0.1, max_value=1.5, value=st.session_state.temperature, step=0.1,
            help="0.1–0.5 = focused & consistent · 0.6–0.9 = balanced (recommended) · 1.0–1.5 = experimental",
        )
        st.session_state.temperature = temperature

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Start the Story →", type="primary", use_container_width=True):
            if not title.strip():
                st.error("Please enter a story title.")
            elif not hook.strip():
                st.error("Please provide an opening hook.")
            else:
                with st.spinner("Crafting your opening…"):
                    try:
                        opening = generate_opening(title.strip(), genre, hook.strip(), temperature)
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

        _show_and_clear_error()

    with col_guide:
        # Single dynamic card — shows selected genre details only
        g = st.session_state.genre
        meta = GENRE_META[g]
        st.markdown(f"""
        <div class="ma-card" style="margin-bottom:1rem">
          <div class="ma-card-label">Selected Genre</div>
          <div style="font-size:2rem;margin-bottom:8px;line-height:1">{meta['emoji']}</div>
          <div style="font-size:16px;font-weight:700;margin-bottom:6px">{g}</div>
          <div class="ma-card-value">{meta['desc']}</div>
        </div>
        <div class="ma-card">
          <div class="ma-card-label">How it works</div>
          <div class="ma-card-value">
            The AI receives your <b>full story history</b> on every turn —
            no summarisation, no forgetting. Characters, world rules, and
            plot threads stay consistent no matter how long it grows.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STORY SCREEN
# ─────────────────────────────────────────────────────────────────────────────

else:

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Story identity
        st.markdown(f"""
        <div style="margin-bottom:0.25rem">
          <span class="ma-badge">{genre_badge(st.session_state.genre)}</span>
        </div>
        <div style="font-size:15px;font-weight:700;margin-top:8px;line-height:1.3;color:var(--text-primary)">
          {st.session_state.title}
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Creativity slider
        st.session_state.temperature = st.slider(
            "🎨 Creativity",
            min_value=0.1, max_value=1.5,
            value=st.session_state.temperature,
            step=0.1,
            help="Adjust anytime — takes effect on the next AI turn.",
        )

        # Token usage
        story_text = get_full_story_text(st.session_state.segments)
        est_tokens = estimate_tokens(story_text)
        st.caption(f"~{est_tokens:,} tokens · {len(st.session_state.segments)} turns")
        if est_tokens > TOKEN_WARN_THRESHOLD:
            st.warning("Story is getting long — oldest turns may be trimmed.")

        st.divider()

        # Character tracker
        st.markdown('<div class="ma-card-label" style="padding:0 0 8px 0">👥 Characters</div>', unsafe_allow_html=True)
        if st.session_state.characters:
            for char in st.session_state.characters:
                with st.expander(char.get("name", "Unknown")):
                    st.caption(char.get("description", ""))
        else:
            st.caption("Characters appear here as they're introduced.")

        st.divider()

        # Actions — full-width stacked (easier to tap, clearer labels)
        can_undo = len(st.session_state.segments) > 1
        if st.button("↩  Undo Last Turn", use_container_width=True, disabled=not can_undo,
                     help="Remove the last AI turn and your prompt that triggered it."):
            popped = st.session_state.segments.pop()
            st.session_state.undo_stack.append(popped)
            if (st.session_state.segments
                    and st.session_state.segments[-1]["role"] == "user"):
                st.session_state.undo_stack.append(st.session_state.segments.pop())
            st.session_state.pending_choices = []
            st.rerun()

        md_content = export_to_markdown(
            st.session_state.title,
            st.session_state.genre,
            st.session_state.segments,
        )
        st.download_button(
            "↓  Export as Markdown",
            data=md_content,
            file_name=f"{safe_filename(st.session_state.title)}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download your story as a formatted .md file.",
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("＋  New Story", use_container_width=True):
            for k in ["story_started", "title", "genre", "hook", "segments",
                      "characters", "pending_choices", "undo_stack", "error_msg"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Main content ──────────────────────────────────────────────────────────

    # Page header — smaller than setup hero
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(226,232,240,0.4)">
      <h2 style="margin:0;font-size:1.5rem!important">📖 {st.session_state.title}</h2>
      <span class="ma-badge">{genre_badge(st.session_state.genre)}</span>
    </div>
    """, unsafe_allow_html=True)

    _show_and_clear_error()

    # Scrollable story history — input area stays fixed below, history scrolls above
    story_container = st.container(height=520, border=False)
    with story_container:
        for seg in st.session_state.segments:
            if seg["role"] == "assistant":
                st.markdown(f'<div class="story-block">{seg["content"]}</div>',
                            unsafe_allow_html=True)
            elif seg["content"] not in ("Continue the story.", ""):
                st.markdown(f'<div class="user-block">✏️ {seg["content"]}</div>',
                            unsafe_allow_html=True)

    # Streaming placeholder — lives outside the scroll container so it appears below
    stream_placeholder = st.empty()

    st.divider()

    # ── Choice mode ───────────────────────────────────────────────────────────
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
            risk, badge_cls = risk_labels[i]
            # Render badge HTML above button via markdown
            st.markdown(
                f'<span class="choice-badge {badge_cls}">{risk}</span>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"{letter}.  {choice_text}",
                key=f"choice_{i}",
                use_container_width=True,
            ):
                st.session_state.segments.append(
                    {"role": "user", "content": f"I choose: {choice_text}"}
                )
                st.session_state.pending_choices = []

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
                    updated = extract_characters(get_full_story_text(st.session_state.segments))
                    if updated:
                        st.session_state.characters = updated
                except Exception as e:
                    _handle_llm_error(e)

                st.rerun()

    # ── Normal input mode ─────────────────────────────────────────────────────
    else:
        user_input = st.text_area(
            "Steer the story",
            placeholder="Add a line or two to guide the next turn — or leave blank and let the AI decide…",
            height=88,
            label_visibility="visible",
            key="user_text_input",
        )

        # Choices (secondary) left · Continue (primary CTA) right — standard UX flow
        col_choices, col_continue = st.columns(2, gap="small")

        with col_choices:
            if st.button("🎭  Give Me Choices", use_container_width=True):
                if user_input.strip():
                    st.session_state.segments.append(
                        {"role": "user", "content": user_input.strip()}
                    )
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
                            st.session_state.segments.append(
                                {"role": "assistant", "content": story_part}
                            )
                            st.session_state.pending_choices = choices
                        else:
                            st.session_state.segments.append(
                                {"role": "assistant", "content": raw}
                            )

                        updated = extract_characters(get_full_story_text(st.session_state.segments))
                        if updated:
                            st.session_state.characters = updated
                    except Exception as e:
                        _handle_llm_error(e)

                st.rerun()

        with col_continue:
            if st.button("✍️  Continue", type="primary", use_container_width=True):
                if user_input.strip():
                    st.session_state.segments.append(
                        {"role": "user", "content": user_input.strip()}
                    )
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
                    updated = extract_characters(get_full_story_text(st.session_state.segments))
                    if updated:
                        st.session_state.characters = updated
                except Exception as e:
                    _handle_llm_error(e)

                st.rerun()
