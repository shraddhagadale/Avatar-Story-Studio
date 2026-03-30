GENRES = ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Comedy"]

GENRE_RULES = {
    "Fantasy": (
        "Magic systems must remain internally consistent — never introduce new powers without prior hints. "
        "Use epic, lyrical prose. Honor the world's geography, politics, and lore once established. "
        "Lean into archetypes but give characters personal depth beyond their role."
    ),
    "Sci-Fi": (
        "Technology must obey the rules of the established universe — don't invent convenient solutions. "
        "Ground descriptions in scientific plausibility. Favor ideas-driven tension over action for its own sake. "
        "Maintain the established tech level: near-future / far-future / alternate history — don't mix."
    ),
    "Mystery": (
        "Every clue planted must be retrievable by the reader — fair-play rules apply. "
        "Red herrings are allowed but must be logically consistent. "
        "Do NOT reveal the culprit or solution prematurely. Build dread and suspicion gradually."
    ),
    "Romance": (
        "Emotional authenticity matters more than plot mechanics. "
        "Build tension through subtext, near-misses, and internal conflict. "
        "Characters must earn their connection — no instant resolutions. "
        "Track the relationship arc: meeting → tension → vulnerability → bond."
    ),
    "Horror": (
        "Never over-explain the threat — ambiguity amplifies fear. "
        "Use sensory detail to build dread: sounds, smells, textures before visuals. "
        "Pace carefully: buildup → release → deeper dread. "
        "Psychological unease outlasts gore. Prioritize atmosphere over shock."
    ),
    "Comedy": (
        "Humor emerges from character and situation, not just jokes. "
        "Use callbacks to earlier story beats for payoff. "
        "Escalate absurdity gradually — don't peak too early. "
        "Maintain comedic timing: short punchy sentences for punchlines."
    ),
}

# Groq model
MODEL = "llama-3.3-70b-versatile"

# Max tokens for AI output per response
MAX_TOKENS_OUTPUT = 1024

# Token budget for history (conservative — free tier is 6K TPM total)
# ~4 chars per token estimate; keep total request under ~4000 tokens
TOKEN_WARN_THRESHOLD = 3000    # show warning in UI
TOKEN_TRIM_THRESHOLD = 4500    # start trimming oldest segments
