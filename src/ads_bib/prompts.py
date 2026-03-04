"""Reusable prompt templates for LLM-based labeling and translation.

Import named constants from here instead of hardcoding prompts inline.
Add new domain-specific variants as needed — the notebook selects via import.
"""

# ---------------------------------------------------------------------------
# BERTopic topic labeling
# ---------------------------------------------------------------------------

BERTOPIC_LABELING_GENERIC = """\
You are an experienced researcher. You are labeling research topic clusters \
based on scientific abstracts.

Documents: [DOCUMENTS]
Keywords: [KEYWORDS]

Task:
- Generate EXACTLY ONE topic label.
- The label MUST contain between 4 and 7 words (inclusive).
- The label should read like a review article or textbook chapter title.

Guidelines:
- Use standard scientific terminology from the field.
- Be specific about the phenomenon or method.
- Avoid generic terms like "studies", "analysis", or "research".

Output format (single line):
topic: <label>

Do NOT write anything else (no explanations, no additional sentences, \
no quotes, no bullet points)."""

BERTOPIC_LABELING_PHYSICS = """\
You are an experienced researcher in gravitational physics, astrophysics, \
and cosmology. You are labeling research topic clusters based on scientific \
abstracts.

Documents: [DOCUMENTS]
Keywords: [KEYWORDS]

Task:
- Generate EXACTLY ONE topic label.
- The label MUST contain between 4 and 7 words (inclusive).
- The label should read like a review article or textbook chapter title.

Guidelines:
- Use standard scientific terminology from the field.
- Be specific about the physical phenomenon or method.
- Avoid generic terms like "studies", "analysis", or "research".

Output format (single line):
topic: <label>

Do NOT write anything else (no explanations, no additional sentences, \
no quotes, no bullet points)."""


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM = (
    "You are a highly accurate translator specializing in scientific and "
    "technical texts. Only translate the text. Do not comment or provide "
    "additional information."
)
