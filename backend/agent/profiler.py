"""
profiler.py

Infers a learner's level from their first message.
Runs once per session — result is stored and used for all subsequent responses.
"""

import json
import re

from backend.llm_client import llm_complete

PROFILE_TIMEOUT = 12   # seconds — fail fast, default to beginner on timeout

_PROMPT = """You are a learner-level classifier for an AI tutoring system.

Analyse the student's first question and infer their background in machine learning.

Signals:
- Vocabulary: do they use precise ML terms, or everyday language?
- Specificity: broad ("what is this paper?") vs targeted ("why scale by 1/√d_k?")
- Assumed knowledge: what does the question presuppose?

Return ONLY valid JSON — no markdown, no preamble:
{
  "level": "beginner" | "intermediate" | "advanced",
  "known_concepts": ["concepts the student clearly knows"],
  "preferred_style": "analogy" | "math" | "code",
  "reasoning": "one sentence"
}

Level definitions:
  beginner:     everyday language, broad questions, no ML jargon assumed
  intermediate: knows gradient/layer/embedding, asks how things work
  advanced:     asks why, cites papers, discusses tradeoffs"""


def profile_learner(first_message: str) -> dict:
    """
    Infer learner level from first message.
    Returns a profile dict. Fails gracefully to beginner defaults.
    """
    default = {
        "level":           "beginner",
        "known_concepts":  [],
        "preferred_style": "analogy",
        "profiled":        True,
    }
    try:
        messages = [
            {"role": "system", "content": _PROMPT},
            {"role": "user",   "content": f"Student's question: {first_message}"},
        ]
        raw = llm_complete(messages, temperature=0.1, timeout=PROFILE_TIMEOUT)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(raw)
        return {
            "level":           parsed.get("level",           "beginner"),
            "known_concepts":  parsed.get("known_concepts",  []),
            "preferred_style": parsed.get("preferred_style", "analogy"),
            "profiled":        True,
        }
    except Exception:
        return default
