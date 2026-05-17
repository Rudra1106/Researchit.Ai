"""
intent_detector.py

Detects whether the student wants to LEARN a topic from scratch (Mentor Mode)
or is asking a factual QUESTION about the paper (Q&A Mode).
"""

import json
import re
from backend.llm_client import llm_complete

INTENT_TIMEOUT = 8

# Fast keyword check — avoids LLM call for clear factual questions
_LEARN_SIGNALS = [
    "i want to learn", "teach me", "help me learn", "guide me through",
    "from scratch", "build my understanding", "step by step",
    "explain everything", "start from basics", "i'm a beginner",
    "help me understand", "walk me through", "build up to",
    "i want to understand", "take me through", "learn about",
]

_PROMPT = """You classify student messages for an AI tutoring system.

LEARN: student wants a guided, step-by-step journey through a topic.
QA: student wants a specific answer to a specific question.

Examples:
  "I want to learn transformer architecture from scratch" → LEARN, topic: "transformer architecture"
  "What BLEU score did the model achieve?" → QA, topic: ""
  "Help me understand how attention works" → LEARN, topic: "attention mechanism"
  "What equation is on page 4?" → QA, topic: ""

Return ONLY valid JSON — no markdown:
{"intent": "LEARN" | "QA", "topic": "extracted topic or empty string"}"""


def detect_intent(message: str) -> dict:
    """
    Returns {"intent": "LEARN"|"QA", "topic": "..."}
    Defaults to QA on failure.
    """
    default = {"intent": "QA", "topic": ""}
    msg_lower = message.lower()

    # Fast path — no LLM call if no learning signal found
    has_signal = any(kw in msg_lower for kw in _LEARN_SIGNALS)
    if not has_signal:
        return default

    try:
        msgs = [
            {"role": "system", "content": _PROMPT},
            {"role": "user",   "content": f"Student message: {message}"},
        ]
        raw = llm_complete(msgs, temperature=0.1, timeout=INTENT_TIMEOUT)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(raw)
        return {
            "intent": parsed.get("intent", "QA"),
            "topic":  parsed.get("topic", "").strip(),
        }
    except Exception:
        # If LLM fails but we saw a learning signal, assume LEARN
        return {"intent": "LEARN", "topic": message.strip()}
