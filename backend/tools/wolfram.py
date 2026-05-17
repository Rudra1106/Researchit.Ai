"""
wolfram.py

Responsibility: evaluate mathematical expressions and get
plain-English answers using Wolfram Alpha.

We use two Wolfram endpoints:
  1. Short Answers API — one-line plain English result
     Best for: "what is the derivative of softmax?",
               "evaluate sum of 1/n for n=1 to 10"

  2. Simple API — returns a PNG image of the full computation
     Best for: equations that need step-by-step working shown
     (We don't use this in v1 — the frontend isn't ready for images yet)

Wolfram Alpha is best used for:
  ✓ Evaluating specific formulas ("softmax([1,2,3])")
  ✓ Derivatives and integrals
  ✓ Matrix operations
  ✓ "What does this formula equal when x=0.5?"

Not great for:
  ✗ Conceptual questions ("why is ReLU better than sigmoid?")
  ✗ Open-ended questions (use the LLM for those)

Detection heuristic:
  We check if the question contains math signals before calling Wolfram.
  This avoids wasting API quota on purely conceptual questions.
"""

import os
import re
import requests

WOLFRAM_SHORT_URL = "https://api.wolframalpha.com/v1/result"
REQUEST_TIMEOUT   = 15   # Wolfram can be slow on complex queries

# Patterns that suggest the question has a mathematical component
MATH_SIGNALS = [
    r"\bderivative\b", r"\bintegral\b", r"\bgradient\b",
    r"\bevaluate\b",   r"\bcompute\b",  r"\bcalculate\b",
    r"\bformula\b",    r"\bequation\b", r"\bmatrix\b",
    r"\bsoftmax\b",    r"\bsigmoid\b",  r"\brelu\b",
    r"\bexp\(",        r"\blog\(",      r"\bsum\b",
    r"\$\$",           r"\\frac",       r"\^",
    r"\d+\s*[+\-\*/]\s*\d+",           # arithmetic like "3 + 4"
    r"=[^=]",                           # equations
]


def is_math_question(question):
    """
    Return True if the question likely involves mathematics.
    Used to decide whether to call Wolfram Alpha.
    """
    q = question.lower()
    return any(re.search(pattern, q) for pattern in MATH_SIGNALS)


def _get_app_id():
    """
    Load the Wolfram App ID from environment.
    Returns None if not configured — callers treat this as "Wolfram unavailable".
    """
    app_id = os.environ.get("WOLFRAM_APP_ID", "").strip()
    return app_id if app_id else None


def query_wolfram(question):
    """
    Send a question to Wolfram Alpha's Short Answers API.

    This gives us a plain-English one-liner like:
      "The derivative of softmax(x_i) is softmax(x_i)(1 - softmax(x_i))"

    Args:
        question: a mathematical question string

    Returns:
        dict with keys: source, query, result, interpretation
        or None if Wolfram is unavailable, the query fails, or returns no answer
    """
    app_id = _get_app_id()
    if not app_id:
        return None   # Wolfram not configured — skip silently

    # Clean up the question for Wolfram
    # Remove markdown-style math fences if present
    cleaned = re.sub(r"\$+", "", question).strip()

    try:
        response = requests.get(
            WOLFRAM_SHORT_URL,
            params={
                "appid":  app_id,
                "i":      cleaned,
                "timeout": 10,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return None

    # 501 = Wolfram couldn't understand the query — common, not an error
    if response.status_code == 501:
        return None

    if response.status_code != 200:
        return None

    result_text = response.text.strip()
    if not result_text:
        return None

    return {
        "source":         "wolfram_alpha",
        "query":          cleaned,
        "result":         result_text,
        "interpretation": f"Wolfram Alpha computes: {result_text}",
    }


def query_wolfram_if_math(question):
    """
    Convenience wrapper: only calls Wolfram if the question looks mathematical.
    This is the function the enrichment pipeline calls — it handles the
    math detection so callers don't have to.

    Returns:
        dict (Wolfram result) or None
    """
    if not is_math_question(question):
        return None
    return query_wolfram(question)