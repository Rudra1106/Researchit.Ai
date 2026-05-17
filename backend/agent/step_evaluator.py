"""
step_evaluator.py

Evaluates whether the student's answer to a Socratic check demonstrates understanding.
Generous by design — partial understanding passes. Gatekeeping kills motivation.
"""

import json
import re
from backend.llm_client import llm_complete

EVAL_TIMEOUT = 10

_PROMPT = """You evaluate a student's answer to a Socratic check question.

Be GENEROUS. Partial understanding passes. The goal is learning, not gatekeeping.

PASS if: student shows they grasped the core idea, even imperfectly.
FAIL if: student has a clear misconception OR just said "I don't know" / gave no answer.

Return ONLY valid JSON:
{
  "passed": true | false,
  "feedback": "1-2 encouraging sentences — affirm what they got right, gently correct if needed",
  "missed_insight": "if failed: the one key thing they missed. If passed: empty string."
}"""


def evaluate_check_answer(
    check_question: str,
    student_answer: str,
    concept: str,
) -> dict:
    """
    Evaluate a student's Socratic check answer.
    Returns {passed, feedback, missed_insight}.
    Defaults to passed=True on any failure (never block learning on a bug).
    """
    default = {
        "passed":         True,
        "feedback":       "Great thinking! Let's keep going.",
        "missed_insight": "",
    }

    # Very short answer → treat as "skip, continue anyway"
    answer_stripped = student_answer.strip()
    if len(answer_stripped) < 10:
        return {
            "passed":         True,
            "feedback":       "No worries — let's keep moving!",
            "missed_insight": "",
        }

    try:
        user_msg = (
            f"Concept being taught: {concept}\n"
            f"Check question asked: {check_question}\n"
            f"Student's answer: {student_answer}"
        )
        msgs = [
            {"role": "system", "content": _PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        raw    = llm_complete(msgs, temperature=0.1, timeout=EVAL_TIMEOUT)
        raw    = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(raw)
        return {
            "passed":         bool(parsed.get("passed", True)),
            "feedback":       parsed.get("feedback",       "Good effort! Let's continue."),
            "missed_insight": parsed.get("missed_insight", ""),
        }
    except Exception:
        return default


def extract_check_question(response_text: str) -> str:
    """
    Extract the Socratic check question from the end of a mentor response.
    Looks for the last sentence ending with '?'.
    """
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if sentence.endswith("?") and len(sentence) > 15:
            # Skip meta-questions like "Shall we continue?"
            skip = ["shall we", "ready to", "want to", "would you like", "make sense?"]
            if not any(s in sentence.lower() for s in skip):
                return sentence
    return ""
