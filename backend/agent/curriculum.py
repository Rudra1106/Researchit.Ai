"""
curriculum.py

Generates a personalised step-by-step learning curriculum for a topic.
Called once when the student enters Mentor Mode.
The curriculum is stored in the session and drives the rest of the conversation.
"""

import json
import re
from backend.llm_client import llm_complete

CURRICULUM_TIMEOUT = 25

_PROMPT = """You are a curriculum designer for an AI tutoring system.

The student wants to deeply understand a topic. Design a clear, sequential learning path.

Rules:
- 5-7 steps total (not more — keep it focused)
- Start with WHY the problem exists (motivation), not with the solution
- Each step must build on the previous one
- End with the full concept understood and applied
- For a beginner: start from intuition, introduce math gradually
- For advanced: can start with the formal definition sooner
- Make steps specific to the paper context if available

Return ONLY valid JSON (no markdown, no explanation):
{
  "topic": "formal name of what will be learned",
  "tagline": "one sentence: what the student will be able to do after this curriculum",
  "steps": [
    {
      "id": 1,
      "concept": "short name (3-6 words max)",
      "objective": "what the student understands after this step",
      "builds_on": []
    }
  ]
}"""


def generate_curriculum(topic: str, learner_profile: dict, paper_context: str = "") -> dict:
    """
    Generate a learning curriculum for the given topic.
    Returns a curriculum dict. Falls back gracefully on LLM failure.
    """
    level   = learner_profile.get("level", "beginner")
    ctx_str = f"\nPaper context (tailor curriculum to this):\n{paper_context[:600]}" if paper_context else ""

    user_msg = (
        f"Topic to learn: {topic}\n"
        f"Student level: {level}\n"
        f"{ctx_str}\n"
        "Generate the curriculum."
    )

    try:
        msgs = [
            {"role": "system", "content": _PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        raw    = llm_complete(msgs, temperature=0.2, timeout=CURRICULUM_TIMEOUT)
        raw    = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(raw)
        if "steps" not in parsed or not parsed["steps"]:
            raise ValueError("No steps returned")
        # Ensure IDs are sequential integers
        for i, step in enumerate(parsed["steps"]):
            step["id"] = i + 1
        return parsed
    except Exception as e:
        print(f"[curriculum] Generation failed ({e}). Using fallback.")
        return _fallback_curriculum(topic)


def _fallback_curriculum(topic: str) -> dict:
    return {
        "topic":   topic,
        "tagline": f"Build a solid understanding of {topic} from first principles.",
        "steps": [
            {"id": 1, "concept": f"Why {topic} was needed", "objective": "Understand the problem it solves", "builds_on": []},
            {"id": 2, "concept": f"Core intuition of {topic}", "objective": "Build mental model without math", "builds_on": [1]},
            {"id": 3, "concept": f"The math behind {topic}", "objective": "Understand the key formula", "builds_on": [2]},
            {"id": 4, "concept": f"Implementing {topic}", "objective": "Code it from scratch", "builds_on": [3]},
            {"id": 5, "concept": f"{topic} in practice", "objective": "See it working on real data", "builds_on": [4]},
        ],
    }
