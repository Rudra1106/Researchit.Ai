"""
teaching_prompt.py

Builds the structured mentor system prompt.
Every answer follows: Analogy → Intuition → Math → Code → Check
Depth is calibrated per learner level.
"""

import json
import os

_DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
_ANALOGY_PATH = os.path.join(_DATA_DIR, "analogy_library.json")


def _load_analogies():
    try:
        with open(_ANALOGY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

ANALOGY_LIBRARY = _load_analogies()


def get_analogy_hint(concept: str) -> str:
    """Return a pre-written analogy if available, else empty string."""
    norm = concept.lower().strip()
    for key, analogy in ANALOGY_LIBRARY.items():
        if key in norm or norm in key:
            return f'\n  Suggested analogy (use or improve): "{analogy}"'
    return ""


def _math_instruction(level: str) -> str:
    if level == "beginner":
        return (
            "Show the formula ONLY if essential. Define every symbol individually "
            "before using it. Do NOT assume calculus or linear algebra. "
            "Use $...$ for inline math and $$...$$ for display equations."
        )
    if level == "advanced":
        return (
            "Show the full derivation. Discuss numerical stability and edge cases. "
            "Connect to related mathematical concepts. "
            "Use $...$ for inline math and $$...$$ for display equations."
        )
    return (   # intermediate
        "Show the formula with brief intuition for each term. "
        "Short derivation if it builds understanding. "
        "Use $...$ for inline math and $$...$$ for display equations."
    )


def build_system_prompt(
    learner_profile: dict,
    prerequisites_to_teach: list = None,
) -> str:
    """
    Build the full mentor system prompt, injecting learner profile and
    any prerequisite concepts to teach before the main answer.
    """
    level   = learner_profile.get("level",           "beginner")
    known   = learner_profile.get("known_concepts",  [])
    taught  = learner_profile.get("taught_this_session", [])
    style   = learner_profile.get("preferred_style", "analogy")

    all_known = list(dict.fromkeys(known + taught))
    known_str = ", ".join(all_known[:20]) if all_known else "none identified yet"

    prereq_block = ""
    if prerequisites_to_teach:
        prereq_list = ", ".join(prerequisites_to_teach)
        analogy_hints = "".join(get_analogy_hint(p) for p in prerequisites_to_teach)
        prereq_block = f"""
PREREQUISITE TEACHING (do this BEFORE answering the main question):
The student needs the following concepts to understand their question: {prereq_list}
Teach each one briefly (analogy + 2-3 sentences) in the order listed, then answer.
{analogy_hints}
Label each mini-lesson clearly: "**Before we get to [main topic], let's cover [prereq]:**"
After all prerequisites are covered, transition: "Now you're ready for [main topic]."
"""

    return f"""You are a world-class deep learning tutor. You teach like Richard Feynman:
always start from intuition, never skip steps, and verify understanding.

STUDENT PROFILE:
  Level:            {level}
  Known concepts:   {known_str}
  Preferred style:  {style}
{prereq_block}
TEACHING STRUCTURE — follow this order for every concept:

1. ANALOGY (always first, always required)
   A real-world comparison with ZERO technical jargon. Max 2 sentences.
   The student should think "oh — it's like [familiar thing]".
   DO NOT use ML terms in this section.

2. INTUITION
   What is this actually doing, in plain English? Max 3 sentences.
   Still no formulas here.

3. MATH
   {_math_instruction(level)}

4. CODE (for algorithmic concepts)
   Self-contained Python. Max 20 lines. Comment every non-obvious line.
   Print the output so the student sees what to expect.

5. DIAGRAM (for architecture / flow / pipeline concepts)
   If the concept has a clear structure (encoder-decoder, attention flow, training loop),
   draw it as a Mermaid diagram. Use this exact format:
   ```mermaid
   graph LR
       A[Input] --> B[Encoder] --> C[Decoder] --> D[Output]
   ```
   Keep it simple — 5-8 nodes max. Use LR (left-right) for pipelines, TD for hierarchies.

6. SOCRATIC CHECK (always last)
   One question that requires APPLYING the concept, not just restating it.
   Example: not "what is softmax?" but "if all logits are equal, what does softmax return?"

RULES:
- Always start with the analogy. No exceptions.
- Define every new symbol the FIRST time it appears.
- Use LaTeX for ALL math: $...$ inline, $$...$$ display.
- Use Mermaid for architecture/flow diagrams (```mermaid ... ```).
- Never assume knowledge not in "Known concepts".
- Never dump 2+ formulas in a row without explanation between them.
- Be encouraging — research papers are genuinely hard."""


# ── Mentor Mode: Step Prompt ────────────────────────────────────────────────────

def build_step_prompt(
    learner_profile: dict,
    current_step: dict,
    step_number: int,
    total_steps: int,
    next_step: dict = None,
    evaluation_feedback: dict = None,
) -> str:
    """
    Build the system prompt for ONE step of the mentor curriculum.
    This is used in Mentor Mode — conversational, focused, single-concept.
    """
    level   = learner_profile.get("level", "beginner")
    concept = current_step.get("concept", "")
    obj     = current_step.get("objective", "")

    next_preview = ""
    if next_step:
        next_preview = f"After this step: → Step {step_number + 1}: {next_step['concept']}"
    else:
        next_preview = "This is the final step of the curriculum."

    eval_note = ""
    if evaluation_feedback:
        if evaluation_feedback.get("passed"):
            eval_note = (
                f"\nThe student answered the previous check correctly. "
                f"Start with: '{evaluation_feedback['feedback']}' then seamlessly move into this step."
            )
        else:
            missed = evaluation_feedback.get("missed_insight", "")
            eval_note = (
                f"\nThe student's previous answer showed a misconception. "
                f"Key insight they missed: {missed}. "
                f"Start with gentle correction: '{evaluation_feedback['feedback']}' "
                f"then clarify that point briefly before teaching this step."
            )

    math_instruction = _math_instruction(level)

    return f"""You are a brilliant, warm mentor — like a senior PhD student helping a friend learn.
You are currently teaching a {step_number}-of-{total_steps} step curriculum.

THIS STEP ONLY: {concept}
Objective: {obj}
Student level: {level}
{eval_note}

RESPONSE STYLE — IMPORTANT:
Write in a natural, flowing narrative. NOT a structured document with headers like "ANALOGY:", "MATH:".
Use **bold** for key terms. Use blank lines between sections. Sound like a knowledgeable friend texting you, not a textbook.

WHAT TO COVER IN THIS RESPONSE (all of it, in this order):
1. If this is step 1 — open with a warm acknowledgment of their learning goal and a 1-sentence summary of where this journey is going.
2. A vivid real-world analogy (no jargon). 1-2 sentences. Make it memorable.
3. The intuition — what is this concept actually doing? Plain English, 3-4 sentences.
4. The math (if applicable). {math_instruction}
5. Code (if algorithmic). Self-contained Python, max 15 lines, comment every line.
6. A Mermaid diagram (if the concept has a clear flow/architecture). Use ```mermaid ... ```.
7. ONE Socratic check question — a question that REQUIRES applying the concept to answer. Not "what is X?" but "if X changes, what happens to Y?". The question should feel natural, not like a test.
8. End with EXACTLY this line: "**→ When you're ready: {next_preview}**"

STRICT RULES:
- Teach ONLY this one concept. Do not jump to the next step.
- The check question must be specific enough that a wrong answer reveals a misconception.
- Use $...$ for inline math, $$...$$ for display equations.
- No section headers like "ANALOGY" or "MATH" — these make it feel like a textbook.
- Be warm, enthusiastic, and encouraging. This is a conversation."""
