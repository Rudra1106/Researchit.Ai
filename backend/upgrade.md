# AI Mentor Feature — Detailed Plan
### Paper Tutor: Phase 5 & Beyond

> **Goal:** Transform Paper Tutor from a Q&A bot into a Socratic AI mentor that builds genuine intuition, teaches prerequisites, and guides deep learning students through complex research papers — the way a great professor would.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Feature Overview](#2-feature-overview)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase A — Learner Profiling](#phase-a--learner-profiling)
5. [Phase B — Prerequisite Engine](#phase-b--prerequisite-engine)
6. [Phase C — Mentor Teaching Mode](#phase-c--mentor-teaching-mode)
7. [Phase D — Web & Math Enrichment](#phase-d--web--math-enrichment)
8. [Phase E — Visual Intuition Layer](#phase-e--visual-intuition-layer)
9. [Phase F — Anti-Hallucination Stack](#phase-f--anti-hallucination-stack)
10. [Phase G — Adaptive Reading Path](#phase-g--adaptive-reading-path)
11. [Prompt Engineering Strategy](#11-prompt-engineering-strategy)
12. [Managing Large Data & Context Windows](#12-managing-large-data--context-windows)
13. [Why Not Recursive Language Models?](#13-why-not-recursive-language-models)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [What Success Looks Like](#15-what-success-looks-like)

---

## 1. The Problem

Your current agent already does something impressive: it retrieves relevant chunks from a paper and explains them clearly. A student asking *"what does this paper talk about?"* gets a coherent, well-structured answer.

But a beginner learning deep learning doesn't just need an answer. They need a **learning experience**. Here's what's currently missing:

| Gap | Current State | Needed |
|-----|--------------|--------|
| No learner awareness | Every user gets the same depth | Responses adapt to the student's level |
| No prerequisite checking | Explains attention without checking if student knows dot products | Detects gaps and teaches them first |
| No pedagogical structure | Dumps information | Analogy → intuition → math → code → check |
| No visual support | Text-only responses | Interactive diagrams and visualisations |
| No learning path | Students read paper linearly | Personalised reading order based on concept dependencies |
| Potential hallucination | LLM states math results confidently | Multi-layer grounding: RAG + Wolfram + citation tags |

The analogy: imagine hiring a tutor who is extremely knowledgeable but has no idea who you are, always lectures at PhD level, never checks if you understood, and occasionally invents facts. That's the current system. The new feature turns it into a tutor who first asks "what do you already know?", starts from your level, builds up step by step, and checks their own answers before telling you.

---

## 2. Feature Overview

The AI Mentor Feature is built on three pillars:

**Pillar 1 — Know the student.** Before teaching anything, profile the learner's level and map what they know against what the paper requires.

**Pillar 2 — Teach in layers.** Every explanation follows a structured pedagogical flow: analogy → intuition → math → code → Socratic question. Depth adapts to the learner.

**Pillar 3 — Trust but verify.** Every fact is grounded in the paper, every math result is verified by Wolfram Alpha, every claim carries a citation. The agent signals uncertainty explicitly instead of hallucinating confidently.

---

## 3. Architecture Overview

```
PDF Upload
    ↓
[Existing Pipeline: PyMuPDF → ChromaDB → BM25 + Vector Hybrid Search → Knowledge Graph]
    ↓
NEW: Learner Profiling (Phase A)
    ↓
Student Question
    ↓
┌─────────────────────────────────────────────────────┐
│              Mentor Orchestrator                     │
│                                                     │
│  1. Prerequisite Check (Phase B)                   │
│     └─ Graph BFS → missing concepts → teach first  │
│                                                     │
│  2. Paper Retrieval (existing hybrid RAG)          │
│     └─ BM25 + vector + multi-query + parent-child  │
│                                                     │
│  3. Web + Math Enrichment (Phase D)                │
│     └─ Wikipedia + Wolfram + YouTube (parallel)    │
│                                                     │
│  4. Structured Teaching Prompt (Phase C)           │
│     └─ Analogy → intuition → math → code          │
│                                                     │
│  5. Confusion Detection → Adapt (Phase C)          │
└─────────────────────────────────────────────────────┘
    ↓
Response (with citations, LaTeX, runnable code)
    ↓
Visual Layer (Phase E) — diagrams, animations, plots
    ↓
Session Memory Updated — what did student learn today?
```

The key design principle: **the mentor orchestrator is a chain of small, focused LLM calls**, not one giant prompt. Each call is grounded, short, and verifiable. This is what prevents confident hallucination.

---

## Phase A — Learner Profiling

### What it is

Before the agent answers a single question, it runs a lightweight profiler that infers the student's background from their message and conversation history. It doesn't ask "are you a beginner?" — it detects it from how the student phrases questions.

### Why it matters

Right now, a PhD student and a first-year undergrad both get the same response. The PhD is bored; the undergrad is lost. The profiler is the foundation everything else builds on.

### How it works

**Step 1 — Initial profile inference.** On the student's first message, a small, fast LLM call (can use `llama3.2:3b` for speed) analyses the vocabulary, question type, and phrasing:

- "What does this paper say?" → probably beginner; broad question, no specific term
- "Can you explain the QKV decomposition in the attention mechanism?" → intermediate; knows the terms, wants deeper understanding
- "Why is the attention complexity O(n²) and how does linear attention address this?" → advanced; knows the tradeoffs, wants nuance

**Step 2 — Concept map initialisation.** The profiler maps the student's known concepts onto the knowledge graph nodes. If the paper is "Attention Is All You Need" and the student mentions they know "matrix multiplication" and "neural networks" but not "softmax" or "positional encoding", those gaps are stored.

**Step 3 — Style preference detection.** Does the student ask for "intuition"? They want analogies first. Do they immediately jump to "how does the math work"? They want equations. Do they say "can you show me in code"? They want a Python walkthrough. The profiler sets a `preferred_style` flag that influences every subsequent response.

### Session memory structure

```python
learner_profile = {
    "level": "beginner",              # beginner | intermediate | advanced
    "known_concepts": ["matrix multiplication", "neural networks", "gradient descent"],
    "gaps": ["softmax", "attention", "positional encoding", "layer normalisation"],
    "preferred_style": "analogy",     # analogy | math | code
    "taught_this_session": [],        # grows as session progresses
    "confusion_events": []            # tracks which concepts triggered re-explanation
}
```

### Prompt design (Anthropic best practices: role + structured XML output)

```xml
<system>
You are a learner profiler for an AI tutoring system. From the student's message
and any conversation history, infer their background in machine learning and deep learning.

Use vocabulary, question specificity, and concept familiarity as signals.

Return ONLY valid XML — no prose, no preamble:
<profile>
  <level>beginner|intermediate|advanced</level>
  <known_concepts>comma-separated list of concepts the student seems familiar with</known_concepts>
  <gaps>comma-separated list of likely knowledge gaps</gaps>
  <preferred_style>analogy|math|code</preferred_style>
  <reasoning>one sentence explaining your inference</reasoning>
</profile>
</system>

<student_message>{first_message}</student_message>
```

**Why this prompt works:** The XML output contract forces the model to be precise and machine-parseable. The `<reasoning>` field acts as a chain-of-thought that improves classification accuracy. Restricting output to XML removes the risk of verbose preamble.

---

## Phase B — Prerequisite Engine

### What it is

The most important new feature. Before explaining concept X, the engine traverses the knowledge graph backwards to find what the student must already understand. If they don't know a prerequisite, the mentor teaches it first and then returns to the original question.

### Why it matters

This is the single biggest failure mode in the current system. If a student asks "explain multi-head attention" and doesn't know what a dot product is, the current agent answers anyway. The student reads words they don't understand, gives up, and concludes they're "not smart enough for AI". A real tutor would never do this.

### How it works

**Step 1 — Concept extraction from the question.** A fast LLM call identifies the core concepts being asked about.

```
"Explain multi-head attention" → ["multi-head attention"]
"What is the purpose of dropout in transformers?" → ["dropout", "transformer", "overfitting"]
```

**Step 2 — Prerequisite graph traversal.** The knowledge graph (already built by your NetworkX implementation) is extended with a new edge type: `REQUIRES`. These edges are added either:
- Automatically during paper processing (if the paper says "building on [concept]..." the extractor adds a REQUIRES edge)
- From a static DL prerequisite map you define (e.g. `multi-head attention REQUIRES attention`, `attention REQUIRES dot product`, `dot product REQUIRES vector`)

```python
# Prerequisite edges (a curated DL prerequisite map)
DL_PREREQUISITES = {
    "multi-head attention": ["scaled dot-product attention", "linear projection"],
    "scaled dot-product attention": ["dot product", "softmax", "query-key-value"],
    "softmax": ["exponential function", "normalisation"],
    "backpropagation": ["chain rule", "partial derivative", "loss function"],
    "transformer": ["attention mechanism", "residual connections", "layer normalisation"],
    # ... extensible
}
```

**Step 3 — Gap detection.** The engine compares the required prerequisites against the session's `known_concepts` list. Any prerequisite not in `known_concepts` and not in `taught_this_session` becomes a teaching target.

**Step 4 — Queue and teach in order.** Missing prerequisites are sorted topologically (most fundamental first) and taught before the original concept. After each prerequisite is taught, it's added to `taught_this_session` and the engine checks again before proceeding.

**Step 5 — Return to original question.** Once all prerequisites are covered, the mentor answers the original question — now with the student actually ready for it.

### Example flow

```
Student: "Explain multi-head attention"
Profiler says: beginner, knows neural networks and matrix multiplication, gap: attention

Prerequisite engine:
  multi-head attention REQUIRES: [scaled dot-product attention] ← not known
  scaled dot-product attention REQUIRES: [dot product, softmax] ← dot product known, softmax not known
  softmax REQUIRES: [exponential, normalisation] ← known

Queue: [softmax, scaled dot-product attention, multi-head attention]

Mentor says:
"Before we get to multi-head attention, let me make sure two building blocks are clear.
First — softmax. Here's what it does..."
[teaches softmax]
"Great. Now, with softmax in hand, let's look at how attention actually works..."
[teaches scaled dot-product attention]
"Now you're ready for multi-head attention. Here's why 'multi' matters..."
[teaches multi-head attention — the original question]
```

### Prompt design (Anthropic best practices: chain-of-thought reasoning)

```xml
<system>
You are a prerequisite checker for an AI tutoring system.

Given a concept the student wants to learn and their current knowledge profile,
identify which prerequisite concepts must be taught first.

Think step by step inside <thinking> tags before answering:
<thinking>
  What does understanding [concept] fundamentally require?
  Which of those prerequisites are NOT in the student's known_concepts?
  Order them from most fundamental to most advanced.
</thinking>

Then return:
<prerequisites>
  <concept order="1">most fundamental missing concept</concept>
  <concept order="2">next concept</concept>
  ...
</prerequisites>
</system>

<target_concept>{concept}</target_concept>
<known_concepts>{learner_profile.known_concepts}</known_concepts>
<taught_this_session>{learner_profile.taught_this_session}</taught_this_session>
```

---

## Phase C — Mentor Teaching Mode

### What it is

The core teaching engine. Every explanation follows a structured five-step pedagogical flow that mirrors how the best human tutors teach: start with what the student already knows, build an analogy, develop intuition, introduce the math, show the code, then verify understanding.

### The teaching flow

Every response from the mentor (for a substantive concept) follows this structure:

```
1. ANALOGY       — one real-world parallel, zero jargon
2. INTUITION     — what is it actually doing, in plain English?
3. MATH          — depth calibrated to learner level
4. CODE          — minimal working Python, runnable
5. CHECK         — one Socratic question to verify understanding
```

**Analogy examples for DL concepts:**

| Concept | Analogy used by the mentor |
|---------|---------------------------|
| Softmax | "Like converting raw scores on a test into percentages that sum to 100%" |
| Attention | "Like a student highlighting the most relevant parts of a textbook before answering a question" |
| Dropout | "Like a sports team that practices without random players — so every player learns the full strategy, not just their part" |
| Residual connections | "Like GPS that always knows your starting point — even if you take a wrong turn, you can always get back" |
| Layer normalisation | "Like adjusting the volume on each instrument in an orchestra so no single one drowns out the others" |

### Depth calibration by learner level

The same concept is explained differently based on the learner's profile:

**Softmax — beginner:**
```
Softmax takes a list of numbers and converts them into probabilities that add up to 1.
If your scores are [2, 1, 0.1], softmax makes them [0.66, 0.24, 0.10].
The biggest number gets the most probability. Think of it like voting — the highest score wins the most votes.
```

**Softmax — intermediate:**
```
Softmax applies exp() to each logit and divides by the sum:
σ(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

The exp() does two things: makes all values positive, and amplifies differences between logits.
This is why a small difference in raw scores can become a large difference in probabilities.
```

**Softmax — advanced:**
```
Softmax is the gradient of the log-sum-exp function:
∂/∂zᵢ log(Σⱼ exp(zⱼ)) = exp(zᵢ) / Σⱼ exp(zⱼ) = σ(zᵢ)

This is why it pairs naturally with cross-entropy loss — the Jacobian of softmax combined
with cross-entropy has a beautifully clean gradient: ŷ - y.
Numerically: always subtract max(z) before exponentiating to prevent overflow.
```

### Confusion detection

After each explanation, a lightweight classifier checks:
- Was the answer long (>400 words)?
- Did it use more than 3 undefined technical terms?
- Did the student's follow-up suggest confusion ("I don't get the part where...", "wait, what is...", "can you re-explain...")?

If confusion is detected, the mentor proactively asks: *"Which part wasn't clear — the analogy, the math, or the code example?"* and re-explains that specific piece at a lower abstraction level.

### Prompt design (Anthropic best practices: XML structure + positive/negative examples + persona)

```xml
<system>
You are a patient, enthusiastic deep learning tutor. You teach like Richard Feynman:
always start with the clearest possible mental model, never skip steps, and check
that the student actually understands before moving on.

For every explanation, follow this exact structure:

<teach_flow>
  <analogy>One real-world parallel with zero technical jargon. Max 2 sentences.</analogy>
  <intuition>What is this actually doing, in plain language? Max 3 sentences.</intuition>
  <math>
    If level=beginner: show the formula with every symbol explained, step by step.
    If level=intermediate: show the derivation and discuss the intuition behind each term.
    If level=advanced: discuss edge cases, numerical stability, and connections to related concepts.
  </math>
  <code>Minimal working Python. Self-contained. Add 1-line comments on non-obvious lines.</code>
  <check>One Socratic question that reveals whether the student truly understood.</check>
</teach_flow>

Rules:
- DO: Start every explanation with the analogy, no exceptions.
- DO: Define every symbol the first time it appears in a formula.
- DO: Keep code snippets under 20 lines.
- DO NOT: Dump multiple formulas without explanation between them.
- DO NOT: Assume knowledge not listed in <known_concepts>.
- DO NOT: Use a technical term in the analogy section.

Learner profile: <level>{level}</level>, <style>{preferred_style}</style>
Known concepts: <known_concepts>{known_concepts}</known_concepts>
</system>

<paper_chunks>{retrieved_chunks}</paper_chunks>
<student_question>{question}</student_question>
```

---

## Phase D — Web & Math Enrichment

### What it is

An extension of your existing `enrichment.py`, expanded to pull learning-specific resources: visual explanations (3Blue1Brown, Khan Academy), ArXiv related papers, and concrete numerical verification via Wolfram Alpha.

### Why it matters

Research papers are dense and self-referential. They don't explain background concepts — they cite them. A student reading "Attention Is All You Need" who doesn't know what a seq2seq model is has nowhere to turn inside the paper. The enrichment layer pulls in exactly what a professor would pull from a whiteboard or a recommended textbook.

### Sources and what each provides

| Source | What it provides | When to use |
|--------|-----------------|-------------|
| Wikipedia | Background concepts, definitions, history | Always — for every new concept |
| Wolfram Alpha | Numerical evaluation of formulas, symbolic math | For every mathematical claim |
| ArXiv | Related papers, original concept sources | When the paper cites a technique |
| YouTube transcripts | 3Blue1Brown, Andrej Karpathy walkthroughs | For visual/intuition concepts |
| Khan Academy API | Foundational math (calculus, linear algebra) | When profiler detects beginner-level math gaps |

### Parallel fetching (already in your enrichment.py, extended)

```python
from concurrent.futures import ThreadPoolExecutor

def enrich_concept(concept: str, learner_level: str) -> dict:
    sources = [
        fetch_wikipedia(concept),
        fetch_wolfram(concept),          # math verification
        fetch_arxiv_related(concept),     # NEW
        fetch_youtube_transcript(concept) # NEW — searches for 3b1b videos
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda f: f(), sources))

    # Curator LLM call: pick the 2-3 most relevant results
    return curate_enrichment(results, learner_level)
```

**Total wait time: ~2-3 seconds (slowest source), not 8-12 (sequential).**

### Math verification flow

Every time the mentor generates a formula with a specific numerical result, a mandatory Wolfram Alpha call verifies it before it reaches the student:

```
Mentor generates: "For d_k = 64, the scaling factor is 1/√64 = 0.125"
                            ↓
Wolfram Alpha call: "1/sqrt(64)"
                            ↓
Wolfram returns: 0.125 ✓
                            ↓
Response delivered with: <verified by="wolfram">0.125</verified>
```

If Wolfram disagrees, the mentor's claim is flagged as uncertain and the student sees the actual Wolfram result.

### Prompt design (Anthropic best practices: tool use + source grounding)

```xml
<system>
You have access to these tools: [web_search, wolfram_eval, fetch_wikipedia, fetch_youtube].

Rules for tool use:
- For ANY mathematical expression with a specific numerical result: call wolfram_eval FIRST.
  Never state a number without tool verification.
- For any concept with a well-known visual explanation: call fetch_youtube and link to it.
- Cite every external fact: <source url="...">claim</source>
- If a tool call fails: state "unverified" and continue. Never invent a result.

Format enrichment as:
<enrichment>
  <background source="wikipedia" url="...">2-3 sentence background on the concept</background>
  <visual url="youtube_link">what this video shows and why it's worth watching</visual>
  <math_check expression="..." wolfram_result="..." matches_paper="true|false"/>
</enrichment>
</system>
```

---

## Phase E — Visual Intuition Layer

### What it is

Deep learning is fundamentally visual. Attention weights, gradient descent surfaces, embedding spaces, matrix transformations — these live in geometric space, not in prose. The visual layer generates interactive diagrams and runnable visualisation code for every key concept.

### Why it matters

When Andrej Karpathy explains backpropagation, he draws it on a whiteboard. When 3Blue1Brown explains linear transformations, they animate it. Text alone fails beginners — not because they're not smart enough, but because the human brain learns spatial concepts spatially.

### What gets visualised

| Concept | Visual |
|---------|--------|
| Attention weights | Heatmap: token × token grid showing attention scores |
| Softmax | Bar chart showing logits → probabilities transformation |
| Gradient descent | Contour plot with animated ball rolling toward minimum |
| Embedding space | 2D t-SNE scatter of word vectors, coloured by similarity |
| Matrix multiplication | Step-by-step animated grid showing each dot product |
| Positional encoding | Heatmap of the sinusoidal encoding pattern |
| Multi-head attention | Side-by-side attention heads with different patterns |
| Transformer layers | Interactive stack — click a layer to expand it |

### Runnable code generation

For every visual, the mentor generates a self-contained Python snippet:

```python
# Generated by mentor for concept: "softmax"
import numpy as np
import matplotlib.pyplot as plt

logits = np.array([2.0, 1.0, 0.1])  # raw scores before softmax

# Softmax: exp(x) / sum(exp(x))
exp_logits = np.exp(logits - logits.max())  # subtract max for numerical stability
probs = exp_logits / exp_logits.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(['Class A', 'Class B', 'Class C'], logits, color='steelblue')
ax1.set_title('Raw logits (before softmax)')
ax2.bar(['Class A', 'Class B', 'Class C'], probs, color='coral')
ax2.set_title(f'Probabilities (after softmax)\nsum = {probs.sum():.3f}')
plt.tight_layout()
plt.show()

# Notice: Class A had logit 2.0, gets 65% of probability
# Class C had logit 0.1, gets only 10% — differences are amplified
print(f"Probabilities: {probs.round(3)}")
```

The student can copy this, run it, change the logit values, and see immediately how the probabilities shift. This is active learning.

### Sandboxed execution

Code generated by the mentor runs in a sandboxed subprocess with a timeout:

```python
import subprocess, json

def run_code_sandbox(code: str, timeout: int = 10) -> dict:
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True,
        timeout=timeout
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }
```

The actual output is shown to the student — not a description of what the output would be. No hallucinated results.

### Prompt design (Anthropic best practices: structured code generation output)

```xml
<system>
For visual or mathematical concepts, generate a self-contained Python visualisation.

Return:
<visualization>
  <concept>name of what's being visualised</concept>
  <type>plot|animation|interactive</type>
  <libraries>numpy, matplotlib (always available in sandbox)</libraries>
  <code>
    # Complete, runnable Python code
    # Must work with: numpy, matplotlib, torch (optional)
    # Must produce output in ≤10 seconds
    # Add comment on every non-obvious line
  </code>
  <what_to_notice>One sentence pointing to the KEY insight the student should observe</what_to_notice>
  <experiment>One thing the student should try changing to deepen understanding</experiment>
</visualization>
</system>
```

---

## Phase F — Anti-Hallucination Stack

### What it is

A multi-layer system to ensure every claim the mentor makes is either verifiable or explicitly flagged as uncertain. For a student learning deep learning, a wrong explanation doesn't just waste time — it builds wrong intuition that's very hard to undo.

### The three layers

**Layer 1 — Paper-grounded RAG (already exists, improved)**

Every mentor response is anchored to retrieved paper chunks. The prompt explicitly instructs the model to cite which chunk supports each claim:

```xml
<cite chunk_id="3" section="3.2.1">
  The scaling factor 1/√d_k is applied to prevent dot products from growing large
  in magnitude, pushing softmax into regions of small gradients.
</cite>
```

If a student asks about something not covered in the paper, the mentor explicitly says:
*"This concept isn't covered in this paper. Here's what I know from general knowledge — but I'd recommend verifying this in a textbook:"* and then answers with a lower-confidence framing.

**Layer 2 — Wolfram Alpha math verification (Phase D integration)**

Every mathematical result with a specific number goes through Wolfram Alpha before being stated. This catches the most dangerous hallucination category: math that looks right but isn't.

**Layer 3 — Uncertainty hedging**

The prompt instructs the model to use explicit uncertainty language when it's working from general knowledge rather than the paper:

```
Certain (paper-grounded):   "The paper states that..."
Probable (general knowledge): "Generally in deep learning, though not stated in this paper..."
Uncertain:                    "I believe this is the case, but I'd recommend verifying..."
Never:                        Stating uncertain things as facts
```

### Confusion-resistant response structure

The mentor always separates:
1. What the paper says (cited)
2. What general DL knowledge adds (labelled)
3. What analogies/intuitions are simplifications (acknowledged)

This gives the student clear epistemic layers — they know what's fact, what's interpretation, and what's simplification.

### Prompt design (Anthropic best practices: explicit grounding + hallucination reduction)

```xml
<system>
You are a rigorous AI tutor. You ONLY make claims you can support.

Grounding hierarchy:
1. Paper chunk (highest confidence) → cite with <cite chunk_id="X" section="Y">
2. Wolfram-verified math → cite with <math verified="wolfram">result</math>
3. General DL knowledge → label with <general_knowledge>
4. Uncertain / inferred → label with <uncertain>

NEVER state a specific numerical result without tool verification.
NEVER present an analogy as a precise technical description — always note it's a simplification.
If you don't know something: say "I don't have enough information from this paper to answer this confidently."

The student trusts you. Don't betray that trust with a confident wrong answer.
</system>

<paper_chunks>{retrieved chunks — placed FIRST per Anthropic long-context best practice}</paper_chunks>
<student_question>{question}</student_question>
```

---

## Phase G — Adaptive Reading Path

### What it is

On PDF upload, the mentor analyses the paper's section structure and concept dependency graph, then generates a personalised reading order for the specific student — not the order the authors wrote it, but the order that makes sense for someone at this learner's level.

### Why it matters

Research papers are written for experts reviewing contributions, not for students learning concepts. The abstract references the results section. Section 3 assumes you've read five cited papers. A beginner starting at page 1 is immediately overwhelmed.

The adaptive reading path re-routes them: *"Start with Section 2 (background), then the introduction (now you have context), then Section 3.1, skip 3.2 for now, here's why..."*

### How it works

**Step 1 — Section extraction.** Your existing `preprocessor.py` already detects headings. Extended to extract a section map:

```python
section_map = {
    "Abstract": {"concepts": ["transformer", "attention", "sequence transduction"], "requires": []},
    "1. Introduction": {"concepts": ["seq2seq", "recurrent models"], "requires": []},
    "2. Background": {"concepts": ["sequential computation", "attention mechanisms"], "requires": ["1. Introduction"]},
    "3.1 Encoder and Decoder Stacks": {"concepts": ["multi-head attention", "residual connections"], "requires": ["2. Background"]},
    "3.2.1 Scaled Dot-Product Attention": {"concepts": ["dot product", "softmax", "scaling"], "requires": ["2. Background"]},
    # ...
}
```

**Step 2 — Learner-aware path planning.** The path planner LLM call takes the section map, the learner profile, and produces an ordered reading plan:

```
Beginner path for "Attention Is All You Need":
  1. Section 2 (Background) — introduces the concepts you'll need
  2. Introduction — now you have context for what's new
  3. Section 3.2.1 (Scaled Dot-Product Attention) — the core mechanism, start here
  4. Section 3.2.2 (Multi-Head Attention) — natural extension of 3.2.1
  5. Section 3.1 (Full Architecture) — now you can see how it all fits together
  6. Section 5 (Results) — what this all achieves
  Skip: Section 3.3, 3.4 (training details) — read after you understand the architecture

Prerequisites outside this paper (we'll teach these first):
  - Seq2seq models (10 minutes)
  - Attention mechanism basics (15 minutes)
```

**Step 3 — Milestone tracking.** As the student works through the paper, the session tracks which milestones they've reached. A milestone is a concept they can correctly answer a Socratic question about.

**Step 4 — Progress display.** The Streamlit frontend (Phase 5) shows a visual reading path with completed and upcoming sections.

### Prompt design (Anthropic best practices: document-first long-context prompting)

Per Anthropic's official guidance: *place long documents at the top of the prompt, before instructions. Queries at the end improve response quality by up to 30%.*

```xml
<system>
You are a curriculum designer for a deep learning tutoring system.
</system>

<paper_structure>
{full section map with concepts and dependencies — placed FIRST}
</paper_structure>

<learner_profile>
{level, known_concepts, gaps}
</learner_profile>

Now analyse the paper structure and the learner's profile to produce a personalised reading plan.

Return:
<reading_plan>
  <rationale>2-3 sentences on the student's situation and what they need</rationale>
  <sections>
    <section order="1" id="2. Background" estimated_time="15min">
      <why>reason this comes first for this learner</why>
    </section>
    ...
  </sections>
  <prerequisites_outside_paper>
    <concept estimated_time="10min">seq2seq models</concept>
  </prerequisites_outside_paper>
  <milestones>
    <milestone after_section="3.2.1">Student can explain why we scale by 1/√d_k</milestone>
    <milestone after_section="3.2.2">Student can explain what 'multi-head' adds over single attention</milestone>
  </milestones>
</reading_plan>
```

---

## 11. Prompt Engineering Strategy

All prompts in this system follow Anthropic's official best practices. Here's the complete strategy:

### Technique 1 — XML tags for structure

Every prompt uses XML tags to create clear boundaries between sections. This dramatically reduces prompt injection risks and makes outputs machine-parseable:

```xml
<system>role and rules</system>
<paper_chunks>retrieved context</paper_chunks>
<learner_profile>current student state</learner_profile>
<student_question>the actual question</student_question>
```

### Technique 2 — Document-first placement

Long content (paper chunks, enrichment results) always goes at the top of the prompt, before the instructions. Per Anthropic's documented testing, this improves response quality by up to 30% in long-context settings.

### Technique 3 — Chain-of-thought in `<thinking>` blocks

For reasoning tasks (prerequisite detection, confusion assessment, path planning), the prompt explicitly asks the model to think step-by-step inside `<thinking>` tags before producing the answer. This improves accuracy on multi-step reasoning tasks.

### Technique 4 — Positive AND negative examples

Every teaching prompt includes both:
- **Positive examples:** "Here is a good analogy for softmax: [example]"
- **Negative examples:** "DO NOT write: 'Softmax is a mathematical function that...' — this is jargon, not an analogy"

Per Anthropic's guidance, negative examples are especially important for preventing off-distribution behaviour — they define the *boundaries* of what's acceptable.

### Technique 5 — Prompt chaining, not one giant prompt

Each sub-task (profile → prerequisite check → enrich → teach → confusion detect) is a separate, focused LLM call. Benefits:
- Each call is verifiable independently
- Errors don't cascade through the whole pipeline
- Cheaper and faster (smaller prompts, faster models for simple tasks)
- Easier to improve one stage without breaking others

### Technique 6 — Explicit uncertainty instructions

Every prompt that might touch uncertain ground includes: *"If you don't know, say you don't know. Never invent. Uncertainty labelled explicitly is always better than confident error."*

---

## 12. Managing Large Data & Context Windows

Research papers are long. "Attention Is All You Need" is 15 pages of dense content. Here's how the system handles the data without overwhelming the context window:

### Problem: papers are too long for the context window

Even with 100k token context windows, loading an entire paper plus conversation history plus system prompt is wasteful, slow, and degrades response quality (the "lost in the middle" problem — LLMs attend poorly to content in the middle of very long contexts).

### Solution: hierarchical retrieval

Your existing parent-child chunking already handles this. The mentor mode extends it:

```
Level 1 — Paper sections (coarse)
  → "The student is asking about attention mechanisms"
  → Retrieve: Section 3 heading + summary
  
Level 2 — Parent chunks (~300 words)
  → Retrieve: The full attention mechanism subsection
  
Level 3 — Child chunks (~100 words)
  → Retrieve: The specific paragraph about scaling
```

Only Level 3 chunks are used for precise retrieval. The parent chunk is sent to the LLM for full context. The section heading is used for citation.

### Context window budget

```
System prompt (teaching instructions):  ~800 tokens
Learner profile:                        ~200 tokens
Retrieved paper chunks (3-5):         ~1,500 tokens
Enrichment (Wikipedia + Wolfram):       ~500 tokens
Conversation history (last 6 turns):  ~1,200 tokens
Student question:                       ~100 tokens
─────────────────────────────────────────────────
Total input:                          ~4,300 tokens
Response budget:                      ~2,000 tokens
```

This fits comfortably in any current model's context window while leaving room for complex responses.

### Caching for speed

Your existing SQLite cache (planned in Phase 7) should prioritise:
1. Wikipedia summaries for common DL concepts (softmax, attention, dropout) — these don't change
2. Wolfram results for standard formulas — `1/sqrt(64)` is always `0.125`
3. Prerequisite maps — the DL dependency graph is static knowledge

---

## 13. Why Not Recursive Language Models?

The question of whether to use Recursive Language Models (or related architectures) for this system is worth addressing directly.

**What a Recursive LM would offer:** Hierarchical processing of the paper — processing sections, then subsections, then sentences in a tree structure. Theoretically better at capturing document-level dependencies.

**Why prompt chaining is the right choice here instead:**

1. **Verifiability.** Each step in the prompt chain produces output you can inspect, log, and improve independently. A recursive LM produces one opaque output.

2. **Grounding.** The anti-hallucination stack requires explicit retrieval at each step. Prompt chaining integrates naturally with RAG at every step; recursive LMs do not.

3. **Practicality.** Recursive LMs are research-stage. Prompt chaining with Ollama/Groq/Gemini works today, runs locally, and is production-ready.

4. **Modularity.** You can swap in a better profiling model, a faster prerequisite checker, and a more capable teaching model independently. A monolithic recursive LM ties all these together.

**What IS useful from the recursive/hierarchical space:**

- **Parent-child chunking** (already in your system): hierarchical document representation
- **Multi-query retrieval** (already in your system): multiple rephrasings in parallel
- **Knowledge graph traversal** (already in your system): NetworkX BFS is effectively hierarchical reasoning over concepts

The combination of these three is, in practice, more powerful and reliable than a theoretical recursive LM for this specific task.

---

## 14. Implementation Roadmap

### Sprint 1 (1-2 weeks) — Learner Profiling + Prerequisite Engine

**Why start here:** Highest leverage, least UI dependency. Both run in the backend and improve every subsequent response.

- [ ] Add `learner_profile` dict to `session_store.py`
- [ ] Write `profiler.py` — fast LLM call on first message, XML output parsed into profile
- [ ] Extend `graph.py` — add `REQUIRES` edge type, load static DL prerequisite map
- [ ] Write `prerequisite_engine.py` — BFS traversal, gap detection, teaching queue
- [ ] Update `llm.py` — inject learner profile into every prompt
- [ ] Test: ask "explain attention" as a beginner → confirm prerequisites are taught first

### Sprint 2 (1-2 weeks) — Structured Teaching Mode

- [ ] Write `teaching_prompt.py` — five-step structured template, level-adaptive
- [ ] Add confusion detector — small classifier on response length + undefined term count
- [ ] Add analogy library — 30 pre-written analogies for common DL concepts
- [ ] Update `chat.py` — route through teaching mode for substantive concept questions
- [ ] Test: verify beginner, intermediate, advanced responses differ meaningfully

### Sprint 3 (1-2 weeks) — Math Verification + Enhanced Enrichment

- [ ] Extend `enrichment.py` — add YouTube transcript fetcher, ArXiv search
- [ ] Add mandatory Wolfram verification gate for numerical claims
- [ ] Add citation injector — every paper claim gets `<cite chunk_id="X">`
- [ ] Add uncertainty hedging to prompt — explicit language for uncertain claims
- [ ] Test: generate a wrong math result deliberately → confirm Wolfram catches it

### Sprint 4 (2 weeks) — Visual Layer + Code Sandbox

- [ ] Write `visualisation_generator.py` — concept-to-code mapping
- [ ] Build code sandbox — subprocess runner with timeout and output capture
- [ ] Write 10 visualisation templates (attention heatmap, softmax bars, gradient descent, etc.)
- [ ] Integrate with Streamlit frontend — display plots inline
- [ ] Test: "explain softmax" → plot appears, student can modify and re-run

### Sprint 5 (1 week) — Adaptive Reading Path

- [ ] Extend `preprocessor.py` — extract section map with concept lists
- [ ] Write `path_planner.py` — LLM call with document-first prompt, returns reading plan
- [ ] Add milestone tracker to session store
- [ ] Display reading path in Streamlit sidebar
- [ ] Test: upload "Attention Is All You Need" as a beginner → verify sensible path generated

---

## 15. What Success Looks Like

### For a beginner student

Before this feature:
> Student: "What is multi-head attention?"
> Agent: [explains multi-head attention assuming knowledge of dot products, softmax, and seq2seq]
> Student: loses interest, gives up

After this feature:
> Student: "What is multi-head attention?"
> Mentor: detects beginner level, finds gaps: softmax not known
> Mentor: "Before we get to multi-head attention, let me make sure softmax is clear — it comes up everywhere in deep learning. Think of it like converting scores into percentages..."
> [teaches softmax with analogy, formula, and 10-line code]
> Mentor: "Great. Now attention. Imagine you're reading a sentence and highlighting the words most relevant to understanding the current word..."
> [builds up to multi-head attention in 3 logical steps]
> Student: "Oh, that's why they call it 'multi'-head — it's like having multiple highlighters for different types of relationships?"
> Mentor: "Exactly. You've got it."

### Measurable outcomes

| Metric | Before | Target |
|--------|--------|--------|
| Concepts understood per session | 1-2 (surface) | 4-6 (with prerequisites) |
| Student confusion events | Frequent, silent | Detected and resolved proactively |
| Math accuracy | Unverified | 100% Wolfram-verified |
| Session depth | Single Q&A | Progressive concept building |
| Beginner completion rate | Drops off after 2-3 questions | Sustained engagement through reading path |

---

## Appendix — Key Files to Create/Modify

```
paper-tutor/
  core/
    profiler.py              ← NEW: learner profile inference
    prerequisite_engine.py   ← NEW: BFS gap detection + teaching queue
    teaching_prompt.py       ← NEW: structured five-step teach template
    visualisation_generator.py ← NEW: concept-to-code mapping
    path_planner.py          ← NEW: adaptive reading path
    code_sandbox.py          ← NEW: safe subprocess runner
    enrichment.py            ← EXTEND: YouTube + ArXiv + mandatory Wolfram gate
    graph.py                 ← EXTEND: add REQUIRES edges + DL prerequisite map
    preprocessor.py          ← EXTEND: extract section map on upload
    llm.py                   ← EXTEND: inject learner profile into all prompts
  data/
    dl_prerequisites.json    ← NEW: static DL concept dependency map (curated)
    analogy_library.json     ← NEW: 30+ pre-written analogies for DL concepts
  prompts/
    system.txt               ← EXTEND: add teaching mode instructions
    profiler.txt             ← NEW: learner profiling prompt
    prerequisite.txt         ← NEW: gap detection prompt
    path_planner.txt         ← NEW: reading path prompt
  app/
    main.py                  ← EXTEND: reading path sidebar, plot display
```

---
\
*References: [Anthropic Prompting Best Practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) | [Anthropic Hallucination Reduction Guide](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)*