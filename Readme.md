# Research AI Mentor 🎓

Upload any research paper (PDF) and let a **Socratic AI Mentor** guide you through it from first principles.

Unlike traditional RAG systems that just retrieve text and summarize, this agent **teaches** you. It generates personalized learning curricula, renders complex architectures as Mermaid.js diagrams, explains math step-by-step, and uses Socratic questioning to ensure you understand one concept before moving to the next.

## Why Our Agent is Unique & More Helpful

| Feature | Generic Chat-with-PDF | Research AI Mentor |
|---|---|---|
| **Teaching Approach** | Reactive (only answers what you ask) | **Proactive Mentor Mode**: Generates a step-by-step curriculum and guides you through it. |
| **Understanding Check** | None | **Socratic Check-Gates**: Asks you questions to verify comprehension before advancing. |
| **Visual Learning** | Text only | **Live Diagrams**: Dynamically generates Mermaid.js flowcharts and architectures inline. |
| **Math & Formulas** | Describes formulas in plain text | **LaTeX + Wolfram**: Renders beautiful math and evaluates expressions numerically. |
| **Prerequisite Awareness**| Assumes you know the basics | **Knowledge Graph Traversal**: Automatically detects and teaches missing prerequisites first. |
| **Web Enrichment** | None (or slow search tools) | **Parallel Enrichment**: Wikipedia & Wolfram Alpha run in the background to add context. |
| **UI/UX** | Basic chat interfaces | **Premium Glassmorphism**: Beautiful, dynamic interface with interactive learning path tracking. |
| **Resilience** | Breaks if API is down | **Hybrid Engine**: Uses blazing-fast **Groq** by default, falls back to local **Ollama** seamlessly. |

---

## Core Features (Socratic Mentor Mode)

- **Curriculum Generation**: Say "I want to learn about transformers from scratch." The agent analyzes the paper, profiles your current knowledge level, and generates a 5-step learning path.
- **Turn-by-Turn Orchestration**: Teaches one concept at a time in a conversational, friendly tone. No overwhelming walls of text.
- **Socratic Evaluation**: Ends each step with a check question. Evaluates your answer (forgivingly) and either auto-advances the curriculum or gently corrects you.
- **Learner Profiling**: Infers if you are a beginner, intermediate, or advanced learner based on your questions, tailoring the depth and tone of the explanation.
- **Hybrid RAG + Knowledge Graph**: Combines BM25 exact keyword search with vector semantic search (Reciprocal Rank Fusion). Extracts concept triples into a NetworkX graph to find non-obvious connections across the paper.

## Setup (5 minutes)

```bash
# 1. Clone this repository
cd paper-tutor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Environment Variables
# Get a free Groq API key at console.groq.com (Primary fast inference)
# Optional: Install Ollama (ollama.com) and run `ollama pull llama3.2` for local fallback
cp .env .env.local
# Edit .env and add GROQ_API_KEY=your_key

# 5. Run the full stack
# Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000 &

# Start the Streamlit frontend
streamlit run frontend/app.py
```

The app opens at `http://localhost:8501`.

---

## Architecture & Tech Stack

### Frontend (`frontend/app.py`)
Built with **Streamlit** using custom HTML/CSS injections for a premium **Glassmorphism** UI. 
Features live rendering of **Mermaid.js** diagrams via JS components and native **LaTeX** support for math. The sidebar tracks your Learner Profile, System Health, Knowledge Graph Stats, and active Learning Path progress.

### Backend Orchestrator (`backend/main.py`)
Built with **FastAPI**. Exposes `/upload` for PDF processing and `/chat` for the Mentor Engine.

### Mentor Engine (`backend/agent/`)
- `intent_detector.py`: Detects if the user wants to LEARN a broad topic vs asking a specific factual question.
- `curriculum.py`: Generates the step-by-step learning path tailored to the paper context.
- `teaching_prompt.py`: Crafts the highly structured, pedagogical LLM prompt forcing analogy → intuition → math → check question.
- `step_evaluator.py`: Grades user responses to Socratic check questions to determine if they can advance.
- `profiler.py` & `prerequisite_engine.py`: Profiles user expertise and consults the Knowledge Graph to inject necessary prerequisites.

### RAG Pipeline (`backend/retrieval/`)
- **PyMuPDF**: Intelligent, font-size-aware document chunking.
- **ChromaDB**: Vector storage using `sentence-transformers/all-MiniLM-L6-v2`.
- **BM25Okapi**: Keyword index for exact-match retrieval.
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) blending vector + BM25 results.
- **Parent-Child Chunking**: Retrieves precise 100-word chunks, but feeds 300-word parent context to the LLM.

### Knowledge Graph & Web Tools
- **NetworkX**: Builds a directed graph of concept triples extracted from the paper. Used for query augmentation and prerequisite detection.
- **Web Enrichment**: Parallel ThreadPool execution of Wikipedia summaries and Wolfram Alpha math evaluation to inject external knowledge into the context window.

---

## What Happens When You Learn (Data Flow)

1. **User asks**: "I want to learn about convolutional neural networks."
2. **Intent Detection**: Classifier flags this as a `LEARN` intent.
3. **Curriculum Engine**: Fetches chunks about CNNs, generates a 5-step path (e.g., Step 1: Convolutions, Step 2: Pooling...).
4. **Mentor Mode Activated**: 
    - Session state locks into Step 1.
    - System retrieves paper chunks, Wikipedia context, and knowledge graph edges specifically for Step 1.
    - `build_step_prompt` forces the LLM to explain *only* Step 1 using analogies and end with a check question.
5. **User Replies**: User answers the check question.
6. **Evaluation**: `step_evaluator` grades the answer. If correct, the sidebar progress bar updates and the engine auto-advances to Step 2!

---

## Testing

Over 200 unit tests cover the RAG, Graph, and orchestration layers. Run them locally:

```bash
pytest backend/tests/ -v
```

All external services (Groq, Wolfram, Wikipedia, Chroma) are mocked during testing to ensure tests run offline in milliseconds.
