"""
main.py

The FastAPI application.

Responsibility: receive HTTP requests, call the right modules,
return clean JSON responses.

This file should stay thin — it delegates all real work to
pdf_processor, vector_store, llm_client, and session_store.

Run the server with (from the repo root):
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000/docs to see the interactive API docs
that FastAPI generates automatically.
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Internal modules (all imported with the `backend.` package prefix so
#    the server can be launched from the repo root as:
#    uvicorn backend.main:app --reload) ────────────────────────────────────────

from backend.pdf_processor import process_pdf
from backend.vector_store   import VectorStore
from backend.llm_client     import chat, check_ollama_running, llm_complete
from backend.session_store  import SessionStore

from backend.retrieval.bm25_store  import BM25Store
from backend.retrieval.multi_query import multi_query_search

from backend.knowledge.graph           import KnowledgeGraph
from backend.knowledge.graph_retrieval import augment_query, get_context_for_concept

from backend.tools.enrichment import enrich, extract_triples

# ── Mentor agent modules ──────────────────────────────────────────────────────────
from backend.agent.profiler             import profile_learner
from backend.agent.prerequisite_engine  import get_prerequisite_queue
from backend.agent.teaching_prompt      import build_system_prompt


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Paper Tutor",
    description="Upload a research paper PDF and chat with an AI tutor about it.",
    version="0.1.0",
)

# Allow requests from any origin during development.
# In production you'd restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# These are module-level singletons — created once when the server starts.
vector_store    = VectorStore()
bm25_store      = BM25Store()
session_manager = SessionStore()
knowledge_graph = KnowledgeGraph()


# ── Request / Response models ──────────────────────────────────────────────────
#
# Pydantic models do two things:
#   1. Validate incoming request data (wrong types → automatic 422 error)
#   2. Document the API (FastAPI uses these to generate /docs)

class ChatRequest(BaseModel):
    session_id: str
    question:   str

class ChatResponse(BaseModel):
    answer:               str
    sources:              list
    learner_profile:      dict = {}
    prerequisites_taught: list = []

class UploadResponse(BaseModel):
    session_id:  str
    chunk_count: int
    sections:    list[str]
    message:     str

class SessionResponse(BaseModel):
    session_id:     str
    paper_filename: str | None
    turn_count:     int
    history_length: int
    created_at:     str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _summarise_sources(chunks):
    """
    Turn retrieved chunks into lightweight source references for the response.
    We don't send full chunk text to the frontend — just enough to show
    "this answer came from Methods p.4".
    """
    return [
        {
            "section": c["section"],
            "page":    c["page"],
            "score":   round(c.get("score", c.get("rrf_score", 0.0)), 3),
            "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
        }
        for c in chunks
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Quick check: is the server alive? Is Ollama running? How many chunks stored?
    Call this first to verify everything is working.
    """
    groq_ok = bool(os.getenv("GROQ_API_KEY", "").strip())
    return {
        "status":            "ok",
        "ollama_running":    check_ollama_running(),
        "groq_available":    groq_ok,
        "active_llm":        "groq" if groq_ok else "ollama",
        "chunks_stored":     vector_store._collection.count(),
        "active_sessions":   len(session_manager.list_sessions()),
        "graph_nodes":       knowledge_graph.node_count(),
        "graph_edges":       knowledge_graph.edge_count(),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, process it into chunks, and store it.
    Returns a session_id you'll use for all subsequent /chat calls.

    The file is saved to a temp location, processed, then deleted.
    We never store the original PDF — only the chunks.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process the PDF into chunks
        chunks = process_pdf(tmp_path)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from this PDF. "
                       "Is it a scanned image PDF? Try a text-based PDF."
            )

        # ── Store chunks in both vector store and BM25 index ──────────────────
        vector_store.clear()
        vector_store.add_chunks(chunks)

        bm25_store.index(chunks)   # rebuild keyword index with the same chunks

        # ── Build knowledge graph from paper chunks ───────────────────────────
        knowledge_graph.clear()
        paper_triples = []
        for chunk in chunks[:30]:   # first 30 chunks — enough for a good graph
            triples = extract_triples(chunk["text"], source_label="paper")
            paper_triples.extend(triples)

        knowledge_graph.add_triples(paper_triples)
        knowledge_graph.save()
        print(f"Knowledge graph: {knowledge_graph.node_count()} nodes, "
              f"{knowledge_graph.edge_count()} edges")

        # Create a new session for this upload
        session_id = session_manager.create_session(paper_filename=file.filename)

        stats = vector_store.get_stats()

        return UploadResponse(
            session_id=session_id,
            chunk_count=len(chunks),
            sections=stats["sections"],
            message=f"Successfully processed '{file.filename}'. "
                    f"Ready to answer questions.",
        )

    finally:
        # Always clean up the temp file, even if processing failed
        os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """
    Ask a question about the uploaded paper.

    Mentor Orchestrator pipeline:
      1. Profile learner (first turn only)
      2. Detect missing prerequisites via BFS
      3. Augment query with knowledge-graph concepts
      4. Multi-query hybrid retrieval
      5. Web enrichment (Wikipedia / Wolfram)
      6. Build structured teaching prompt (analogy → intuition → math → code → check)
      7. LLM call with full context
    """
    session_id = request.session_id

    if not session_manager.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first via POST /upload."
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")
    if vector_store._collection.count() == 0:
        raise HTTPException(status_code=400, detail="No paper loaded. Upload a PDF first.")

    # ───────────────────────────────────────────────────────────────────────
    # Step 1 — Profile learner on first turn
    # ───────────────────────────────────────────────────────────────────────
    learner_profile = session_manager.get_profile(session_id)
    if not learner_profile.get("profiled", False):
        print("[mentor] Profiling learner...")
        inferred = profile_learner(question)
        session_manager.update_profile(session_id, inferred)
        learner_profile = session_manager.get_profile(session_id)
        print(f"[mentor] Level: {learner_profile['level']}")

    # ───────────────────────────────────────────────────────────────────────
    # Step 2 — Prerequisite detection
    # ───────────────────────────────────────────────────────────────────────
    # Extract the first meaningful term from the question as the target concept
    import re as _re
    stopwords = {"what","how","why","does","do","is","are","the","a","an",
                 "in","of","this","paper","explain","describe","tell","me","about"}
    words = _re.findall(r"[a-z]+", question.lower())
    key_words = [w for w in words if w not in stopwords and len(w) > 3]
    target_concept = " ".join(key_words[:3]) if key_words else ""

    prerequisites_to_teach = []
    if target_concept and learner_profile.get("level") != "advanced":
        prerequisites_to_teach = get_prerequisite_queue(
            target_concept, learner_profile, knowledge_graph
        )
        if prerequisites_to_teach:
            print(f"[mentor] Prerequisites for '{target_concept}': {prerequisites_to_teach}")

    # ───────────────────────────────────────────────────────────────────────
    # Step 3 — Retrieval (augmented query + hybrid search)
    # ───────────────────────────────────────────────────────────────────────
    augmented_question = augment_query(question, knowledge_graph)
    child_chunks    = multi_query_search(augmented_question, vector_store, bm25_store, n_results=3)
    relevant_chunks = child_chunks if child_chunks else vector_store.search(question, n_results=3)

    # ───────────────────────────────────────────────────────────────────────
    # Step 4 — Web enrichment
    # ───────────────────────────────────────────────────────────────────────
    enriched    = enrich(question, relevant_chunks, skip_triple_extraction=True)
    graph_ctx   = get_context_for_concept(target_concept, knowledge_graph) if target_concept else ""
    extra       = graph_ctx + ("\n\n" if graph_ctx else "") + enriched["context"]

    # ───────────────────────────────────────────────────────────────────────
    # Step 5 — Build structured teaching prompt + LLM call
    # ───────────────────────────────────────────────────────────────────────
    system_prompt = build_system_prompt(learner_profile, prerequisites_to_teach)

    history  = session_manager.get_history(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for past_q, past_a in history:
        messages.append({"role": "user",      "content": past_q})
        messages.append({"role": "assistant", "content": past_a})

    # Format paper chunks + extra context for the user message
    chunk_text = "\n\n---\n\n".join(
        f"[{c['section']}, p.{c['page']}]\n{c['text']}" for c in relevant_chunks
    ) or "No paper sections found."
    user_msg = f"PAPER CONTEXT:\n{chunk_text}"
    if extra.strip():
        user_msg += f"\n\nADDITIONAL CONTEXT:\n{extra}"
    user_msg += f"\n\nSTUDENT QUESTION: {question}"
    messages.append({"role": "user", "content": user_msg})

    try:
        answer = llm_complete(messages, temperature=0.3)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="No LLM available. Check Ollama or GROQ_API_KEY.")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Model took too long to respond.")

    # ───────────────────────────────────────────────────────────────────────
    # Step 6 — Update session state
    # ───────────────────────────────────────────────────────────────────────
    session_manager.add_turn(session_id, question, answer)
    for concept in prerequisites_to_teach:
        session_manager.mark_concept_taught(session_id, concept)
    if target_concept:
        session_manager.mark_concept_taught(session_id, target_concept)

    final_profile = session_manager.get_profile(session_id)

    return {
        "answer":                answer,
        "sources":               _summarise_sources(relevant_chunks),
        "learner_profile":       {
            "level":   final_profile.get("level", "beginner"),
            "taught":  final_profile.get("taught_this_session", []),
        },
        "prerequisites_taught": prerequisites_to_teach,
    }


@app.get("/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """
    Get current session state — useful for the frontend to restore UI state.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return SessionResponse(
        session_id=session_id,
        paper_filename=session["paper_filename"],
        turn_count=session["turn_count"],
        history_length=len(session["history"]),
        created_at=session["created_at"],
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """
    Clear a session's history and the vector store.
    Use this when the user wants to load a new paper.
    """
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    session_manager.clear_session(session_id)
    vector_store.clear()
    bm25_store.index([])   # clear keyword index too

    return {"status": "cleared", "session_id": session_id}