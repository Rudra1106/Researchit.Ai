# Paper Tutor — Phase 1

Upload any research paper (PDF) and have a deep conversation about it.
The AI explains all concepts from scratch: plain language, math, code examples.

## Setup (5 minutes)

```bash
# 1. clone / download this folder, then:
cd paper-tutor

# 2. create a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# 3. install dependencies
pip install -r requirements.txt

# 4. add your API key(s) to .env
#    get a free Groq key at: console.groq.com
#    get a free Gemini key at: aistudio.google.com
cp .env .env.local
nano .env    # or open in any editor

# 5. run  (always from the project root, not from inside app/)
streamlit run run.py
```

The app opens at http://localhost:8501

## How it works (Phase 1)

```
PDF upload
  ↓
preprocessor.py   — PyMuPDF extracts text, detects headings/math, splits into chunks
  ↓
vectorstore.py    — sentence-transformers embeds chunks, ChromaDB stores them
  ↓
[user asks a question]
  ↓
retriever.py      — embeds question, finds top-4 matching chunks (+ math boost)
  ↓
llm.py            — sends system prompt + history + chunks + question to Groq/Gemini
  ↓
main.py           — displays response with source citations
```

## Folder structure

```
paper-tutor/
  app/
    main.py          ← Streamlit UI
    chat.py          ← conversation loop
    state.py         ← session state management
  core/
    preprocessor.py  ← PDF → chunks
    vectorstore.py   ← ChromaDB wrapper
    retriever.py     ← semantic search
    llm.py           ← Groq / Gemini API
  prompts/
    system.txt       ← tutor system prompt
  data/              ← git-ignored
    uploads/
    chroma_db/
  requirements.txt
  .env
```

## What's coming in Phase 2

- Hybrid BM25 + semantic search
- Parent-child chunking
- Knowledge graph (NetworkX)
- Math explanation tool
- Code generation tool

# Research Paper AI Tutor

A local, fully-offline AI system that turns any research paper PDF into an interactive tutor. Upload a paper, ask questions at any depth, and receive explanations grounded in the paper's own text — enriched with web sources, rendered math, runnable code, and a knowledge graph that connects concepts across sections.

Built entirely with free tools: Ollama (local LLMs), ChromaDB, sentence-transformers, NetworkX, FastAPI. No OpenAI. No cloud APIs (except optional Wolfram Alpha free tier).

---

## What makes this different from ChatGPT + a PDF

| Feature | Generic chatbot | This system |
|---|---|---|
| Source grounding | Hallucination-prone | Every answer cites the exact paper section |
| Math | Describes formulas | Retrieves exact LaTeX, evaluates with Wolfram Alpha |
| Memory | Forgets after context window | Tracks what you've learned across the session |
| Retrieval | Keyword search or none | Hybrid BM25 + vector + multi-query + parent-child |
| Concept connections | Single-turn only | Knowledge graph links ideas across paper sections |
| Web enrichment | None | Wikipedia + Wolfram Alpha run in parallel per question |
| Prerequisite awareness | None | Checks what you need to know before answering |
| Runs locally | No | Yes — everything runs on your machine |

---

## Unique features (planned and implemented)

### Implemented

**Hybrid RAG retrieval** — combines semantic vector search (finds conceptually similar chunks) with BM25 keyword search (finds exact terms like `d_k` or `BLEU`). Results are blended using Reciprocal Rank Fusion so the best of both systems wins.

**Multi-query retrieval** — the orchestrator rephrases your question three ways and runs all four searches in parallel. If "how does position work?" misses the chunk, "positional encoding mechanism" finds it.

**Parent-child chunking** — stores small child chunks (≈100 words) for precise retrieval, but fetches their parent paragraphs (≈300 words) to give the LLM rich context. Precise search, rich reading.

**Knowledge graph** — every concept triple extracted from the paper ("ReLU → solves → vanishing gradient") is stored in a NetworkX DiGraph. When you ask about ReLU, the graph automatically appends related concepts (vanishing gradient, sigmoid, slow training) to the search query — finding chunks that don't contain the word "ReLU" but are deeply relevant.

**Web enrichment** — Wikipedia and Wolfram Alpha run in parallel for every question. Wikipedia adds encyclopedic background. Wolfram Alpha evaluates mathematical expressions numerically. Both sources feed triples back into the knowledge graph for future questions.

**Parallel source fetching** — web sources run concurrently via `ThreadPoolExecutor`. Total wait time is the slowest source (~2–3 seconds), not the sum of all sources.

**Triple extraction from web results** — Wikipedia summaries are passed to the LLM for concept-relation-concept extraction. These triples enter the knowledge graph, so concepts learned from Wikipedia inform future paper retrievals.

**Session state management** — each upload creates a session. The session tracks conversation history (trimmed to last 6 turns for context window management), which paper is loaded, and how many turns have occurred.

**Graceful degradation** — every external call (Wolfram, Wikipedia, Ollama) is wrapped in try/except. If Wolfram is down, the answer continues without it. If multi-query rephrasing fails, the original query is used. Nothing breaks the core flow.

### Planned (prioritised)

**Live math sandbox** — edit any formula extracted from the paper and see the output change. `d_k = 64` in the attention formula? Change it to 8 and watch the softmax distribution shift. Wolfram Alpha evaluates it. A small Python runner shows the numerical result.

**Prerequisite task queue** — before explaining dropout, the system checks the knowledge graph: does this session already know what overfitting is? What a neuron is? If not, it queues those concepts first and teaches them before answering the original question. Turns an accurate answer into an actually learnable one.

**Paper notation glossary** — on upload, the system scans every section for mathematical variables (`α`, `λ`, `d_k`, `b^i_{x,y}`) and builds a searchable symbol dictionary. When the user sees an unfamiliar symbol, they can look it up instantly without re-reading the paper.

**Decision reasoning mode** — answers "why did the authors choose X?" by traversing the knowledge graph for the causal chain. Why ReLU? Graph path: `slow training → caused by → vanishing gradients → caused by → saturating activations → solved by → ReLU`. The explanation is built around the path, not just the conclusion.

**Analogy engine** — every technical concept gets a real-world parallel before the formal explanation. Convolution = sliding a flashlight over a dark room. Dropout = a sports team where any player might be absent, so everyone learns to be independent. Analogy first, math second.

**Confusion detector** — after each answer, a small classifier checks: was the answer abstract? Long? Did it use many undefined terms? If so, the system proactively asks "which part was unclear?" instead of waiting for the user to struggle silently.

**Cross-paper comparison** — upload two papers. Ask how they approach the same problem differently. AlexNet vs VGGNet on depth. Attention Is All You Need vs BERT on pretraining. The system runs retrieval against both vector stores simultaneously.

**Adaptive reading path** — on upload, the system analyses the paper's section structure and concept dependency graph, then proposes a personalised reading order. "Before reading Section 3.3, you should understand Sections 3.1 and 2. Want me to start there?"

**LaTeX rendering** — math in responses renders as proper equations in the frontend, not raw `\frac{}` strings. KaTeX renders in the browser at the Streamlit layer.

**Sandboxed code execution** — generated Python code actually runs. `softmax([1, 2, 3])` returns `[0.09, 0.245, 0.665]`. The user sees real output, not just the code.

**Session persistence** — sessions survive server restarts via SQLite. Continue a conversation about AlexNet days later without re-uploading or re-explaining context.

**Search result caching** — Wikipedia and Wolfram responses are cached by concept string in a local SQLite store. Repeated questions about "softmax" hit cache in milliseconds, not the network.

---

## Architecture

```
research_tutor/
├── backend/
│   ├── main.py                    # FastAPI application — all HTTP endpoints
│   ├── pdf_processor.py           # PDF extraction and intelligent chunking
│   ├── vector_store.py            # ChromaDB wrapper — embedding and retrieval
│   ├── llm_client.py              # Ollama HTTP client — prompt building and chat
│   ├── session_store.py           # In-memory session state management
│   ├── retrieval/
│   │   ├── bm25_store.py          # BM25 keyword search index
│   │   ├── hybrid_search.py       # RRF blending of BM25 + vector results
│   │   ├── multi_query.py         # LLM rephrasing + result merging
│   │   └── parent_child.py        # Parent/child chunk splitting and lookup
│   ├── tools/
│   │   ├── web_search.py          # Wikipedia + DuckDuckGo search
│   │   ├── wolfram.py             # Wolfram Alpha Short Answers API
│   │   └── enrichment.py         # Parallel source coordinator + triple extraction
│   ├── knowledge/
│   │   ├── graph.py               # NetworkX knowledge graph — CRUD + traversal
│   │   └── graph_retrieval.py     # Query augmentation + path explanation
│   └── tests/
│       ├── test_pdf_processor.py
│       ├── test_vector_store.py
│       ├── test_llm_client.py
│       ├── test_main.py
│       ├── test_bm25_store.py
│       ├── test_hybrid_search.py
│       ├── test_multi_query.py
│       ├── test_web_search.py
│       ├── test_wolfram.py
│       ├── test_enrichment.py
│       ├── test_graph.py
│       └── test_graph_retrieval.py
├── frontend/
│   └── app.py                     # Streamlit chat interface (planned)
├── chroma_db/                     # ChromaDB on-disk storage (auto-created)
├── knowledge_graph.json           # Persisted knowledge graph (auto-created)
├── requirements.txt
└── .env                           # WOLFRAM_APP_ID (never commit this)
```

---

## File-by-file reference

### `backend/pdf_processor.py`

**What it does:** Takes a PDF file path. Returns a list of structured chunk dictionaries.

**Key functions:**

`extract_blocks(pdf_path)` — opens the PDF with PyMuPDF (`fitz`), iterates every page, and classifies each text block by type. Classification uses font size ratios (headings are larger than body text), y-position thresholds (footnotes live in the bottom 12% of the page), and regex patterns (figure captions start with "Figure N" or "Table N"). Returns raw classified blocks with page numbers and bounding boxes.

`chunk_blocks(blocks)` — converts classified blocks into semantic chunks. A new heading resets the current section and flushes the buffer. Body text accumulates until it exceeds `MAX_CHUNK_WORDS` (300). Captions always get their own chunk. Each chunk carries: `chunk_id`, `section`, `page`, `text`, `word_count`, `has_math`, `has_code`.

`process_pdf(pdf_path)` — public entry point. Calls `extract_blocks` then `chunk_blocks`. This is the only function `main.py` calls.

`_detect_math(text)` — regex-based check for LaTeX patterns (`\frac`, `$$`, `^{`, `\sum`, etc.). Sets `has_math=True` on the chunk.

`_detect_code(text)` — looks for Python keywords (`def`, `for`, `import`) and indented lines. Sets `has_code=True`.

`_get_median_font_size(page)` — computes the most common font size on a page using PyMuPDF's character-level `"dict"` extraction. The median becomes the baseline for heading detection on that page.

**Why it matters:** Everything downstream depends on chunk quality. A bad chunk (cut mid-equation, mixed across sections) means bad retrieval, which means bad answers. This module's heuristics are the foundation.

**Design decisions:**
- Position-based classification over ML — font size ratios are debuggable. An ML classifier is a black box that fails silently.
- Section-boundary chunking over fixed-token chunking — preserves semantic coherence. A chunk about "Results" won't accidentally include a sentence from "Methods".
- Metadata on every chunk — enables filtered retrieval later (search only in Methods section, search only math-containing chunks).

---

### `backend/vector_store.py`

**What it does:** Stores chunk embeddings in ChromaDB. Answers similarity queries.

**Key class:** `VectorStore`

`add_chunks(chunks)` — batch-embeds all chunk texts using `all-MiniLM-L6-v2` (a 80MB local model, 384-dimensional output). Calls ChromaDB's `upsert()` so re-uploading the same paper doesn't create duplicates. Stores text, embedding, and metadata separately — ChromaDB indexes the embedding, keeps text for retrieval.

`search(query, n_results, where)` — embeds the query with the same model (critical — mismatched models give random results). Calls ChromaDB's `query()`. Returns results with a `score` field (1 − cosine distance, so 1.0 = identical, 0.0 = unrelated). Accepts an optional `where` filter for metadata-constrained search.

`search_with_filter(query, section, has_math)` — convenience wrapper for common filter patterns. `section="Methods"` restricts results to that section. `has_math=True` restricts to math-containing chunks.

`clear()` — deletes and recreates the ChromaDB collection. Called before loading a new paper so old chunks don't contaminate results.

`get_stats()` — returns total chunk count, list of unique sections, and counts of math/code chunks. Used by `/health` endpoint and for debugging.

**Why it matters:** Embedding quality directly determines which chunks get retrieved for a question. The `all-MiniLM-L6-v2` model is small enough to run on CPU but strong enough for technical text similarity.

**Design decisions:**
- `PersistentClient` for production (survives restarts), `EphemeralClient` in tests (no disk I/O, no cleanup needed).
- `upsert` over `add` — `add()` throws on duplicate IDs. `upsert()` updates in place. Safer for iterative development.
- Cosine similarity over L2 distance — cosine is direction-sensitive, not magnitude-sensitive. A longer chunk and a shorter chunk about the same topic get similar scores. L2 would penalise the longer one.

---

### `backend/llm_client.py`

**What it does:** Wraps Ollama's HTTP API. Builds prompts. Returns answers.

**Key functions:**

`chat(question, chunks, history, extra_context)` — main entry point. Calls `_build_messages()` then `_call_ollama()`. Returns the LLM's response string.

`_build_messages(question, chunks, history, extra_context)` — assembles the message list Ollama expects. Structure: system prompt → history (interleaved user/assistant turns) → current message (paper chunks + web context + question). History is replayed in full each turn — this is how the LLM gets memory.

`_format_chunks_as_context(chunks)` — formats each chunk with its section and page number as a labeled header. Ensures the LLM knows where each piece of text came from and can reference it naturally.

`_call_ollama(messages)` — POSTs to `http://localhost:11434/api/chat`. Sets `temperature=0.3` (low = consistent, accurate; high = creative, variable). Raises `ConnectionError` if Ollama isn't running, `TimeoutError` if it takes over 120 seconds. These are caught by `main.py` and converted to clean HTTP error responses.

`check_ollama_running()` — GETs `/api/tags`. Returns True/False. Called at server startup and in `/health`.

`SYSTEM_PROMPT` — the constant that defines the tutor's personality. It instructs the LLM to: assume beginner level, explain math symbol by symbol, give analogies, show minimal code examples, only use paper content. This is the most important string in the system — changing it changes the entire character of the tutor.

**Design decisions:**
- No streaming in v1 — simpler to test, simpler error handling. Phase 5 will add streaming for better UX.
- History trimmed to 6 turns in `session_store.py` — LLMs have context window limits. 6 turns ≈ 2000–3000 tokens of history, leaving plenty of space for paper context.
- Temperature 0.3 — educational explanations need to be accurate and repeatable. Higher temperature risks the model fabricating formula details.

---

### `backend/session_store.py`

**What it does:** Tracks conversation state per user session. Lives entirely in memory.

**Key class:** `SessionStore`

`create_session(paper_filename)` — generates a UUID session ID, creates a session dictionary with empty history, filename, creation timestamp, and turn counter. Returns the ID.

`add_turn(session_id, question, answer)` — appends `(question, answer)` to the session's history list. If history exceeds `MAX_HISTORY_TURNS` (6), removes the oldest turn. This sliding window prevents context window overflow.

`get_history(session_id)` — returns the list of `(question, answer)` tuples. Passed directly to `llm_client.chat()`.

`clear_session(session_id)` — resets history and paper filename but keeps the session alive. Called when the user wants to ask about a different paper without losing their session ID.

**Why it matters:** Without state, every question is answered without context. "What is dropout?" answered well. "How does it relate to what we just discussed?" answered as if nothing came before. State is what turns a search engine into a tutor.

**Design decisions:**
- In-memory dict over a database — fast, simple, zero dependencies. Planned upgrade: SQLite for persistence across restarts.
- UUID session IDs — unguessable, unique, no coordination needed.
- Session survives `clear_session()` — the frontend doesn't need to handle a new session ID when the user uploads a new paper.

---

### `backend/main.py`

**What it does:** FastAPI application. Defines all HTTP endpoints. Wires modules together. Handles all error translation.

**Module-level singletons:** `vector_store`, `bm25_store`, `parent_store`, `session_manager`, `knowledge_graph` — created once when the server starts. All endpoints share these instances. This is intentional: the vector store and knowledge graph are stateful resources that persist across requests.

**Endpoints:**

`GET /health` — returns Ollama status, chunk count, active session count, and knowledge graph node/edge counts. Always returns 200. Call this first to verify the system is working.

`POST /upload` — accepts a PDF file via multipart form. Saves to a temp file. Calls `process_pdf()` → `split_into_parent_children()` → `vector_store.add_chunks(children)` → `bm25_store.index(children)` → `parent_store.index(parents)` → knowledge graph triple extraction from first 30 chunks. Creates a new session. Returns `session_id`, chunk count, and sections found. Always cleans up the temp file in a `finally` block.

`POST /chat` — validates session, validates question length (max 1000 chars). Runs the full retrieval pipeline: `augment_query()` → `multi_query_search()` → `parent_store.get_many()` → `enrich()` → `get_context_for_concept()` → `chat()`. Saves the turn to session history. Returns answer and source references.

`GET /session/{session_id}` — returns session metadata: filename, turn count, history length, creation time. Lets the frontend restore UI state after a page refresh.

`DELETE /session/{session_id}` — clears history and the vector store. Use when loading a new paper.

**Error handling:**
- `400` — bad input (wrong file type, empty question, question too long)
- `422` — valid input, but couldn't process (PDF has no extractable text — likely a scanned image)
- `404` — session not found
- `503` — Ollama is not running
- `504` — Ollama timed out

**Why thin controllers matter:** `main.py` contains no business logic. It validates, delegates, and translates. Every real operation is in a module that can be tested independently. If `/chat` breaks, you can bisect by testing `multi_query_search()` directly without starting a server.

---

### `backend/retrieval/bm25_store.py`

**What it does:** Keyword-based search over chunk text using the BM25Okapi algorithm.

**Key class:** `BM25Store`

`index(chunks)` — tokenises each chunk's text (lowercase, split on non-alphanumeric characters, preserving numbers), builds a `BM25Okapi` index. Must be called after chunks are loaded into the vector store.

`search(query, n_results)` — tokenises the query, scores all indexed chunks, returns the top results sorted by `bm25_score`. Filters out zero-score chunks (no keyword overlap).

`_tokenise(text)` — regex-based tokenisation: `re.findall(r"[a-z0-9]+", text.lower())`. Keeps numbers because `d_k`, `512`, and `0.0005` are meaningful tokens in research papers.

**Why BM25 and not just vector search:** Vector search understands meaning but misses exact terms. If someone asks "what is d_k", the vector for that question is semantically near "dimensionality scaling" but may not find the specific chunk where `d_k` appears. BM25 finds it instantly because it matches the exact token `dk`. BM25 and vector search fail on complementary cases — running both and blending is strictly better than either alone.

**Design decisions:**
- `n_results=10` default (more than vector search's 5) — BM25 candidates are cheap to generate. More candidates give RRF more material to rerank.
- Rebuild on every upload — BM25 is an in-memory index. Fast to rebuild (milliseconds), no persistence needed.

---

### `backend/retrieval/hybrid_search.py`

**What it does:** Blends BM25 and vector search results using Reciprocal Rank Fusion.

**Key function:** `hybrid_search(query, vector_store, bm25_store, n_results, where)`

`_reciprocal_rank_fusion(vector_results, bm25_results, k=60)` — the core algorithm. For each chunk that appeared in either result list, computes `score = 1/(k + rank_vector) + 1/(k + rank_bm25)`. A chunk ranked #1 in both searches gets the maximum possible score. A chunk ranked #1 in only one search still scores well. A chunk missing from one search gets a score contribution of zero from that side.

**Why RRF over averaging raw scores:** Vector similarity scores (0.0–1.0) and BM25 scores (0 to ∞) live on completely different numerical scales. Averaging them directly would let BM25's larger numbers dominate. Ranks are always comparable: rank 1 beats rank 2 regardless of the underlying scoring algorithm. RRF normalises this problem away.

**The k=60 constant:** From the original RRF paper (Cormack et al., 2009). Higher k smooths the advantage of top-ranked results. k=60 is the empirically validated default and works well across diverse retrieval tasks.

---

### `backend/retrieval/multi_query.py`

**What it does:** Generates multiple rephrasings of a question and merges the search results.

**Key function:** `multi_query_search(question, vector_store, bm25_store, n_results)`

`_generate_rephrasings(question)` — calls Ollama with a strict JSON prompt asking for `NUM_REPHRASINGS` (3) alternative phrasings. Strips markdown code fences the model sometimes adds despite instructions. Returns `[]` on any failure — this is a best-effort enhancement.

`_merge_results(all_results, n_results)` — deduplicates by `chunk_id`. When the same chunk appears in multiple search results, keeps the version with the highest `rrf_score`. Returns top `n_results` sorted by score.

**Why this matters:** A single phrasing misses chunks that use different vocabulary. "How does position work?" might miss the chunk titled "Positional encoding" if the vector isn't similar enough. Three rephrasings cast a wider net.

**Design decisions:**
- Falls back to original query if rephrasing fails — never breaks the user-facing flow.
- Temperature 0.7 for rephrasings — higher than the main LLM (0.3) because we want diversity, not accuracy.

---

### `backend/retrieval/parent_child.py`

**What it does:** Splits chunks into small children (for retrieval) and larger parents (for LLM context).

**Key functions:**

`split_into_parent_children(chunks)` — takes standard chunks from `pdf_processor`. Splits each into child chunks of ≈100 words. Each child carries a `parent_id` pointing back to its parent. Returns `(parents, children)` as separate lists.

`ParentStore.get_many(parent_ids)` — fetches parent chunks by ID, deduplicated, preserving first-appearance order. Called after retrieval returns child chunks.

**The core insight:** Embedding a 300-word paragraph means the embedding vector averages over all the information in it — diluting the signal of any specific sentence. A 100-word child chunk has a more focused embedding. But sending 100 words to the LLM gives it too little context. Parent-child gives you both: child precision for retrieval, parent richness for generation.

---

### `backend/tools/web_search.py`

**What it does:** Fetches external information from Wikipedia and DuckDuckGo.

**Key functions:**

`_extract_core_concept(question)` — strips question prefixes ("what is", "how does", "explain") and takes the first 4 meaningful words. Converts "what is softmax in the attention formula?" to "softmax". This is what gets searched on Wikipedia — raw questions return disambiguation pages or nothing.

`search_wikipedia(query)` — hits `https://en.wikipedia.org/api/rest_v1/page/summary/{concept}`. Returns title, first 3 paragraphs of the summary, and URL. Returns `None` for disambiguation pages (not useful), 404s (concept not found), and network errors.

`search_duckduckgo(query)` — hits DuckDuckGo's Instant Answer API. Returns the "Abstract" field when present (well-known technical topics). No API key required.

`web_search(query)` — tries Wikipedia first, falls back to DuckDuckGo. Returns the single best result or None.

**Why Wikipedia over general web search:** Wikipedia articles are structured, encyclopedic, and reliably accurate for ML concepts. General web search returns blog posts, Stack Overflow, and tutorials of variable quality. For a tutor grounded in a specific paper, Wikipedia's background information is the right supplement.

---

### `backend/tools/wolfram.py`

**What it does:** Evaluates mathematical expressions using Wolfram Alpha's Short Answers API.

**Key functions:**

`is_math_question(question)` — checks for math signals: "derivative", "integral", "formula", "equation", `\frac`, `^{`, arithmetic operators. Returns True/False. Prevents unnecessary API calls on conceptual questions.

`query_wolfram(question)` — calls `https://api.wolframalpha.com/v1/result`. Returns a one-line plain English result ("0.731" for `sigmoid(1)`, "e^x(1-e^x)/(1+e^x)^2" for the derivative of sigmoid). Returns None on 501 (Wolfram couldn't interpret the query — common for conceptual questions), network errors, or missing API key.

`query_wolfram_if_math(question)` — combines detection and query. The function `enrichment.py` calls — zero API waste on non-math questions.

**Free tier limits:** 2000 queries/month. With caching (planned), this stretches to cover many sessions.

**Setup:** Create a free account at developer.wolframalpha.com. Add `WOLFRAM_APP_ID=your_key` to `backend/.env`.

---

### `backend/tools/enrichment.py`

**What it does:** Coordinates all information sources and assembles the final context block for the LLM.

**Key function:** `enrich(question, paper_chunks)`

Runs Wikipedia and Wolfram in parallel using `ThreadPoolExecutor(max_workers=2)`. Uses `as_completed()` with `timeout=8` — never waits more than 8 seconds total for all web sources combined. If either source times out, the answer continues with whatever was collected.

`_format_paper_chunks(chunks)` — wraps chunks with section/page labels. Always runs first.

`_format_web_result(result)` — wraps Wikipedia/DuckDuckGo results with their source label and URL.

`_format_wolfram_result(result)` — formats the Wolfram computation result.

`_build_enriched_context(formatted_sources)` — joins all formatted sources with `=== source name ===` separators. The LLM sees clearly labelled sections from different sources.

`extract_triples(text, source_label)` — calls Ollama with a strict JSON prompt to extract concept-relation-concept triples. Returns a list of `{subject, relation, object, source}` dicts. Used to feed Wikipedia content into the knowledge graph. Temperature 0.1 — structured extraction needs determinism.

**Returns:**
```python
{
  "context":      str,   # formatted text for LLM prompt
  "sources_used": list,  # ["paper", "wikipedia", "wolfram_alpha"]
  "triples":      list,  # concept triples for knowledge graph
}
```

---

### `backend/knowledge/graph.py`

**What it does:** Stores and queries a concept knowledge graph using NetworkX DiGraph.

**Key class:** `KnowledgeGraph`

`add_triples(triples)` — validates and adds concept triples. Normalises all concepts to lowercase for consistent storage. Preserves original casing as a `display` attribute on each node. When the same edge already exists, increments its `weight` counter and appends the new source — multiple sources for the same triple is evidence of importance.

`find_node(concept)` — tries exact normalised match first, then fuzzy match using `difflib.get_close_matches` with a 0.75 cutoff. This handles "transformers" → "transformer", "self attention" → "self-attention". Without fuzzy matching, the graph is only useful for exact queries.

`find_related_concepts(concept, depth=2)` — BFS traversal up to `depth` hops in both directions (predecessors and successors). Returns display names of all reachable concepts, limited to `MAX_RELATED_CONCEPTS` (15). Depth 1 = direct connections. Depth 2 = connections of connections.

`get_path(concept_a, concept_b)` — finds the shortest path between two concepts using `nx.shortest_path` on the undirected version of the graph. Returns the path as `(subject, relation, object)` hop tuples. Used to answer "how is X related to Y?" questions.

`save() / _load_if_exists()` — serialises the graph to JSON. Sets are converted to lists (JSON doesn't support sets). Loads automatically on instantiation if a save file exists. The graph survives server restarts without any database dependency.

**Design decisions:**
- DiGraph (directed) over Graph (undirected) — "ReLU solves vanishing gradient" has a direction. But traversal uses `to_undirected()` for discovery because knowledge connections are relevant regardless of direction.
- Lowercase normalisation — "ReLU" and "relu" must be the same node. Without normalisation, the graph fragments: separate islands of the same concept.
- JSON persistence over a graph database — zero setup, zero dependencies, survives restarts. Neo4j would be overkill for a single-user local tool.

---

### `backend/knowledge/graph_retrieval.py`

**What it does:** Uses the knowledge graph to improve retrieval quality and generate relationship explanations.

**Key functions:**

`augment_query(query, graph)` — extracts key terms from the query (removing stopwords and short tokens, adding bigrams), finds related concepts for each term via graph traversal, and appends the related concepts to the query string. The augmented query then runs through hybrid search.

Example: `"why is ReLU better?"` → graph finds `[sigmoid, vanishing gradient, slow training]` → augmented query: `"why is ReLU better? sigmoid vanishing gradient slow training"` → retrieval now finds chunks about vanishing gradients even though the user didn't mention them.

`_extract_key_terms(query)` — removes stopwords (`STOPWORDS` set of ~60 common English words), removes tokens shorter than 3 characters, generates bigrams from adjacent non-stopword words. Returns deduplicated list preserving order.

`explain_path(concept_a, concept_b, graph)` — calls `graph.get_path()` then formats the hop list as a natural language chain: "ReLU solves vanishing gradient, which causes slow training". Used when the user asks "how is X related to Y?".

`get_context_for_concept(concept, graph)` — returns a bullet-point list of all edges involving a concept, labeled with sources. Injected into the LLM prompt to give it explicit graph knowledge alongside the retrieved text.

---

## Data flow: what happens when you ask a question

```
User types: "why is ReLU better than tanh?"

1. main.py /chat receives request, validates session

2. graph_retrieval.augment_query():
   - key terms: ["relu", "relu-better", "tanh", "relu better"]
   - graph finds neighbors: ["sigmoid", "vanishing gradient", "slow training", "saturation"]
   - augmented: "why is ReLU better than tanh? sigmoid vanishing gradient slow training"

3. retrieval.multi_query_search(augmented_question):
   - LLM generates 3 rephrasings of the augmented question
   - runs hybrid_search() for each (original + 3 rephrasings = 4 searches)
   - each hybrid_search() runs BM25 + vector → RRF blend
   - _merge_results() deduplicates across 4 result sets

4. parent_store.get_many(parent_ids):
   - retrieval returned child chunk IDs
   - fetches their parent paragraphs (300 words each)
   - LLM gets rich context, not just the matching sentence

5. tools.enrichment.enrich(question, parent_chunks):
   - Wikipedia and Wolfram run in parallel (ThreadPoolExecutor)
   - Wikipedia: fetches article on "ReLU"
   - Wolfram: detects no math → skips
   - extract_triples(wiki_text) → ["ReLU → avoids → saturation", ...]
   - returns formatted context + triples

6. knowledge_graph.add_triples(web_triples):
   - new triples from Wikipedia enter the graph
   - saved to knowledge_graph.json

7. graph_retrieval.get_context_for_concept("relu", graph):
   - returns: "ReLU → solves → vanishing gradient (paper)\n  ReLU → replaces → sigmoid (paper, wikipedia)"

8. llm_client.chat():
   - builds message list: system + history + [paper + wiki + graph context + question]
   - POSTs to Ollama (llama3.2, temperature=0.3)
   - returns answer string

9. session_store.add_turn(session_id, question, answer)

10. main.py returns:
    { "answer": "...", "sources": [{"section": "3.1", "page": 3, "score": 0.94, "preview": "..."}] }
```

---

## Test suite

201 tests across 12 test files. Run all with:

```bash
pytest backend/tests/ -v
```

Run a specific file:

```bash
pytest backend/tests/test_graph.py -v
```

| File | Tests | What it covers |
|---|---|---|
| `test_pdf_processor.py` | 18 | Text cleaning, math detection, footnote detection, caption detection, chunking logic |
| `test_vector_store.py` | 19 | Chunk storage, similarity search, metadata filtering, semantic ranking |
| `test_llm_client.py` | 24 | Prompt construction, history interleaving, mocked Ollama responses, error handling |
| `test_main.py` | 26 | All HTTP endpoints, status codes, error messages, session lifecycle |
| `test_bm25_store.py` | 14 | Tokenisation, index building, keyword matching, score ordering |
| `test_hybrid_search.py` | 13 | RRF math, dual-list blending, score ordering, real in-memory integration |
| `test_multi_query.py` | 10 | Rephrasing JSON parsing, markdown fence stripping, result merging, deduplication |
| `test_web_search.py` | 14 | Wikipedia API responses, DDG fallback, disambiguation handling, connection errors |
| `test_wolfram.py` | 10 | Math detection, API call mocking, 501 handling, missing API key |
| `test_enrichment.py` | 12 | Parallel execution, source formatting, context assembly, graceful degradation |
| `test_graph.py` | 27 | Triple addition, case normalisation, fuzzy matching, BFS traversal, path finding, persistence |
| `test_graph_retrieval.py` | 14 | Stopword removal, bigram generation, query augmentation, path explanation |

**Testing philosophy used throughout:**

- External services (Ollama, Wikipedia, Wolfram, ChromaDB on disk) are always mocked. Tests run offline, in milliseconds.
- In-memory ChromaDB (`EphemeralClient`) for vector store tests — no cleanup, no disk state, no interference between tests.
- Fake block lists and fake chunk lists for testing chunking and retrieval logic — no real PDF needed.
- The most important test in the suite: `test_relevant_query_scores_higher_than_irrelevant` in `TestSearch`. If semantic similarity doesn't work, everything downstream fails.

---

## Setup

### Prerequisites

- Python 3.10+
- Ollama installed: [ollama.ai](https://ollama.ai)
- A research paper PDF

### Installation

```bash
# Clone / create the project directory
cd research_tutor

# Install dependencies
pip install -r requirements.txt

# Pull the LLM (one-time, ~2GB download)
ollama pull llama3.2

# (Optional) Get a free Wolfram Alpha API key
# https://developer.wolframalpha.com
echo "WOLFRAM_APP_ID=your_key_here" > backend/.env

# Run all tests to verify everything works
pytest backend/tests/ -v

# Start the server
uvicorn backend.main:app --reload --port 8000
```

### Verify the server

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "ok",
  "ollama_running": true,
  "chunks_stored": 0,
  "active_sessions": 0,
  "graph_nodes": 0,
  "graph_edges": 0
}
```

### Upload a paper and chat

```bash
# Upload
SESSION=$(curl -s -X POST http://localhost:8000/upload \
  -F "file=@path/to/paper.pdf" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"question\": \"what is this paper about?\"}"
```

Or use the interactive API docs at `http://localhost:8000/docs`.

---

## Configuration

All constants are at the top of their respective files for easy tuning:

| Constant | File | Default | Effect |
|---|---|---|---|
| `MAX_CHUNK_WORDS` | `pdf_processor.py` | 300 | Larger = more context per chunk, slower embedding |
| `HEADING_FONT_RATIO` | `pdf_processor.py` | 1.15 | Higher = stricter heading detection |
| `FOOTNOTE_Y_THRESHOLD` | `pdf_processor.py` | 0.88 | Higher = more footnotes discarded |
| `EMBEDDING_MODEL_NAME` | `vector_store.py` | `all-MiniLM-L6-v2` | Swap for larger model for better quality |
| `DEFAULT_N_RESULTS` | `vector_store.py` | 5 | More results = more context, slower |
| `OLLAMA_MODEL` | `llm_client.py` | `llama3.2` | Change to `llama3.2:70b` for better answers |
| `REQUEST_TIMEOUT` | `llm_client.py` | 120s | Increase for slow machines |
| `RRF_K` | `hybrid_search.py` | 60 | Lower = more weight on top ranks |
| `NUM_REPHRASINGS` | `multi_query.py` | 3 | More = better coverage, slower |
| `CHILD_MAX_WORDS` | `parent_child.py` | 100 | Smaller = more precise retrieval |
| `ENRICHMENT_TIMEOUT` | `enrichment.py` | 8s | Lower for faster response, higher for reliability |
| `MAX_HISTORY_TURNS` | `session_store.py` | 6 | More = better memory, more tokens used |
| `FUZZY_MATCH_CUTOFF` | `graph.py` | 0.75 | Lower = more fuzzy matches, higher = stricter |

---

## Build phases

### Phase 1 — Working prototype ✓
PDF upload → PyMuPDF extraction → section-based chunking → ChromaDB storage → basic semantic search → Ollama chat with history.

### Phase 2 — Better retrieval ✓
BM25 keyword search → RRF hybrid blending → multi-query rephrasing → parent-child chunking.

### Phase 3 — Web enrichment ✓
Wikipedia background → Wolfram Alpha math evaluation → parallel source fetching → triple extraction for knowledge graph → enriched LLM context.

### Phase 4 — Knowledge graph ✓
NetworkX concept graph → fuzzy node matching → BFS traversal → query augmentation → path explanation → graph persistence → web triples fed back in per turn.

### Phase 5 — Frontend (next)
Streamlit chat interface → PDF upload UI → rendered markdown + KaTeX math → source reference display → concept map visualisation → session persistence.

### Phase 6 — Agent tools
Prerequisite task queue → notation glossary auto-extraction → decision reasoning mode → analogy engine → sandboxed code execution → confusion detector.

### Phase 7 — Performance
Search result caching (SQLite) → session persistence (SQLite) → streaming responses → cross-paper comparison → adaptive reading path.

---

## Known limitations (current)

- Single user only — one vector store shared across all sessions. Two simultaneous uploads would corrupt each other. Fix: per-session vector store (Phase 5).
- No math rendering — LaTeX strings in answers display as raw text. Fix: KaTeX in frontend (Phase 5).
- No code execution — generated code can't be run. Fix: sandboxed Python subprocess (Phase 6).
- Sessions lost on restart — in-memory only. Fix: SQLite session store (Phase 7).
- Scanned PDFs fail — PyMuPDF extracts nothing from image-only PDFs. Fix: OCR with Tesseract (future).
- Knowledge graph grows unbounded — no pruning. Fix: confidence-weighted decay for low-evidence triples (future).

---

## Dependencies

```
pymupdf==1.24.5          # PDF parsing with position-aware text extraction
chromadb==0.5.3          # Local vector database with cosine similarity search
sentence-transformers==3.0.1  # all-MiniLM-L6-v2 embedding model (runs on CPU)
rank-bm25==0.2.2         # BM25Okapi keyword search
networkx==3.3            # Knowledge graph — nodes, edges, BFS, shortest path
fastapi==0.111.0         # HTTP framework — automatic /docs, Pydantic validation
uvicorn==0.30.1          # ASGI server for FastAPI
python-multipart==0.0.9  # Multipart form parsing (PDF file uploads)
requests==2.32.3         # HTTP client for Ollama, Wikipedia, Wolfram Alpha
pytest==8.2.2            # Test runner
httpx==0.27.0            # Async HTTP client for FastAPI TestClient
pytest-asyncio==0.23.7   # Async test support
```

All run locally. No cloud. No API keys required (Wolfram Alpha is optional and has a free tier).

---
