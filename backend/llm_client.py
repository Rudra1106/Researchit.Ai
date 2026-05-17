"""
llm_client.py

Provider priority:
  1. Ollama (local, free, private) — always tried first
  2. Groq  (cloud, free tier)      — auto-fallback if Ollama unreachable

Configure in backend/.env:
  GROQ_API_KEY  = gsk_...                  (enables fallback)
  OLLAMA_MODEL  = llama3.2                 (optional)
  GROQ_MODEL    = llama-3.3-70b-versatile  (optional)

Public API (unchanged for callers):
  chat(question, chunks, history, extra_context) → str
  check_ollama_running()                          → bool
  llm_complete(messages, temperature)             → str  ← new, used by enrichment + multi_query
"""

import os
import requests
from dotenv import load_dotenv

# Load .env from the backend directory regardless of cwd
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "phi3:3.8b")
REQUEST_TIMEOUT = 120

# ── Groq (OpenAI-compatible) ───────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research paper tutor helping someone new to machine learning and AI.

Your job is to explain concepts from the research paper clearly and from scratch.

Rules you ALWAYS follow:
1. Assume the user is a beginner unless they show otherwise. Never skip steps.
2. Format ALL mathematical expressions using LaTeX:
   - Inline math: $...$ (e.g. $\\frac{1}{n}$, $W^T x + b$)
   - Display math: $$...$$ on its own line for important equations
   Never write raw LaTeX commands without wrapping them — the frontend renders them.
3. After explaining theory, always give a concrete real-world analogy or example.
4. When relevant, show a minimal Python code example (10–20 lines max).
5. If the user seems confused, try a completely different explanation approach.
6. Only use information from the provided paper sections. If something isn't in the paper, say so.
7. Be encouraging. Learning research papers is genuinely hard.

You have access to specific sections from the paper. Use them as your source of truth."""


# ── Core LLM call ──────────────────────────────────────────────────────────────

def llm_complete(messages, temperature=0.3, timeout=REQUEST_TIMEOUT):
    """
    Single call-point for ALL LLM calls in the system.

    Priority:
      1. Groq (cloud, instant) — used if GROQ_API_KEY is set
      2. Ollama (local)        — fallback when Groq unavailable / no key

    Args:
        messages:    list of {"role": ..., "content": ...} dicts
        temperature: 0.0–1.0
        timeout:     seconds (use short values for best-effort calls like
                     rephrasing; full value for the final answer)
    Returns:
        str — the model's response text
    Raises:
        ConnectionError / TimeoutError
    """
    # 1. Groq — fast cloud, use when API key is configured ────────────────────
    if GROQ_API_KEY:
        try:
            resp = requests.post(
                GROQ_API_URL,
                json={
                    "model":       GROQ_MODEL,
                    "messages":    messages,
                    "temperature": temperature,
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                timeout=min(timeout, 60),  # Groq is fast; cap at 60s
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"[llm] Groq ({GROQ_MODEL}) OK.")
            return content
        except requests.exceptions.Timeout:
            print("[llm] Groq timed out — falling back to Ollama.")
        except Exception as exc:
            print(f"[llm] Groq failed ({exc}) — falling back to Ollama.")

    # 2. Ollama — local fallback (or primary when no Groq key) ────────────────
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   False,
                "options":  {"temperature": temperature},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            print(f"[llm] Ollama ({OLLAMA_MODEL}) OK.")
            return resp.json()["message"]["content"]
        raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:100]}")
    except requests.exceptions.ConnectionError:
        pass  # Ollama not running
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Both Groq and Ollama timed out (>{timeout}s). Try a shorter question."
        )
    except RuntimeError as exc:
        raise ConnectionError(str(exc)) from exc

    raise ConnectionError(
        "No LLM available. Add GROQ_API_KEY to backend/.env "
        "or start Ollama (`ollama serve`)."
    )


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _format_chunks_as_context(chunks):
    """Turn retrieved chunks into a labeled context block for the prompt."""
    if not chunks:
        return "No relevant sections found in the paper."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Excerpt {i} — Section: {chunk['section']}, Page: {chunk['page']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _build_messages(question, chunks, history, extra_context=""):
    """Assemble the full message list: system → history → current turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_q, past_a in history:
        messages.append({"role": "user",      "content": past_q})
        messages.append({"role": "assistant", "content": past_a})

    context = _format_chunks_as_context(chunks)
    user_msg = f"Here are the relevant sections from the paper:\n\n{context}"
    if extra_context:
        user_msg += f"\n\n---\n\nAdditional context from external sources:\n\n{extra_context}"
    user_msg += f"\n\n---\n\nMy question: {question}"

    messages.append({"role": "user", "content": user_msg})
    return messages


# ── Public API ─────────────────────────────────────────────────────────────────

def chat(question, chunks, history=None, extra_context=""):
    """
    Ask the LLM a question, grounded in retrieved paper chunks.
    Internally delegates to llm_complete() (Ollama → Groq fallback).
    """
    if history is None:
        history = []
    messages = _build_messages(question, chunks, history, extra_context)
    return llm_complete(messages, temperature=0.3)


def check_ollama_running():
    """Returns True if Ollama is up and reachable."""
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_available_models():
    """Return model names currently downloaded in Ollama."""
    try:
        data = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []