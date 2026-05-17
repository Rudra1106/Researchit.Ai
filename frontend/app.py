"""
app.py  —  Streamlit frontend for the Research Paper AI Tutor

Run from the repo root:
    streamlit run frontend/app.py

The app talks to the FastAPI backend.
Set BACKEND_URL env var or edit it in the sidebar.
Default: http://localhost:8000
"""

import os
import time
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- gradient title ---------- */
.grad-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED, #A78BFA, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.grad-sub {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-bottom: 1.5rem;
}

/* ---------- sidebar section headers ---------- */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
    margin: 1.2rem 0 0.5rem 0;
}

/* ---------- stat cards in sidebar ---------- */
.stat-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.stat-card {
    flex: 1;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    text-align: center;
}
.stat-num  { font-size: 1.3rem; font-weight: 700; color: #A78BFA; line-height: 1; }
.stat-lbl  { font-size: 0.65rem; color: #94A3B8; margin-top: 0.2rem; }

/* ---------- source pills ---------- */
.src-pill {
    display: inline-block;
    background: rgba(96, 165, 250, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    color: #93C5FD;
    margin: 0.15rem 0.15rem 0 0;
}

/* ---------- welcome card ---------- */
.welcome-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(96,165,250,0.08));
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin: 2rem auto;
    max-width: 620px;
    text-align: center;
}
.welcome-card h2 { color: #E2E8F0; margin-bottom: 0.5rem; }
.welcome-card p  { color: #94A3B8; font-size: 0.9rem; line-height: 1.6; }
.step-list { text-align: left; margin: 1.5rem 0 0 0; list-style: none; padding: 0; }
.step-list li {
    color: #CBD5E1;
    font-size: 0.88rem;
    padding: 0.4rem 0;
    padding-left: 1.8rem;
    position: relative;
}
.step-list li::before {
    content: attr(data-n);
    position: absolute; left: 0;
    background: rgba(124,58,237,0.3);
    color: #A78BFA;
    border-radius: 50%;
    width: 1.2rem; height: 1.2rem;
    font-size: 0.7rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
}

/* ---------- feature chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; margin-top: 1.5rem; }
.chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
    color: #CBD5E1;
}

/* ---------- message source expander ---------- */
.sources-label {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ---------- provider badge in sidebar ---------- */
.provider-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
}
.provider-ollama { background: rgba(34,197,94,0.15);  color: #4ADE80; border: 1px solid rgba(34,197,94,0.3); }
.provider-groq   { background: rgba(251,146,60,0.15); color: #FB923C; border: 1px solid rgba(251,146,60,0.3); }
.provider-down   { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

/* ---------- scrollable chat area ---------- */
.chat-container { max-height: 70vh; overflow-y: auto; padding-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "session_id":   None,
    "messages":     [],        # [{"role": "user"|"assistant", "content": str, "sources": list}]
    "paper_info":   None,      # {"filename": str, "chunk_count": int, "sections": list}
    "backend_url":  os.getenv("BACKEND_URL", "http://localhost:8000"),
    "health":       None,      # cached health dict
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── API helpers ────────────────────────────────────────────────────────────────

def api(method, path, **kwargs):
    """Thin wrapper around requests. Returns (data_dict_or_None, error_str_or_None)."""
    url = f"{st.session_state.backend_url}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=300, **kwargs)
        if resp.status_code in (200, 201):
            try:
                return resp.json(), None
            except Exception:
                return {}, None
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend. Is `uvicorn backend.main:app --port 8000` running?"
    except requests.exceptions.Timeout:
        return None, "Request timed out — the model is still thinking. Try again."
    except Exception as e:
        return None, str(e)


def fetch_health():
    data, err = api("get", "/health")
    st.session_state.health = data if data else {}
    return st.session_state.health, err


def upload_pdf(file_bytes, filename):
    data, err = api("post", "/upload", files={"file": (filename, file_bytes, "application/pdf")})
    return data, err


def send_chat(question):
    data, err = api("post", "/chat", json={
        "session_id": st.session_state.session_id,
        "question":   question,
    })
    return data, err


def clear_session():
    if st.session_state.session_id:
        api("delete", f"/session/{st.session_state.session_id}")
    st.session_state.session_id = None
    st.session_state.messages   = []
    st.session_state.paper_info = None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="grad-title">🎓 Research Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-sub">AI-powered paper reading assistant</div>', unsafe_allow_html=True)

    # ── Backend health ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

    health, h_err = fetch_health()
    if h_err:
        st.markdown('<span class="provider-badge provider-down">⚠ Backend offline</span>',
                    unsafe_allow_html=True)
        st.caption(h_err)
    else:
        groq_key  = health.get("groq_available", False)
        ollama_ok = health.get("ollama_running",  False)
        if groq_key:
            badge = '<span class="provider-badge provider-groq">⚡ Groq (primary)</span>'
            if ollama_ok:
                badge += '&nbsp;<span class="provider-badge provider-ollama">● Ollama (fallback)</span>'
        elif ollama_ok:
            badge = '<span class="provider-badge provider-ollama">● Ollama (only)</span>'
        else:
            badge = '<span class="provider-badge provider-down">⚠ No LLM — add GROQ_API_KEY</span>'
        st.markdown(badge, unsafe_allow_html=True)

    # ── Knowledge graph stats ──────────────────────────────────────────────────
    if health:
        nodes  = health.get("graph_nodes",  0)
        edges  = health.get("graph_edges",  0)
        chunks = health.get("chunks_stored", 0)
        sess   = health.get("active_sessions", 0)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-num">{chunks}</div><div class="stat-lbl">Chunks</div></div>
          <div class="stat-card"><div class="stat-num">{nodes}</div><div class="stat-lbl">Graph nodes</div></div>
          <div class="stat-card"><div class="stat-num">{edges}</div><div class="stat-lbl">Graph edges</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Upload Paper</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded is not None:
        current_name = (st.session_state.paper_info or {}).get("filename")
        if uploaded.name != current_name:
            with st.spinner(f"Processing **{uploaded.name}**…"):
                data, err = upload_pdf(uploaded.read(), uploaded.name)
            if err:
                st.error(err)
            else:
                st.session_state.session_id = data["session_id"]
                st.session_state.messages   = []
                st.session_state.paper_info = {
                    "filename":    data.get("filename",    uploaded.name),
                    "chunk_count": data.get("chunk_count", 0),
                    "sections":    data.get("sections",    []),
                }
                st.rerun()

    # ── Session info ───────────────────────────────────────────────────────────
    if st.session_state.paper_info:
        info = st.session_state.paper_info
        st.success(f"📄 **{info['filename']}**")
        st.caption(f"{info['chunk_count']} chunks · {len(info['sections'])} sections")

        with st.expander("Sections found", expanded=False):
            for s in info["sections"]:
                st.markdown(f"• {s}")

        if st.button("🗑 New paper", use_container_width=True, type="secondary"):
            clear_session()
            st.rerun()

    # ── Settings ───────────────────────────────────────────────────────────────
    with st.expander("⚙ Settings", expanded=False):
        new_url = st.text_input("Backend URL", value=st.session_state.backend_url)
        if new_url != st.session_state.backend_url:
            st.session_state.backend_url = new_url
            st.rerun()
        st.caption("Run: `uvicorn backend.main:app --reload --port 8000`")


# ── Main content ───────────────────────────────────────────────────────────────

if st.session_state.session_id is None:
    # ── Welcome screen ─────────────────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
        <div class="welcome-card">
          <h2>Welcome to Research Paper Tutor</h2>
          <p>Upload any research paper PDF and have an in-depth conversation about it.
             Every answer is grounded in the paper's own text, enriched with Wikipedia
             and Wolfram Alpha, and linked through a knowledge graph.</p>
          <ul class="step-list">
            <li data-n="1">Upload your PDF using the sidebar panel on the left</li>
            <li data-n="2">Wait ~10s while we chunk, embed, and build the knowledge graph</li>
            <li data-n="3">Ask any question — from "what is this paper about?" to deep dives</li>
            <li data-n="4">Explore source references and graph context below each answer</li>
          </ul>
          <div class="chip-row">
            <span class="chip">🔍 Hybrid BM25 + vector search</span>
            <span class="chip">🕸 Knowledge graph</span>
            <span class="chip">🌐 Wikipedia enrichment</span>
            <span class="chip">∫ Math rendering</span>
            <span class="chip">🦙 Ollama · Groq</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Chat interface ─────────────────────────────────────────────────────────
    info = st.session_state.paper_info or {}
    st.markdown(
        f"<h3 style='margin:0 0 1rem 0; color:#E2E8F0;'>💬 {info.get('filename', 'Chat')}</h3>",
        unsafe_allow_html=True,
    )

    # Display all past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            # Render with LaTeX support
            st.markdown(msg["content"])

            # Source references (assistant only)
            sources = msg.get("sources", [])
            if msg["role"] == "assistant" and sources:
                with st.expander(f"📚 {len(sources)} source{'s' if len(sources) > 1 else ''} cited", expanded=False):
                    for src in sources:
                        score_pct = int(src.get("score", 0) * 100)
                        bar = "█" * (score_pct // 10) + "░" * (10 - score_pct // 10)
                        st.markdown(
                            f"**§ {src['section']}** — page {src['page']}"
                            f"  `{bar} {score_pct}%`",
                        )
                        st.caption(src.get("preview", ""))

    # ── Chat input ─────────────────────────────────────────────────────────────
    placeholder = "Ask anything about the paper… (supports follow-up questions)"
    if prompt := st.chat_input(placeholder):
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Call backend and stream the response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Retrieving · Enriching · Answering…"):
                data, err = send_chat(prompt)

            if err:
                answer  = f"⚠ **Error:** {err}"
                sources = []
                st.error(answer)
            else:
                answer  = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                # Render answer (LaTeX via $...$ / $$...$$)
                st.markdown(answer)
                if sources:
                    with st.expander(f"📚 {len(sources)} source{'s' if len(sources) > 1 else ''} cited", expanded=False):
                        for src in sources:
                            score_pct = int(src.get("score", 0) * 100)
                            bar = "█" * (score_pct // 10) + "░" * (10 - score_pct // 10)
                            st.markdown(
                                f"**§ {src['section']}** — page {src['page']}"
                                f"  `{bar} {score_pct}%`",
                            )
                            st.caption(src.get("preview", ""))

        # Save to session state
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })

    # ── Suggested questions (shown when just uploaded, no turns yet) ───────────
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown(
            "<p style='color:#64748B; font-size:0.85rem; margin-bottom:0.75rem;'>"
            "💡 Try asking one of these to get started:</p>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "What is this paper about? Explain it to a complete beginner.",
            "What problem does this paper solve, and why was it hard?",
            "Walk me through the key equations in this paper.",
            "What are the main results and why are they significant?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, use_container_width=True, key=f"sug_{i}"):
                # Inject as a chat input
                st.session_state.messages.append({"role": "user", "content": q, "sources": []})
                st.rerun()
