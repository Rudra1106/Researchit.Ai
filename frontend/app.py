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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── animated gradient orbs ─────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 40% at 20% 20%, rgba(124,58,237,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 50% 35% at 80% 80%, rgba(96,165,250,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 60% 10%, rgba(167,139,250,0.1) 0%, transparent 70%);
    animation: orb-drift 12s ease-in-out infinite alternate;
}
@keyframes orb-drift {
    0%   { opacity: 1; }
    100% { opacity: 0.6; transform: scale(1.05); }
}

/* ── sidebar title ──────────────────────────────────────────── */
.grad-title {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #A78BFA, #60A5FA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2; margin-bottom: 0.2rem;
}
.grad-sub { font-size: 0.8rem; color: #64748B; margin-bottom: 1.2rem; }

/* ── sidebar section headers ────────────────────────────────── */
.section-header {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #475569; margin: 1rem 0 0.4rem 0;
}

/* ── stat cards ─────────────────────────────────────────────── */
.stat-row { display: flex; gap: 0.4rem; margin-bottom: 0.6rem; }
.stat-card {
    flex: 1;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 10px; padding: 0.5rem 0.4rem; text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(167,139,250,0.5); }
.stat-num { font-size: 1.2rem; font-weight: 700; color: #A78BFA; line-height: 1; }
.stat-lbl { font-size: 0.6rem; color: #64748B; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── provider badges ────────────────────────────────────────── */
.provider-badge {
    display: inline-block; border-radius: 6px;
    padding: 0.2rem 0.6rem; font-size: 0.7rem; font-weight: 600; margin: 0.1rem 0.1rem 0 0;
}
.provider-ollama { background: rgba(34,197,94,0.1);  color: #4ADE80; border: 1px solid rgba(34,197,94,0.25); }
.provider-groq   { background: rgba(251,146,60,0.1); color: #FB923C; border: 1px solid rgba(251,146,60,0.25); }
.provider-down   { background: rgba(239,68,68,0.1);  color: #F87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── learner level badge ────────────────────────────────────── */
.level-badge {
    display: inline-block; border-radius: 6px;
    padding: 0.2rem 0.65rem; font-size: 0.7rem; font-weight: 700; margin-top: 0.3rem;
}
.level-beginner     { background: rgba(99,102,241,0.12); color: #818CF8; border: 1px solid rgba(99,102,241,0.25); }
.level-intermediate { background: rgba(234,179,8,0.12);  color: #FBBF24; border: 1px solid rgba(234,179,8,0.25); }
.level-advanced     { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── prereq pills ───────────────────────────────────────────── */
.prereq-pill {
    display: inline-block;
    background: rgba(139,92,246,0.1); color: #A78BFA;
    border: 1px solid rgba(139,92,246,0.2); border-radius: 20px;
    padding: 0.12rem 0.5rem; font-size: 0.67rem; font-weight: 500;
    margin: 0.1rem 0.15rem 0.1rem 0;
}

/* ── source pills ───────────────────────────────────────────── */
.src-pill {
    display: inline-block;
    background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.2);
    border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.7rem; color: #93C5FD;
    margin: 0.15rem 0.1rem 0 0;
}

/* ── welcome hero ───────────────────────────────────────────── */
.hero-wrap {
    display: flex; flex-direction: column; align-items: center;
    padding: 3rem 1rem 2rem; text-align: center;
}
.hero-icon {
    font-size: 3.5rem; margin-bottom: 1rem;
    filter: drop-shadow(0 0 24px rgba(167,139,250,0.6));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin-bottom: 0.75rem;
    background: linear-gradient(135deg, #E2E8F0 0%, #A78BFA 50%, #60A5FA 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 1rem; color: #94A3B8; max-width: 520px;
    line-height: 1.7; margin: 0 auto 2rem;
}
/* ── step cards ─────────────────────────────────────────────── */
.steps-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 0.85rem; width: 100%; max-width: 620px; margin: 0 auto 2rem;
}
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.1rem 1rem;
    text-align: left; transition: border-color 0.25s, background 0.25s;
    position: relative; overflow: hidden;
}
.step-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, #7C3AED, #60A5FA);
    opacity: 0; transition: opacity 0.25s;
}
.step-card:hover { border-color: rgba(167,139,250,0.3); background: rgba(124,58,237,0.06); }
.step-card:hover::before { opacity: 1; }
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.5rem; height: 1.5rem; border-radius: 6px;
    background: rgba(124,58,237,0.2); color: #A78BFA;
    font-size: 0.7rem; font-weight: 700; margin-bottom: 0.5rem;
}
.step-title { font-size: 0.85rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.2rem; }
.step-desc  { font-size: 0.75rem; color: #64748B; line-height: 1.5; }

/* ── feature chips ──────────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; max-width: 620px; }
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 0.3rem 0.9rem;
    font-size: 0.72rem; color: #94A3B8;
    transition: border-color 0.2s, color 0.2s;
}
.chip:hover { border-color: rgba(167,139,250,0.4); color: #C4B5FD; }

/* ── chat header ────────────────────────────────────────────── */
.chat-header {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.75rem 1rem; margin-bottom: 1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
}
.chat-header-icon { font-size: 1.4rem; }
.chat-header-title { font-size: 1rem; font-weight: 600; color: #E2E8F0; }
.chat-header-sub   { font-size: 0.72rem; color: #64748B; }

/* ── premium sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(13,8,32,0.97) 0%,
        rgba(9,6,24,0.99) 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.18) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

/* ── sidebar upload button ──────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(96,165,250,0.15)) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
    color: #C4B5FD !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.4), rgba(96,165,250,0.25)) !important;
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 16px rgba(124,58,237,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── sidebar file uploader ──────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(124,58,237,0.08) !important;
    border: 1px dashed rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.5) !important;
}

/* ── sidebar divider ────────────────────────────────────────── */
[data-testid="stSidebar"] hr {
    border-color: rgba(124,58,237,0.15) !important;
    margin: 0.8rem 0 !important;
}

/* ── sidebar success / error boxes ─────────────────────────── */
[data-testid="stSidebar"] .stSuccess {
    background: rgba(34,197,94,0.08) !important;
    border: 1px solid rgba(34,197,94,0.2) !important;
    border-radius: 10px !important;
}

/* ── mermaid diagram wrapper ────────────────────────────────── */
.mermaid-wrap {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.75rem 0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)



# ── Session state defaults ─────────────────────────────────────────────────────

defaults = {
    "session_id":          None,
    "messages":            [],
    "paper_info":          None,
    "backend_url":         os.getenv("BACKEND_URL", "http://localhost:8000"),
    "health":              None,
    "learner_profile":     {"level": "unknown", "taught": []},
    "prerequisites_cache": {},   # msg_index -> list of prereqs taught
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
import re as _re
import streamlit.components.v1 as _components

def render_message(text: str):
    """
    Render a message that may contain ```mermaid``` blocks.
    - Mermaid blocks → live diagram via Mermaid.js
    - Everything else  → st.markdown (supports LaTeX $...$ / $$...$$)
    """
    pattern = r"```mermaid\s*([\s\S]*?)```"
    parts   = _re.split(pattern, text)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Normal markdown + LaTeX
            if part.strip():
                st.markdown(part)
        else:
            # Mermaid diagram
            diagram_def = part.strip()
            html = f"""
            <div class="mermaid-wrap">
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'dark',
                themeVariables:{{primaryColor:'#7C3AED',primaryTextColor:'#E2E8F0',
                lineColor:'#A78BFA',background:'transparent'}}}});</script>
            <div class="mermaid">{diagram_def}</div>
            </div>
            """
            _components.html(html, height=320, scrolling=False)


def send_chat(question):

    data, err = api("post", "/chat", json={
        "session_id": st.session_state.session_id,
        "question":   question,
    })
    return data, err


def clear_session():
    if st.session_state.session_id:
        api("delete", f"/session/{st.session_state.session_id}")
    st.session_state.session_id      = None
    st.session_state.messages        = []
    st.session_state.paper_info      = None
    st.session_state.learner_profile = {"level": "unknown", "taught": []}


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
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-num">{chunks}</div><div class="stat-lbl">Chunks</div></div>
          <div class="stat-card"><div class="stat-num">{nodes}</div><div class="stat-lbl">Graph nodes</div></div>
          <div class="stat-card"><div class="stat-num">{edges}</div><div class="stat-lbl">Graph edges</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Learner profile ─────────────────────────────────────────────────────────
    profile = st.session_state.learner_profile
    level   = profile.get("level", "unknown")
    taught  = profile.get("taught", [])
    if level != "unknown":
        level_icons = {"beginner": "🌱", "intermediate": "📚", "advanced": "🔬"}
        icon = level_icons.get(level, "🎓")
        st.markdown(
            f'<span class="level-badge level-{level}">{icon} {level.capitalize()}</span>',
            unsafe_allow_html=True,
        )
        if taught:
            st.caption(f"Covered this session: {', '.join(taught[:6])}"
                       + (" +more" if len(taught) > 6 else ""))


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
                st.session_state.session_id      = data["session_id"]
                st.session_state.messages        = []
                st.session_state.learner_profile = {"level": "unknown", "taught": []}
                st.session_state.paper_info      = {
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
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-icon">🎓</div>
      <div class="hero-title">Your AI Research Mentor</div>
      <div class="hero-sub">
        Upload any research paper and get Socratic, structured explanations —
        grounded in the paper's text, enriched with Wikipedia &amp; Wolfram Alpha,
        and calibrated to <em>your</em> level.
      </div>

      <div class="steps-grid">
        <div class="step-card">
          <div class="step-num">1</div>
          <div class="step-title">Upload your PDF</div>
          <div class="step-desc">Any research paper — ArXiv, NeurIPS, Nature. Up to 200 MB.</div>
        </div>
        <div class="step-card">
          <div class="step-num">2</div>
          <div class="step-title">We build the knowledge graph</div>
          <div class="step-desc">Chunks, embeddings, concept triples — all in ~15 seconds.</div>
        </div>
        <div class="step-card">
          <div class="step-num">3</div>
          <div class="step-title">Ask anything</div>
          <div class="step-desc">From "what is this paper about?" to deep equation dives.</div>
        </div>
        <div class="step-card">
          <div class="step-num">4</div>
          <div class="step-title">Learn, not just read</div>
          <div class="step-desc">Analogy → Intuition → Math → Code → Socratic check, every time.</div>
        </div>
      </div>

      <div class="chip-row">
        <span class="chip">🔍 Hybrid BM25 + vector search</span>
        <span class="chip">🕸 Knowledge graph</span>
        <span class="chip">🌐 Wikipedia enrichment</span>
        <span class="chip">∫ LaTeX math rendering</span>
        <span class="chip">🧠 Learner profiling</span>
        <span class="chip">⚡ Groq · Ollama</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat interface ─────────────────────────────────────────────────────────
    info = st.session_state.paper_info or {}
    fname = info.get("filename", "Paper")
    chunks = info.get("chunk_count", 0)
    st.markdown(
        f"""<div class="chat-header">
          <div class="chat-header-icon">📄</div>
          <div>
            <div class="chat-header-title">{fname}</div>
            <div class="chat-header-sub">{chunks} chunks indexed · Ask anything below</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


    # Display all past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            # Show prereq pills for assistant messages that taught prerequisites
            if msg["role"] == "assistant" and msg.get("prereqs"):
                pills = "".join(
                    f'<span class="prereq-pill">✓ {p}</span>' for p in msg["prereqs"]
                )
                st.markdown(
                    f'<div style="margin-bottom:0.4rem;">'
                    f'<span style="font-size:0.68rem;color:#64748B;">Prerequisites covered first: </span>'
                    f'{pills}</div>',
                    unsafe_allow_html=True,
                )

            render_message(msg["content"])

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

        # Call backend
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking…"):
                data, err = send_chat(prompt)

            if err:
                answer  = f"⚠ **Error:** {err}"
                sources = []
                prereqs = []
                st.error(answer)
            else:
                answer  = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                prereqs = data.get("prerequisites_taught", [])

                # Update learner profile in sidebar
                lp = data.get("learner_profile", {})
                if lp:
                    st.session_state.learner_profile = lp

                # Show prereq pills if concepts were taught first
                if prereqs:
                    pills = "".join(
                        f'<span class="prereq-pill">✓ {p}</span>' for p in prereqs
                    )
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;">'
                        f'<span style="font-size:0.68rem;color:#64748B;">Prerequisites covered first: </span>'
                        f'{pills}</div>',
                        unsafe_allow_html=True,
                    )

                render_message(answer)

                if sources:
                    with st.expander(
                        f"📚 {len(sources)} source{'s' if len(sources) > 1 else ''} cited",
                        expanded=False,
                    ):
                        for src in sources:
                            score_pct = int(src.get("score", 0) * 100)
                            bar = "█" * (score_pct // 10) + "░" * (10 - score_pct // 10)
                            st.markdown(
                                f"**§ {src['section']}** — page {src['page']}"
                                f"  `{bar} {score_pct}%`",
                            )
                            st.caption(src.get("preview", ""))

        # Save to session (include prereqs so history can render them)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "prereqs": prereqs if not err else [],
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
