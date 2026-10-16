import time
from datetime import datetime

import streamlit as st
from rag_pipeline import create_rag_components, ask_question

# ---------------------------------------------------------------- Page config
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- Theme (CSS)
st.markdown(
    """
    <style>
    :root {
        --primary: #4F46E5;
        --primary-dark: #3730A3;
        --bg-card: #FFFFFF;
        --border: #E5E7EB;
        --muted: #6B7280;
    }
    .block-container { padding-top: 2rem; max-width: 1000px; }

    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; color: white; }
    .hero p  { margin: .4rem 0 0; opacity: .9; font-size: 1rem; }

    .source-chip {
        display: inline-block;
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        padding: 4px 12px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .meta { color: var(--muted); font-size: 0.8rem; margin-top: .3rem; }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
    }
    div.stButton > button[kind="primary"] {
        background: var(--primary);
        border: none;
    }
    div.stButton > button[kind="primary"]:hover { background: var(--primary-dark); }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Header
st.markdown(
    """
    <div class="hero">
        <h1>✦ DocuMind AI</h1>
        <p>RAG-Based Document Question Answering System</p>
        <p>Explore your research PDFs, get context-grounded answers,
        and verify information with page-level sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- State
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, sources, secs, time
if "pending" not in st.session_state:
    st.session_state.pending = None


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def load_rag_components():
    return create_rag_components()


def unique_sources(sources):
    """Deduplicate sources while keeping order."""
    seen, result = set(), []
    for s in sources or []:
        label = f"{s['source']} · page {s['page']}"
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def render_entry(entry):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(entry["question"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(entry["answer"])
        chips = unique_sources(entry["sources"])
        if chips:
            with st.expander(f"📚 Sources ({len(chips)})", expanded=False):
                st.markdown(
                    "".join(f'<span class="source-chip">{c}</span>' for c in chips),
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No relevant sources were retrieved.")
        st.markdown(
            f'<div class="meta">⏱ {entry["secs"]:.1f}s · {entry["time"]}</div>',
            unsafe_allow_html=True,
        )


def export_chat():
    lines = ["# DocuMind AI: Chat Export\n"]
    for e in st.session_state.history:
        lines.append(f"## Q: {e['question']}\n")
        lines.append(f"{e['answer']}\n")
        srcs = unique_sources(e["sources"])
        if srcs:
            lines.append("**Sources:** " + "; ".join(srcs) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("✦ DocuMind AI")
    st.caption("Powered by RAG · ChromaDB · Groq")

    st.subheader("💡 Try an example")
    examples = [
        "What is retrieval-augmented generation (RAG)?",
        "Explain the difference between supervised and unsupervised learning.",
        "What is overfitting and how can it be prevented?",
        "How do transformers work?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    st.metric("Questions asked", len(st.session_state.history))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    with col2:
        st.download_button(
            "⬇ Export",
            data=export_chat(),
            file_name="chat_export.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.history,
        )

# ---------------------------------------------------------------- Main
try:
    vectorstore, llm = load_rag_components()
except Exception as error:
    st.error(f"Unable to start the chatbot: {error}")
    st.info(
        "Check your .env file, Groq API key, and vector store. "
        "If necessary, run build_vectorstore.py first."
    )
    st.stop()

st.success("Knowledge base connected.", icon="✅")

# Show previous conversation
for entry in st.session_state.history:
    render_entry(entry)

if not st.session_state.history:
    st.info("👋 Ask anything about AI & ML below, or pick an example from the sidebar.")

# Input (chat box or sidebar example)
typed = st.chat_input("Ask a question about AI & Machine Learning...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question and question.strip():
    start = time.time()
    try:
        with st.spinner("Searching documents and generating an answer..."):
            answer, sources = ask_question(question.strip(), vectorstore, llm)
        entry = {
            "question": question.strip(),
            "answer": answer,
            "sources": sources,
            "secs": time.time() - start,
            "time": datetime.now().strftime("%I:%M %p"),
        }
        st.session_state.history.append(entry)
        render_entry(entry)
        st.rerun()
    except Exception as error:
        st.error(f"Something went wrong while answering: {error}")