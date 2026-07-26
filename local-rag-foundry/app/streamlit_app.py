"""Streamlit UI for the local RAG assistant.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.chat_engine import ChatEngine
from src.chunker import document_to_chunks
from src.embedder import LocalEmbedder
from src.foundry_client import FoundryClient
from src.security import UploadRejected, validate_upload
from src.vector_store import VectorStore

st.set_page_config(page_title="Local RAG Assistant", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Loading local model…")
def get_model() -> FoundryClient:
    client = FoundryClient()
    client.init()
    return client


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> LocalEmbedder:
    embedder = LocalEmbedder()
    embedder.init()
    return embedder


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            line = f"- **{s['title']}** ({s['category']}) — relevance {round(s['score'] * 100)}%"
            if s.get("semantic_score") is not None:
                line += f"  _(TF-IDF {round(s['tfidf_score'] * 100)}%, semantic {round(s['semantic_score'] * 100)}%)_"
            st.markdown(line)


store = get_store()
model = get_model()
embedder = get_embedder()
engine = ChatEngine(store, model, embedder=embedder)

if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"|"assistant", "content": str, "sources": [...]}]

# --- Sidebar: status + knowledge base management ---
with st.sidebar:
    st.header("Local RAG Assistant")
    if model.ready:
        st.success(model.message)
    else:
        st.warning(model.message)
    if embedder.ready:
        st.caption(f"🔎 Hybrid retrieval (TF-IDF + semantic) · {embedder.model_id}")
    else:
        st.caption("🔎 TF-IDF retrieval only (semantic embeddings unavailable)")
    st.caption("Fully offline — no cloud, no API keys, no outbound network calls.")

    st.divider()
    st.subheader("Knowledge base")
    docs = store.list_documents()
    st.caption(f"{len(docs)} document(s), {store.count()} chunk(s) indexed")
    for d in docs:
        st.text(f"• {d['title']}  ({d['chunks']} chunk(s))")

    st.divider()
    st.subheader("Add a document")
    uploaded = st.file_uploader("Markdown or text file", type=["md", "markdown", "txt"])
    if uploaded is not None:
        try:
            safe_name = validate_upload(uploaded.name, uploaded.size)
            raw_text = uploaded.getvalue().decode("utf-8")
            chunks = document_to_chunks(raw_text, safe_name)
            if not chunks:
                st.error("The file has no readable content.")
            else:
                embeddings = embedder.embed([c.content for c in chunks]) if embedder.ready else None
                store.remove_document(chunks[0].doc_id)
                store.add_chunks(chunks, embeddings=embeddings)
                st.success(f"Indexed '{chunks[0].title}' ({len(chunks)} chunk(s)).")
                st.rerun()
        except UploadRejected as err:
            st.error(str(err))
        except UnicodeDecodeError:
            st.error("File must be UTF-8 encoded text.")

# --- Main chat area ---
st.title("🛠️ Local Support Assistant")
st.caption("Ask a question grounded in the documents on the left. Answers cite their sources.")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        render_sources(turn.get("sources") or [])

question = st.chat_input("Describe the issue or ask a question…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources: list[dict] = []
        for event in engine.ask(question, st.session_state.history[:-1]):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                full_answer += event["text"]
                placeholder.markdown(full_answer + "▌")
            elif event["type"] == "error":
                full_answer = f"⚠️ {event['message']}"
        placeholder.markdown(full_answer)
        render_sources(sources)

    st.session_state.history.append({"role": "assistant", "content": full_answer, "sources": sources})
