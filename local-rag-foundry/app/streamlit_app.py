"""Streamlit UI for the local RAG assistant.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import config
from src.chat_engine import ChatEngine
from src.chunker import document_to_chunks
from src.document_readers import extract_text
from src.reranker import CrossEncoderReranker
from src.security import UploadRejected, validate_upload
from src.translator import LocalTranslator
from src.vector_store import VectorStore

st.set_page_config(page_title="Local Safety Assistant", page_icon="🛡️", layout="wide")

_HEADER_HTML = """
<style>
  /* Streamlit's own top toolbar (hamburger menu + "Deploy" button) is a
     fixed white bar that sits on top of the page and was covering our
     custom header below it. We don't use Deploy/sharing from this app, so
     hide the whole toolbar rather than fight its z-index/padding. */
  [data-testid="stHeader"] { display: none; }
  [data-testid="stToolbar"] { display: none; }
  .block-container { padding-top: 1.6rem; }
  #ms-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 1.5rem; padding-bottom: 1rem; margin-bottom: 0.4rem;
    border-bottom: 1px solid #E1DFDD;
  }
  #ms-header .title-block h1 {
    margin: 0; font-size: 1.65rem; font-weight: 700; color: #201F1E;
    letter-spacing: -0.01em;
  }
  #ms-header .title-block p {
    margin: 0.15rem 0 0; font-size: 0.92rem; color: #605E5C;
  }
  #ms-badge {
    display: flex; align-items: center; gap: 0.65rem; flex-shrink: 0;
  }
  #ms-logo-grid {
    display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
    gap: 2px; width: 30px; height: 30px; flex-shrink: 0;
  }
  #ms-logo-grid span { display: block; width: 100%; height: 100%; }
  #ms-badge-text { line-height: 1.25; text-align: right; }
  #ms-badge-text .role { font-weight: 600; font-size: 0.86rem; color: #201F1E; }
  #ms-badge-text .org { font-size: 0.8rem; color: #0078D4; font-weight: 600; }
  #ms-badge-text .program { font-size: 0.72rem; color: #8A8886; }
  #reliability-strip {
    display: flex; flex-wrap: wrap; gap: 1.1rem; margin: 0.6rem 0 1.3rem;
    font-size: 0.76rem; color: #605E5C;
  }
  #reliability-strip b { color: #201F1E; }
</style>
<div id="ms-header">
  <div class="title-block">
    <h1>🛡️ Local Safety Assistant</h1>
    <p>Fully offline, multilingual RAG over industrial safety procedures — ask in any language.</p>
  </div>
  <div id="ms-badge">
    <div id="ms-logo-grid">
      <span style="background:#F25022;"></span><span style="background:#7FBA00;"></span>
      <span style="background:#00A4EF;"></span><span style="background:#FFB900;"></span>
    </div>
    <div id="ms-badge-text">
      <div class="role">AI Engineer Intern</div>
      <div class="org">Microsoft</div>
      <div class="program">AI Innovations Internship Program</div>
    </div>
  </div>
</div>
<div id="reliability-strip">
  <span>✅ <b>94%</b> accuracy — 92/98 live-audited questions</span>
  <span>🔒 <b>0</b> hallucinations / wrong citations</span>
  <span>🌐 EN · TR · mixed-language</span>
  <span>🖥️ 100% local — no cloud, no API calls</span>
</div>
"""

RESPONSE_LANGUAGES = ["Auto", "Turkish", "English"]
_BINARY_UPLOAD_SUFFIXES = (".pdf", ".docx")

# Four one-click example questions -- two English, two Turkish, chosen from
# the live 98-question audit (TEST_REPORT.md §15-19) as reliably well-
# answered, so a first-time visitor immediately sees a good result rather
# than one of the documented edge cases.
EXAMPLE_QUESTIONS = [
    "How do I detect a gas leak?",
    "Boru hattını hangi metal korur?",
    "What PPE is required in H2S areas?",
    "Sıcak iş izni almadan önce gaz testi neden yapılmalı?",
]


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Loading local model…")
def get_model():
    if config.LLM_PROVIDER == "ollama":
        from src.ollama_client import OllamaClient
        client = OllamaClient()
    else:
        from src.foundry_client import FoundryClient
        client = FoundryClient()
    client.init()
    return client


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    if config.LLM_PROVIDER == "ollama":
        from src.ollama_embedder import OllamaEmbedder
        embedder = OllamaEmbedder()
    else:
        from src.embedder import LocalEmbedder
        embedder = LocalEmbedder()
    embedder.init()
    return embedder


@st.cache_resource(show_spinner="Loading reranker model…")
def get_reranker() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    reranker.init()
    return reranker


@st.cache_resource(show_spinner="Loading translation model…")
def get_translator() -> LocalTranslator:
    translator = LocalTranslator()
    translator.init()
    return translator


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            line = f"- **{s['title']}** ({s['category']}) — relevance {round(s['score'] * 100)}%"
            if s.get("semantic_score") is not None:
                line += f"  _(BM25 {round(s['bm25_score'] * 100)}%, semantic {round(s['semantic_score'] * 100)}%)_"
            if s.get("rerank_score") is not None:
                line += f"  _(reranked {round(s['rerank_score'], 2)})_"
            st.markdown(line)


def read_uploaded_text(filename: str, raw_bytes: bytes) -> str:
    """Decode markdown/text uploads directly; extract PDF/DOCX uploads via document_readers."""
    if Path(filename).suffix.lower() in _BINARY_UPLOAD_SUFFIXES:
        return extract_text(filename, raw_bytes)
    return raw_bytes.decode("utf-8")


store = get_store()
model = get_model()
embedder = get_embedder()
reranker = get_reranker()
translator = get_translator()
engine = ChatEngine(store, model, embedder=embedder, reranker=reranker, translator=translator)

if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"|"assistant", "content": str, "sources": [...]}]

# --- Sidebar: status + knowledge base management ---
with st.sidebar:
    st.header("🛡️ Local Safety Assistant")
    if model.ready:
        st.success(model.message)
    else:
        st.warning(model.message)
    if embedder.ready:
        st.caption(f"🔎 Hybrid retrieval (TF-IDF + semantic) · {embedder.model_id}")
    else:
        st.caption("🔎 TF-IDF retrieval only (semantic embeddings unavailable)")
    if reranker.ready:
        st.caption(f"🎯 Cross-encoder reranking active · {config.RERANKER_MODEL}")
    else:
        st.caption("🎯 Reranking unavailable — using hybrid order as-is")
    if translator.ready:
        st.caption(f"🌐 Local EN→TR translation active · {config.TRANSLATION_EN_TR_MODEL}")
    else:
        st.caption("🌐 Local translation unavailable — falling back to the chat model for translation")
    st.caption("Fully offline — no cloud, no API keys, no outbound network calls.")

    st.divider()
    response_language = st.selectbox(
        "Response language",
        RESPONSE_LANGUAGES,
        index=0,
        help=(
            "Retrieval works across languages regardless of this setting "
            "(e.g. ask in Turkish about an English document). This only "
            "controls the language the answer is written in. 'Auto' replies "
            "in the same language as your question."
        ),
    )

    st.divider()
    st.subheader("Knowledge base")
    docs = store.list_documents()
    st.caption(f"{len(docs)} document(s), {store.count()} chunk(s) indexed")
    for d in docs:
        st.text(f"• {d['title']}  ({d['chunks']} chunk(s))")

    st.divider()
    st.subheader("Add a document")
    uploaded = st.file_uploader("Markdown, text, PDF, or Word file", type=["md", "markdown", "txt", "pdf", "docx"])
    if uploaded is not None:
        # st.file_uploader keeps returning the same file on every rerun until
        # the user removes it or picks a different one — without this guard,
        # the st.rerun() below would re-process the same file forever.
        upload_identity = (uploaded.name, uploaded.size)
        if st.session_state.get("_last_processed_upload") != upload_identity:
            try:
                safe_name = validate_upload(uploaded.name, uploaded.size)
                try:
                    raw_text = read_uploaded_text(safe_name, uploaded.getvalue())
                except UnicodeDecodeError:
                    raw_text = ""

                if not raw_text:
                    st.error(
                        "Could not extract any text from this file. PDFs made of "
                        "scanned images (no text layer) are not supported."
                    )
                    st.session_state["_last_processed_upload"] = upload_identity
                else:
                    chunks = document_to_chunks(raw_text, safe_name)
                    if not chunks:
                        st.error("The file has no readable content.")
                        st.session_state["_last_processed_upload"] = upload_identity
                    else:
                        embeddings = embedder.embed([c.content for c in chunks]) if embedder.ready else None
                        store.remove_document(chunks[0].doc_id)
                        store.add_chunks(chunks, embeddings=embeddings)
                        st.session_state["_last_processed_upload"] = upload_identity
                        st.success(f"Indexed '{chunks[0].title}' ({len(chunks)} chunk(s)).")
                        st.rerun()
            except UploadRejected as err:
                st.session_state["_last_processed_upload"] = upload_identity
                st.error(str(err))

# --- Main chat area ---
st.markdown(_HEADER_HTML, unsafe_allow_html=True)

if not st.session_state.history:
    st.caption("Try an example:")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example_q in zip(example_cols, EXAMPLE_QUESTIONS):
        if col.button(example_q, use_container_width=True):
            st.session_state["_pending_question"] = example_q

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        render_sources(turn.get("sources") or [])
        if turn.get("notice"):
            st.caption(f"⚠️ {turn['notice']}")

typed_question = st.chat_input("Describe the issue or ask a question…")
question = typed_question or st.session_state.pop("_pending_question", None)
if question:
    # Only role/content ever reach the model as history (see build_messages) —
    # "notice" is UI-only and must never be part of what gets sent back as
    # conversation context, or the model starts imitating it on later turns.
    model_history = [
        {"role": t["role"], "content": t["content"]} for t in st.session_state.history
    ]
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources: list[dict] = []
        notice = None
        for event in engine.ask(question, model_history, response_language=response_language):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                full_answer += event["text"]
                placeholder.markdown(full_answer + "▌")
            elif event["type"] == "notice":
                notice = event["message"]
            elif event["type"] == "error":
                full_answer = f"⚠️ {event['message']}"
        placeholder.markdown(full_answer)
        render_sources(sources)
        if notice:
            st.caption(f"⚠️ {notice}")

    st.session_state.history.append(
        {"role": "assistant", "content": full_answer, "sources": sources, "notice": notice}
    )
