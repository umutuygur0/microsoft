# Local RAG Assistant (Foundry Local)

A fully **offline** Retrieval-Augmented Generation (RAG) assistant. It answers
questions grounded in a local document collection using a locally-served
language model — no cloud, no API keys, no outbound network calls after the
initial model download.

| Layer      | Technology                              | Why                                             |
|------------|------------------------------------------|--------------------------------------------------|
| AI model   | Foundry Local + Phi-3.5 Mini             | On-device inference, native Python SDK           |
| Retrieval  | Hybrid: TF-IDF + optional semantic embeddings | Fast, transparent, keyword-exact by default — degrades gracefully to TF-IDF-only if the embedding model isn't available |
| Vector store | SQLite                                 | Zero infrastructure, single file on disk         |
| Front end  | Streamlit                                | Chat UI, source citations, document upload       |

> Learning/portfolio project — not production medical or safety advice.

## Prerequisites

1. **Python 3.11+**
2. **Foundry Local** (Microsoft's on-device AI runtime):
   ```powershell
   winget install Microsoft.FoundryLocal
   ```
   The chat model (Phi-3.5 Mini, ~2.5 GB) and, optionally, the embedding model
   (qwen3-embedding-0.6b, ~0.5 GB) download automatically on first run.

## Quick start

```powershell
cd local-rag-foundry
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Index the knowledge base (data/docs/*.md) into SQLite, with embeddings
.venv\Scripts\python scripts\run_ingest.py --reset

# Launch the chat UI
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). The sidebar
shows the model status ("Offline Ready · ...") and whether hybrid retrieval is
active. Ask a question in the chat box — the answer streams in and cites its
sources, along with a TF-IDF/semantic score breakdown when hybrid mode is on.

> **No Foundry Local yet?** The app still runs. Retrieval works immediately;
> the sidebar explains how to enable answer generation, and until then the
> assistant shows the most relevant source passage instead of a generated answer.
> Skip embeddings with `python scripts/run_ingest.py --no-embed` for a faster,
> TF-IDF-only index.

## How it works

```
Question ─┬─▶ TF-IDF vector ──▶ inverted-index candidate lookup ──▶ cosine score ─┐
           └─▶ query embedding (if available) ──▶ semantic cosine score ──────────┼─▶ blended rank
                                                                                    ▼
                              top-K chunks → prompt (system + context + history) → Foundry Local
                              → streamed answer + source citations → Streamlit chat UI
```

1. **Ingest** (`scripts/run_ingest.py`) reads every `.md`/`.txt` file in
   `data/docs/`, parses optional YAML front-matter, splits it into ~200-word
   overlapping chunks, and stores each chunk with its term-frequency map — and,
   if the embedding model is available, a semantic embedding vector — in SQLite.
2. **Retrieve** (`src/vector_store.py`) converts the user's question into a
   TF-IDF vector and, if hybrid mode is active, also into an embedding. Chunks
   are ranked by a weighted blend of TF-IDF cosine similarity and semantic
   cosine similarity (`config.HYBRID_TFIDF_WEIGHT` / `HYBRID_EMBEDDING_WEIGHT`).
   Without an embedding model, this degrades transparently to pure TF-IDF.
3. **Augment** (`src/prompts.py`) builds a safety-first system prompt plus the
   retrieved chunks as context.
4. **Generate** (`src/foundry_client.py`) streams the model's response through
   the Foundry Local Python SDK.

## Project layout

```
local-rag-foundry/
├── config.py              # model names, chunk size/overlap, top_k, hybrid weights, paths
├── data/docs/             # knowledge base (.md files)
├── src/
│   ├── tfidf.py            # tokenize / term_frequency / idf / cosine_similarity (sparse + dense)
│   ├── chunker.py           # front-matter parsing + overlapping chunking
│   ├── vector_store.py       # SQLite store + inverted index + hybrid search()
│   ├── ingest.py              # ingestion pipeline (with optional embeddings)
│   ├── embedder.py             # optional semantic embedding client (graceful degradation)
│   ├── foundry_client.py        # Foundry Local chat model loading + streaming
│   ├── prompts.py                 # system prompt + message assembly
│   ├── chat_engine.py              # retrieve → augment → generate orchestration
│   └── security.py                  # upload validation (extension/size/path-traversal)
├── app/streamlit_app.py    # chat UI
├── scripts/run_ingest.py   # CLI: python scripts/run_ingest.py [--reset] [--no-embed]
└── tests/                  # pytest suite (36 tests)
```

## Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

## Known limitations

- **CPU inference, not GPU.** On this machine, Foundry Local's catalogue only
  exposes a `CPUExecutionProvider` variant for the chat and embedding models
  tested (`phi-3.5-mini`, `phi-4-mini`, `qwen2.5-7b`, `qwen3-embedding-0.6b`, ...),
  despite an NVIDIA RTX 5080 being present — `foundry status` reports it, but
  no GPU execution-provider variant is currently offered for these models.
  Inference is still fast enough for interactive use.
- Adding a document via the sidebar only updates the running session's SQLite
  file; conversation history is kept in memory only (lost on page reload).
- Hybrid retrieval computes semantic similarity against every chunk (not just
  TF-IDF candidates), which is fine at this corpus size (tens of chunks) but
  would need an ANN index (e.g. FAISS) to stay fast at a much larger scale.

## Knowledge base attribution

The 20-document gas-field engineering knowledge base in `data/docs/` was
taken, with the author's permission, from
[YusufAtakanUnal/local-rag-foundry](https://github.com/YusufAtakanUnal/local-rag-foundry)
(MIT License).
