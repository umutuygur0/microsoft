# Local RAG Assistant (Foundry Local)

A fully **offline** Retrieval-Augmented Generation (RAG) assistant. It answers
questions grounded in a local document collection using a locally-served
language model — no cloud, no API keys, no outbound network calls after the
initial model download.

| Layer      | Technology                              | Why                                             |
|------------|------------------------------------------|--------------------------------------------------|
| AI model   | Foundry Local + Qwen2.5-7B-Instruct      | On-device inference, native Python SDK — see "Model choice" below |
| Retrieval  | Hybrid: TF-IDF + optional semantic embeddings | Fast, transparent, keyword-exact by default — degrades gracefully to TF-IDF-only if the embedding model isn't available |
| Cross-lingual search | Multilingual embeddings (qwen3-embedding-0.6b) | Ask in Turkish about an English document (or vice versa) — no translation step needed |
| Document formats | Markdown, .txt, PDF, DOCX | `src/document_readers.py` extracts plain text from PDF/DOCX before chunking |
| Vector store | SQLite                                 | Zero infrastructure, single file on disk         |
| Front end  | Streamlit                                | Chat UI, source citations, language selector, document upload |

> Learning/portfolio project — not production medical or safety advice.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the design decisions,
[TEST_REPORT.md](TEST_REPORT.md) for the automated test suite results plus a
log of live queries run against the real model, and
[FINAL_REPORT.md](FINAL_REPORT.md) for a full closing summary — including how
this project maps against its two source requirement documents and an honest
list of what remains a known limitation.

## Prerequisites

1. **Python 3.11+**
2. **Foundry Local** (Microsoft's on-device AI runtime):
   ```powershell
   winget install Microsoft.FoundryLocal
   ```
   The chat model (Qwen2.5-7B-Instruct, ~6.2 GB) and, optionally, the
   embedding model (qwen3-embedding-0.6b, ~0.5 GB) download automatically on
   first run.

### Model choice

The project started with `phi-3.5-mini` (2.5 GB, faster) but live testing
surfaced a serious, reproducible problem: on certain queries — especially
with a forced response language — it would collapse into repeating the same
character, word, or sentence for hundreds to thousands of characters
(see `TEST_REPORT.md` for the full investigation, including a two-pass
grounding/translation architecture, retry-with-higher-temperature, and
frequency/presence-penalty tuning, none of which fully resolved it).
Switching to `qwen2.5-7b` (6.2 GB) eliminated the collapse entirely across
every query that previously triggered it. `qwen3-8b` was also tried but
defaults to an internal `<think>...</think>` reasoning block that itself
collapsed in the same way, so it was not used. Trade-off: a larger download
and slower CPU inference than phi-3.5-mini, in exchange for reliability.
Override with the `FOUNDRY_MODEL` environment variable if you'd rather use a
smaller/faster model and accept the risk.

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

**Multi-language support:** the knowledge base, your question, and the
answer language are all independent. You can ask a Turkish question about an
English document (or mix languages in one query — "Gas leak durumunda ne
yapmalıyım?") and retrieval still finds the right chunks, because the
semantic half of hybrid search uses a multilingual embedding model. Use the
**Response language** dropdown in the sidebar to force the answer into a
specific language regardless of the question's or the documents' language, or
leave it on "Auto" to answer in whatever language you asked in.

**Document formats:** upload `.md`, `.txt`, `.pdf`, or `.docx` files from the
sidebar. PDF/DOCX text is extracted automatically before chunking. Scanned
(image-only) PDFs have no text layer to extract and aren't supported (no OCR).

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
│   ├── tfidf.py            # tokenize (Unicode-aware) / term_frequency / idf / cosine_similarity (sparse + dense)
│   ├── chunker.py           # front-matter parsing + overlapping chunking
│   ├── document_readers.py   # PDF/DOCX -> plain text extraction (pypdf, python-docx)
│   ├── vector_store.py         # SQLite store + inverted index + hybrid search()
│   ├── ingest.py                # ingestion pipeline (.md/.txt/.pdf/.docx, with optional embeddings)
│   ├── embedder.py                # multilingual semantic embedding client (graceful degradation)
│   ├── foundry_client.py           # Foundry Local chat model loading + streaming
│   ├── prompts.py                    # system prompt + message assembly + response-language control
│   ├── chat_engine.py                 # retrieve → augment → generate orchestration
│   └── security.py                     # upload validation (extension/size/path-traversal)
├── app/streamlit_app.py    # chat UI (language selector, PDF/DOCX/MD/TXT upload)
├── scripts/run_ingest.py   # CLI: python scripts/run_ingest.py [--reset] [--no-embed]
└── tests/                  # pytest suite (75 tests)
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
- No OCR: a PDF made of scanned page images (no embedded text layer) yields no
  extractable text and will be rejected as "no readable content."
- The "Response language" selector only forces the *answer's* language; it is
  not a document-language filter (there isn't one — cross-lingual retrieval
  means all documents are always searched, regardless of language).
- **Forced-language translation quality is imperfect.** With `qwen2.5-7b`,
  translated answers no longer collapse into repetition, but fidelity is
  inconsistent: sometimes the translation is fluent, sometimes word order is
  awkward, and occasionally the model answers in English anyway despite the
  explicit instruction, or leaks raw Markdown headers from the source
  content into the output. Two different translation-pass architectures
  were tried (a single whole-block call, and a line-by-line call per
  sentence) — both hit the same underlying disfluency, and the line-by-line
  variant measurably made one case worse (a hallucinated non-word). This
  points to a genuine translation-capability limit of this model for
  English→Turkish rather than something fixable by restructuring the
  prompt/call pattern — see `TEST_REPORT.md` sections 4-5 and
  `FINAL_REPORT.md` for the full investigation. The underlying *content*
  stays accurate and grounded — this is a fluency issue, not a
  hallucination or a stability issue.
- **"Auto" mode does not reliably match the question's language.** A Turkish
  question against English-sourced context sometimes gets an English answer
  even in "Auto" mode, despite the system prompt instructing the model to
  respond in the question's language. Retrieval is unaffected (it is
  language-independent either way) — this only affects which language the
  final answer is written in.

## Knowledge base

`data/docs/` now holds 40 documents (189 chunks), in two distinct groups:

1. **20 narrative procedure documents** (e.g. `corrosion-assessment.md`,
   `esd-protocol.md`) — these originated, with the author's permission, from
   [YusufAtakanUnal/local-rag-foundry](https://github.com/YusufAtakanUnal/local-rag-foundry)
   (MIT License), then were substantially expanded (from a combined ~3,700
   words to ~10,800 words) with additional explanatory prose, rationale, and
   common-failure-mode sections, while keeping the original technical facts
   (pressures, thresholds, procedures) unchanged. File names and internal
   section headings (`Overview`, `Key Safety Precautions`, `Working
   Procedure`, `Source Standard`) were deliberately changed from the
   reference project's naming/template.
2. **20 raw reference documents** (`osha-*.md`, `epa-spcc-overview.md`) —
   real, unedited excerpts fetched directly from OSHA/EPA web pages (U.S.
   federal government works, public domain, no copyright). These are
   intentionally **not** reformatted into the project's narrative template —
   each is just the page's own text (with site navigation/footer chrome
   stripped) plus a `Source:` attribution line, the same way a PDF or web
   page uploaded through the sidebar would be ingested. This demonstrates
   that the pipeline handles heterogeneous, unstructured real-world content
   just as well as the curated narrative documents.
