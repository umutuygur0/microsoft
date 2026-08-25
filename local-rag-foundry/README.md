# Local RAG Assistant

*Built during my AI Engineer Internship — Microsoft AI Innovations Internship Program.*

A fully **offline** Retrieval-Augmented Generation (RAG) assistant for industrial
oil & gas safety procedures. It answers questions grounded in a 40-document
knowledge base, in English, Turkish, or mixed-language queries, using an
entirely locally-served stack — no cloud API, no API keys, no outbound network
calls once the models are on disk.

## Live results

A comprehensive, transparent 98-question audit (easy/medium/hard, EN/TR/mixed,
every one of the 40 source documents covered) was run end-to-end against the
live pipeline three times, fixing a real, chunk-verified bug after each round
— full write-up in [TEST_REPORT.md](TEST_REPORT.md):

| | |
|---|---|
| **Correct** | **92 / 98 (94%)** |
| Honest "not in knowledge base" refusals (no hallucination) | 2 / 98 |
| Confirmed model-capacity limit (chunk-verified, tested against 2 independent fixes) | 4 / 98 |
| **Hallucinations / wrong citations / retrieval errors / language drift** | **0 / 98** |

## Architecture

| Layer | Technology | Why |
|---|---|---|
| Chat model | `llama3.1:8b` via [Ollama](https://ollama.com) | Fully local; native `think=False`/context controls |
| Embedding | `bge-m3` via Ollama | Multilingual — Turkish ↔ English cross-lingual retrieval with no translation step |
| Sparse retrieval | Okapi BM25 (`src/bm25.py`) | Keyword-exact signal, blended with the semantic score |
| Dense retrieval | Cosine similarity + a pure-semantic "backstop" pool | The backstop recovers cases where a stray keyword collision would otherwise corrupt the BM25 blend (see TEST_REPORT.md §16) |
| Reranking | Cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) + confidence floor | Re-scores (query, chunk) pairs together for precision; falls back to hybrid order when the reranker itself has no real signal for a query |
| Translation | Meta **NLLB-200**, language-forced (`forced_bos_token_id`) | Deterministic EN→TR translation as a dedicated model, not a second LLM call — see "Why not just ask the LLM to translate?" below |
| Citations | Deterministic, built from retrieval metadata | The model is never asked to name its own source — see "Why deterministic citations?" below |
| Vector store | SQLite | Zero infrastructure, single file on disk |
| Front end | Streamlit | Chat UI, source citations with relevance breakdown, language selector, live document upload |

> Learning/portfolio project — not production safety advice.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the original design decisions,
[TEST_REPORT.md](TEST_REPORT.md) for the full test suite plus the complete,
chunk-by-chunk investigation log behind every architectural decision below,
and [FINAL_REPORT.md](FINAL_REPORT.md) for the closing summary against the
original requirements.

## Why this stack, not the obvious one

Three non-obvious decisions came directly out of live testing (not
theory) — each is fully documented, with reproduction evidence, in
`TEST_REPORT.md`:

**Why not just ask the LLM to translate?** The chat model *can* translate its
own English answer into Turkish, and earlier versions of this project did
exactly that. Live testing found it unreliable in ways a bigger prompt
couldn't fix: garbled or self-contradictory Turkish, and at least one case of
outright factual drift (a number silently changed mid-translation). Swapping
in a dedicated translation model removed that failure class entirely. That
model (initially a bilingual MarianMT model) then showed its *own* failure
mode — reproducibly drifting into an entirely unrelated language on a handful
of inputs (one answer came back in Portuguese). Meta's NLLB-200 was chosen
specifically because it supports forcing the output language at the token
level, which makes that failure structurally impossible rather than just
unlikely.

**Why deterministic citations?** Every earlier version of this project asked
the model to name its own source inline. Repeated live testing found this
model cannot be trusted to do that correctly — it invented titles, cited the
wrong excerpt, or echoed internal "[Source N]"-style labels verbatim, even
with explicit instructions. Since the actual retrieved chunk and its title
are already known with certainty *before* generation starts, the citation is
now built in code from that data after the fact. The model is never asked to
cite anything at all.

**Why a cross-encoder *and* a confidence floor?** Reranking measurably
improved precision, but a live audit found the reranker itself sometimes has
no real signal for a query — every candidate scores deep negative, and its
"ranking" among them is just noise. In one measured case that noise pushed
the correct, top-ranked hybrid-search result out of the final answer
entirely. `config.RERANK_MIN_CONFIDENCE` makes the reranker defer back to the
hybrid ranking when its own best score doesn't clear a calibrated bar,
instead of trusting a ranking it has no basis for.

## Prerequisites

1. **Python 3.11+**
2. **[Ollama](https://ollama.com)**, with the chat and embedding models pulled:
   ```powershell
   ollama pull llama3.1:8b
   ollama pull bge-m3
   ```
3. **~3.5 GB extra disk** for the reranker and NLLB-200 translation models —
   both download automatically (from Hugging Face) the first time the app runs.

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
shows live status for the chat model, hybrid retrieval, reranker, and
translator. Click one of the four example questions to try it immediately, or
type your own — in English, Turkish, or a mix of both.

**Multi-language support:** ask in Turkish about an English-sourced document
(or vice versa) — retrieval is language-independent because the semantic half
of hybrid search uses a multilingual embedding model. Use the **Response
language** dropdown to force English or Turkish regardless of the question's
language, or leave it on "Auto" to answer in whichever language you asked in.

**Document formats:** upload `.md`, `.txt`, `.pdf`, or `.docx` files from the
sidebar. Scanned (image-only) PDFs have no text layer to extract and aren't
supported (no OCR).

## How it works

```
Question ─┬─▶ BM25 (keyword) score ─────────────────────────┐
           ├─▶ embedding cosine score ──────────────────────┼─▶ hybrid blend
           └─▶ pure-semantic "backstop" pool (bypasses BM25) ┘        │
                                                                       ▼
                                          cross-encoder reranks (query, chunk)
                                          pairs together, or defers to the
                                          hybrid order if it has no real signal
                                                                       │
                                                                       ▼
                       top-K chunks → prompt (system + context + history) → llama3.1:8b
                                                                       │
                              EN→TR? ──▶ NLLB-200 (language-forced) ──┤
                                                                       ▼
                     deterministic "Reference: ..." footer built from the
                     actual retrieved chunks (never from the model) → Streamlit UI
```

1. **Ingest** (`scripts/run_ingest.py`) reads every `.md`/`.txt` file in
   `data/docs/`, splits it into structure-aware chunks (headings/list items
   kept atomic), and stores each chunk's term-frequency map plus, if the
   embedding model is available, a dense vector — in SQLite.
2. **Retrieve** (`src/vector_store.py`) blends BM25 and semantic cosine
   similarity (`config.HYBRID_BM25_WEIGHT` / `HYBRID_EMBEDDING_WEIGHT`), then
   merges in a pure-semantic backstop pool before reranking.
3. **Rerank** (`src/reranker.py`) re-scores the widened candidate pool with a
   cross-encoder, falling back to the hybrid order below a confidence floor.
4. **Augment** (`src/prompts.py`) builds a safety-first system prompt plus the
   retrieved chunks as context — the model is never asked to cite its source.
5. **Generate** (`src/ollama_client.py`) streams the grounded answer from
   Ollama, with a repetition-collapse guard and automatic retry.
6. **Translate** (`src/translator.py`), only if the answer needs to be in
   Turkish: NLLB-200 with the output language forced at the token level,
   falling back to an LLM-based translation pass if the local model is
   unavailable.
7. **Cite**: a `Reference: ...` footer is appended deterministically from the
   actual retrieved chunk metadata (`src/chat_engine.py`), never from
   anything the model wrote.

## Project layout

```
local-rag-foundry/
├── config.py                # model names, retrieval weights, context window, paths
├── data/docs/                # knowledge base (40 .md files)
├── src/
│   ├── tfidf.py                # tokenize (Unicode-aware) / term_frequency / dense cosine similarity
│   ├── bm25.py                  # Okapi BM25 scoring
│   ├── chunker.py                 # structure-aware chunking (atomic headings/list items)
│   ├── document_readers.py          # PDF/DOCX -> plain text extraction
│   ├── vector_store.py                # SQLite store + hybrid search() + semantic backstop
│   ├── reranker.py                      # cross-encoder reranking + confidence floor
│   ├── translator.py                      # NLLB-200 language-forced EN->TR translation
│   ├── ollama_client.py                     # Ollama chat streaming + repetition guard
│   ├── ollama_embedder.py                     # Ollama embedding client
│   ├── ingest.py                                # ingestion pipeline
│   ├── prompts.py                                 # system prompt + message assembly
│   ├── chat_engine.py                               # retrieve -> rerank -> generate -> translate -> cite
│   └── security.py                                    # upload validation
├── app/streamlit_app.py     # chat UI
├── scripts/run_ingest.py    # CLI: python scripts/run_ingest.py [--reset] [--no-embed]
└── tests/                    # pytest suite (116 tests)
```

## Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

## Known limitations

Everything below was found and confirmed via live testing against the real
model — not guessed. Full evidence for each is in `TEST_REPORT.md`.

- **A residual ~4% model-capacity ceiling.** Four questions in the 98-question
  audit get an incomplete or overly hedged answer even though the exact fact
  was verified to be present, verbatim, in the top-ranked retrieved chunk.
  Two independent fixes were tried and tested against the full 98-question
  set — a stricter "quote the source sentence first" prompt rule, and
  `temperature=0.0` — neither reliably helped, and the temperature change
  measurably broke a previously-perfect answer elsewhere. This is treated as
  a genuine `llama3.1:8b` capability limit, not a retrieval or prompt gap.
- **Cross-lingual retrieval is very good but not perfect.** One Turkish query
  about fire-extinguisher selection returns no relevant chunk in any tested
  embedding model — the underlying document doesn't cover that topic at all,
  so this is a corpus gap, correctly surfaced as a "not available" answer
  rather than a hallucination.
- Adding a document via the sidebar only updates the running session's SQLite
  file; conversation history is kept in memory only (lost on page reload).
- No OCR: a PDF made of scanned page images (no embedded text layer) yields no
  extractable text and will be rejected as "no readable content."
- Hybrid retrieval computes semantic similarity against every chunk (not just
  BM25 candidates), which is fine at this corpus size (a few hundred chunks)
  but would need an ANN index (e.g. FAISS) to stay fast at a much larger scale.

## Knowledge base

`data/docs/` holds 40 documents, in two distinct groups:

1. **20 narrative procedure documents** (e.g. `corrosion-assessment.md`,
   `esd-protocol.md`) — originated, with the author's permission, from
   [YusufAtakanUnal/local-rag-foundry](https://github.com/YusufAtakanUnal/local-rag-foundry)
   (MIT License), then substantially expanded with additional explanatory
   prose and common-failure-mode sections, keeping the original technical
   facts (pressures, thresholds, procedures) unchanged.
2. **20 raw reference documents** (`osha-*.md`, `epa-spcc-overview.md`) —
   real, unedited excerpts from OSHA/EPA web pages (U.S. federal government
   works, public domain). Kept in their original, unstructured form to prove
   the pipeline handles heterogeneous real-world content as well as the
   curated narrative documents.
