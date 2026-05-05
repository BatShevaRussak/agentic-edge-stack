# Agentic Edge Stack

A locally hosted, high-performance AI assistant capable of retrieving
technical information and executing logic via an agentic loop.
Demonstrates model serving, RAG, agentic orchestration, and streaming
APIs - all on CPU-only hardware.

## Architecture

The project follows a layered FastAPI-style application layout:

```
app/
├── core/       # Cross-cutting concerns (Pydantic Settings)
├── llm/        # Local Ollama LLM client (Part 1)
├── rag/        # In-memory FAISS RAG pipeline (Part 2)
├── agent/      # Agentic orchestrator (Part 3 - planned)
├── api/        # FastAPI endpoints (Part 4 - planned)
└── schemas/    # Pydantic request / response models
```

The `rag/` package is split into single-responsibility modules:

```
app/rag/
├── types.py          # Chunk, RetrievalHit, RetrievalResult dataclasses
├── errors.py         # RAGError hierarchy
├── chunker.py        # RecursiveCharacterTextSplitter wrapper
├── embeddings.py     # SentenceTransformer (BGE) wrapper
├── vector_store.py   # FAISS IndexFlatIP store with metadata sidecar
├── retriever.py      # Orchestrator + on-disk cache
└── prompt_builder.py # Augmented prompt assembly
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) for local inference

> **LLM:** default `llama3.2:1b` (Llama 3.2 family, Q4_K_M, ~770 MB)
> **Embedding model:** `BAAI/bge-small-en-v1.5` (33M params, 384-dim)

Both are chosen for CPU-only edge hardware - see
[`data/01_llama32_model_card.md`](data/01_llama32_model_card.md) and
[`data/03_sentence_transformers_and_bge.md`](data/03_sentence_transformers_and_bge.md)
for the rationale.

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd agentic-edge-stack
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -e .
copy .env.example .env
```

### 2. Obtain the LLM (Part 1)

Install Ollama, ensure the daemon is running, then choose **one** of two
paths to obtain the model.

#### Path A - Online

```bash
ollama pull llama3.2:1b
```

#### Path B - Offline / air-gapped

Place
[`Llama-3.2-1B-Instruct-Q4_K_M.gguf`](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)
in your `Downloads` folder, then:

```powershell
.\scripts\import_model.ps1
```

The script copies the GGUF into `models/`, runs `ollama create` against
the project's [`Modelfile`](Modelfile), and verifies registration. The
Modelfile embeds Meta's official Llama 3.2 chat template and stop tokens,
so the imported model behaves identically to one fetched via `ollama pull`.

### 3. Obtain the embedding model (Part 2)

By default the project loads the embedding model from a local directory
(`./models/bge-small-en-v1.5/`), which makes Part 2 fully offline at
runtime. To populate that directory in one step, run:

```powershell
.\scripts\import_embed_model.ps1
```

This downloads the 10 files that make up `BAAI/bge-small-en-v1.5`
(~134 MB total) directly from huggingface.co.

Alternatively, set `EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5` in `.env`
to let `sentence-transformers` fetch and cache the model on first use.

### 4. Verify Part 1

```bash
python tests\verify_ollama.py
```

You should see a successful short "hello" reply from the local model.

### 5. Verify Part 2

```bash
python scripts\ingest.py        # build the FAISS index (~5 s, cached afterwards)
python tests\verify_rag.py      # run the demo queries end-to-end
```

`verify_rag.py` runs six demo queries (five in-domain, one deliberately
out-of-domain) and writes a full trace to
`tests/logs/rag_run_<UTC timestamp>.txt`. The trace shows, for every
query, the retrieved chunks with cosine scores, the augmented prompt
sent to the LLM, and the LLM's grounded answer.

Pass `--skip-llm` to run retrieval only (useful in CI).

## Configuration

| Variable | Meaning | Default |
|----------|---------|---------|
| `OLLAMA_HOST` | Ollama base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model tag | `llama3.2:1b` |
| `REQUEST_TIMEOUT` | HTTP timeout seconds | `120` |
| `EMBED_MODEL_NAME` | HuggingFace id or local path | `./models/bge-small-en-v1.5` |
| `EMBED_DIM` | Embedding dimensionality | `384` |
| `RAG_CHUNK_SIZE` | Characters per chunk | `500` |
| `RAG_CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `RAG_TOP_K` | Chunks retrieved per query | `3` |
| `RAG_SCORE_THRESHOLD` | Cosine floor for hits | `0.5` |
| `DATA_DIR` | Corpus directory | `data` |
| `CACHE_DIR` | On-disk vector cache | `data/cache` |

See [`.env.example`](.env.example) for the full template.

## RAG design notes

- **Embedding model:** BGE-small-en-v1.5 (~33M params, 384-dim) chosen
  over `all-MiniLM-L6-v2` (22M, 384-dim) because BGE-small scores ~6
  points higher on MTEB at the same inference cost.
- **Vector store:** FAISS `IndexFlatIP` (exact brute-force) with
  L2-normalized vectors, so inner product equals cosine similarity. At
  ~50-200 chunks the scale is too small to benefit from approximate
  indexes (IVF, HNSW); they would only add tuning overhead.
- **Chunking:** `RecursiveCharacterTextSplitter` with markdown-aware
  separators (`\n## `, `\n### `, ...). Chunks are 500 characters with
  50-character overlap.
- **Cache:** ingested vectors are persisted under `data/cache/`. The
  manifest is keyed on a SHA-256 of the corpus contents plus the
  embedding model name and dimension - any edit invalidates the cache
  automatically.
- **Score threshold (0.5)** filters out weakly-related chunks, so
  out-of-domain queries surface as zero-hit retrievals and the LLM is
  routed to its "I don't have that information" branch.

## Project Status

- [x] **Part 1: Model serving & deployment** - Ollama + Llama 3.2 1B (Q4_K_M) via Modelfile import; `verify_ollama.py` "Hello World" passes locally.
- [x] **Part 2: In-Memory RAG** - FAISS `IndexFlatIP` over BGE-small embeddings; markdown-aware chunking; on-disk cache; `verify_rag.py` produces a per-query trace log under `tests/logs/`.
- [ ] Part 3: Agentic Orchestrator (LangChain tools)
- [ ] Part 4: Streaming API (FastAPI + SSE)
- [ ] Part 5: Bonus tasks (quantization profiling, structured output)

## License

MIT
