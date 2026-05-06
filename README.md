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
├── agent/      # LangGraph agentic orchestrator (Part 3)
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

The `agent/` package follows the same pattern, with the LangGraph
state machine split across modules of one concern each:

```
app/agent/
├── types.py     # AgentState, ToolCall, TraceEvent, AgentResponse
├── errors.py    # AgentError, RoutingError, ToolExecutionError
├── prompts.py   # Router prompt, direct prompt, heuristic pre-filter
├── tools.py     # @tool rag_search + Pydantic args schema
├── nodes.py     # router / rag / synthesis / direct / fallback nodes
├── graph.py     # build_agent_graph() - the StateGraph wiring
└── runner.py    # AgentRunner.run() + trace formatting (text + json)
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

### 6. Verify Part 3

```bash
python tests\verify_agent.py
```

`verify_agent.py` exercises every edge of the agent graph with seven
demo queries (three project-specific that should retrieve, three
general-knowledge that should answer directly, and one deliberately
out-of-domain that should route to RAG and fall through to the canonical
"no information" response). The trace log is written to
`tests/logs/agent_run_<UTC>.txt` and shows, for every query, the router
decision (heuristic vs. LLM), the tool call with retrieved chunks and
cosine scores, the synthesis / direct / fallback output, and per-step
latency.

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

## Agent design notes

Part 3 wraps the Part 2 RAG pipeline as a tool that an autonomous agent
can choose to call, exactly as required by the task brief. The agent is
implemented as a small **LangGraph** state machine - the modern
replacement for `langchain.AgentExecutor`, which has been deprecated
since LangChain 0.3.

### Topology

```mermaid
flowchart LR
    Start([User Query]) --> Router{router}
    Router -->|rag| RagSearch[rag_search]
    Router -->|direct| Direct[direct]
    RagSearch --> Check{hits > 0?}
    Check -->|yes| Synthesis[synthesis]
    Check -->|no| Fallback[fallback]
    Synthesis --> EndNode([Response])
    Direct --> EndNode
    Fallback --> EndNode
```

Five nodes, two conditional edges. `verify_agent.py` exercises every
edge.

### Why LangGraph, not LangChain `AgentExecutor`?

`AgentExecutor` and `create_tool_calling_agent` were deprecated in
LangChain 0.3 and rely on OpenAI-style native function calling, which
`llama3.2:1b` does not produce reliably. LangGraph keeps the state
machine explicit, lets us type the state (`AgentState` is a `TypedDict`
with `Annotated[list, add]` reducers for the trace), and exposes a
`get_graph().draw_mermaid()` view of the compiled graph - the diagram
above is generated from the actual edges, not hand-drawn.

### Two-tier router

A 1B-parameter classifier is unreliable at binary tasks: in early runs,
the LLM router routed *every* query to RAG, regardless of whether the
question was project-specific. The fix is a small **heuristic
pre-filter** in [`app/agent/prompts.py`](app/agent/prompts.py) that
catches arithmetic, translation, and self-identity questions in O(1)
without an LLM call. Anything ambiguous still goes through the LLM
router with a six-example few-shot prompt and a defensive parser.

The trace records `method=heuristic` or `method=llm` for every routing
decision, so the cascade is fully observable.

### LLM-free fallback

When the router chooses RAG but retrieval returns zero chunks above the
score threshold, the graph routes to `fallback_node`, which returns the
canonical *"I don't have that information in my knowledge base."*
string **without** calling the LLM. This saves 5-7 s per dead-end query
and matches the behaviour of the bare RAG pipeline from Part 2.

### Tool definition

The `rag_search` tool uses the modern `@tool` decorator from
`langchain-core` with a Pydantic `args_schema`. The function docstring
*is* the description an LLM-driven router sees, which is why it
explicitly enumerates the topics the knowledge base covers and the
topics it does not.

```python
@tool("rag_search", args_schema=RagSearchInput)
def rag_search(query: str) -> RetrievalResult:
    """Search the local technical knowledge base for facts about the
    deployed Llama 3.2 model, the RAG pipeline (FAISS IndexFlatIP),
    BGE-small embeddings, the Ollama runtime, or this project's chunking
    strategy. ..."""
```

### Trace artefact

Every node emits a single `TraceEvent` with `inputs`, `outputs`, and
`elapsed_ms`. `format_trace_text` turns an `AgentResponse` into the
human-readable trace committed under `tests/logs/`, mirroring the Part 2
RAG log style for easy side-by-side comparison. A second formatter,
`format_trace_json`, is also available for programmatic consumption.

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
- [x] **Part 3: Agentic Orchestrator** - LangGraph state machine with router / rag / synthesis / direct / fallback nodes, `@tool`-decorated `rag_search`, two-tier (heuristic + LLM) router, and dual text + JSON traces under `tests/logs/`.
- [ ] Part 4: Streaming API (FastAPI + SSE)
- [ ] Part 5: Bonus tasks (quantization profiling, structured output)

## License

MIT
