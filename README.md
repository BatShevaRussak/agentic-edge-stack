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
├── api/        # FastAPI + SSE streaming endpoints (Part 4)
└── schemas/    # Pydantic request / response models (reserved for Part 5)
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
├── prompts.py   # Router prompt, direct prompt, parse_route (default = "direct")
├── tools.py     # @tool rag_search + Pydantic args schema
├── nodes.py     # router / rag / synthesis / direct / fallback nodes
├── graph.py     # build_agent_graph() - the StateGraph wiring
└── runner.py    # AgentRunner.run() + trace formatting (text + json)
```

The `api/` package wraps the agent in a FastAPI service with SSE streaming:

```
app/api/
├── main.py      # create_app(), lifespan singleton AgentRunner, CORS
├── routes.py    # POST /chat (EventSourceResponse), GET /health
├── schemas.py   # ChatRequest, HealthResponse Pydantic models
└── sse.py       # StreamEventType, format_sse, langgraph -> SSE adapter
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
general-knowledge that should answer directly, and one ambiguous
out-of-domain query that the router sends to RAG and that falls
through to the canonical "no information" response). The trace log is
written to `tests/logs/agent_run_<UTC>.txt` and shows, for every query,
the router decision, the tool call with retrieved chunks and cosine
scores, the synthesis / direct / fallback output, and per-step latency.

### 7. Verify Part 4 - streaming API

In one terminal, launch the FastAPI server:

```powershell
.\scripts\run_api.ps1
# or, equivalently:
# uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete` (cold start ~5s, warm cache ~1s),
then in a second terminal run:

```powershell
python tests\verify_api.py
```

`verify_api.py` issues four queries to `POST /chat`, parses the
`text/event-stream` body incrementally, and prints tokens to stdout
*as they arrive* - the live proof that streaming actually works. It
records a `time-to-first-token` (TTFT) for each query alongside the
total latency; the gap between the two is the user-visible value of
streaming. Both a human-readable trace and a structured JSON copy are
saved under `tests/logs/api_run_<UTC>.txt` and `.json`.

Quick manual smoke test with `curl` (note: tokens arrive one-by-one):

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{"query": "Translate good morning to Spanish."}'
```

Auto-generated OpenAPI documentation is available at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

> **Heads-up:** Swagger UI buffers `text/event-stream` responses and
> only reveals them after the connection closes. To observe tokens
> *arriving* in real time, use `verify_api.py` or `curl -N` as shown
> above. The API itself streams correctly - this is purely a Swagger
> UI rendering limitation.

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
| `API_HOST` | FastAPI bind host | `0.0.0.0` |
| `API_PORT` | FastAPI bind port | `8000` |
| `API_CORS_ORIGINS` | CORS allow-list (JSON list) | `["*"]` |
| `SSE_KEEPALIVE_SECONDS` | Comment ping interval to defeat idle proxies | `15` |

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

### Deterministic LLM router

A 1B-parameter classifier is unreliable at binary tasks: in early runs,
the LLM router routed *every* query to RAG, regardless of whether the
question was project-specific. The root causes were three knobs working
together against us, all fixed in
[`app/agent/nodes.py`](app/agent/nodes.py) and
[`app/agent/prompts.py`](app/agent/prompts.py):

1. **Sampling.** The router LLM call now passes
   `options={"temperature": 0, "num_predict": 5}` so decoding is greedy
   (reproducible) and capped at five tokens (the model cannot ramble
   into a paragraph that confuses the parser).
2. **Parser default.** When the model output is genuinely ambiguous,
   `parse_route` returns `"direct"` rather than `"rag"`. The asymmetry
   is intentional: a wrong `"direct"` produces a generic LLM answer
   (often still useful); a wrong `"rag"` returns the canonical "I don't
   have that information" fallback whenever the question is unrelated
   to the knowledge base, which is strictly worse for the user.
3. **Few-shot balance.** The router prompt lists DIRECT guidance and
   examples first (primacy bias), with five DIRECT examples vs. three
   RAG examples - counter-weighting the 1B model's tendency to favour
   the more "specialised" looking label.

Every query is evaluated by the LLM (the trace records `method=llm`
on every decision), so the agent's routing is fully agentic and fully
observable.

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

## Streaming API design notes

Part 4 wraps the Part 3 LangGraph agent in a single-endpoint FastAPI
service that streams the agent's response token-by-token over Server-
Sent Events. The goals are (a) production-readiness (lifespan-managed
singletons, validation, CORS, health probe, keepalive, structured error
events), and (b) zero duplication of the agent's orchestration logic -
the graph stays the single source of truth.

### Why SSE, not WebSockets

The interaction is strictly server-to-client text streaming with one
request body up front. SSE matches that shape exactly: it works over
plain HTTP/1.1, supports built-in reconnection, is trivial to consume
with `curl`, and survives most corporate proxies. WebSockets adds
bi-directional framing complexity that this use case never needs.

### Streaming via LangGraph's custom-stream channel

Token-level streaming is delivered through LangGraph's `custom`
[stream mode](https://langchain-ai.github.io/langgraph/concepts/streaming/).
Inside `synthesis_node` and `direct_node`,
[`get_stream_writer()`](https://langchain-ai.github.io/langgraph/reference/runtimes/#langgraph.config.get_stream_writer)
yields a writer that is a no-op when the graph is invoked synchronously
(via `AgentRunner.run`) and a real conduit when invoked through
`graph.stream(stream_mode=["updates","custom"])`. The same node code
therefore powers both the Part 3 batch trace artefact (`agent_run_*`)
and the Part 4 SSE endpoint - one graph, one orchestration layer.

```python
# app/agent/nodes.py (excerpt)
writer = get_stream_writer()
for token in llm.generate_stream(prompt):
    chunks.append(token)
    writer({"type": "token", "node": "synthesis", "value": token})
```

### Rich event protocol (not tokens-only)

`POST /chat` emits four event types defined in
[`app/api/sse.py`](app/api/sse.py):

| `event:` | `data:` payload | When |
|----------|-----------------|------|
| `route` | `{route, method, elapsed_ms}` | After the router node decides |
| `tool_call` | `{name, input, hits, top_score, tool_elapsed_ms, elapsed_ms}` | After `rag_search` runs |
| `token` | `{type:"token", node, value, elapsed_ms}` | Once per LLM token |
| `done` | full `AgentResponse`-shaped trace + `total_elapsed_ms` | At the end of the stream |
| `error` | `{kind, message, elapsed_ms}` | Instead of `done` if a layer raises |

The closing `done` event carries the same trace structure that
`format_trace_json` produces in Part 3, so a single HTTP round-trip
delivers both the streaming UX *and* a complete audit log.

### Sync I/O bridged to the async event loop

`OllamaClient.generate_stream` uses blocking `requests` I/O. The route
handler (`app/api/routes.py`) wraps the synchronous SSE generator in
[`starlette.concurrency.iterate_in_threadpool`](https://www.starlette.io/concurrency/),
keeping the uvicorn event loop responsive for keepalive pings and
client-disconnect detection. `EventSourceResponse` from
[`sse-starlette`](https://github.com/sysid/sse-starlette) handles the
keepalive comment frames automatically (every `SSE_KEEPALIVE_SECONDS`).

### Lifespan singleton

`AgentRunner` is created exactly once in the FastAPI lifespan handler:
the BGE embedding model loads (~150 MB to RAM) and the FAISS index
ingests (or restores from `data/cache/`) *before* the server accepts
traffic. Subsequent requests share the same retriever (read-only after
ingest) and the same compiled graph, so concurrent `/chat` calls cost
only one extra in-flight LLM stream each. This pattern aligns with the
Kubernetes readiness-probe contract: `/health` only returns `status: ok`
once initialization is complete.

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
- [x] **Part 3: Agentic Orchestrator** - LangGraph state machine with router / rag / synthesis / direct / fallback nodes, `@tool`-decorated `rag_search`, deterministic single-stage LLM router (`temperature=0`, `num_predict=5`, asymmetric `direct`-biased parser), and dual text + JSON traces under `tests/logs/`.
- [x] **Part 4: Streaming API** - FastAPI service with lifespan-managed `AgentRunner` singleton, `POST /chat` returning SSE (`route` / `tool_call` / `token` / `done` / `error` events), `GET /health` probe, sync-stream-bridged-to-async I/O, and end-to-end `verify_api.py` artefact under `tests/logs/`.
- [ ] Part 5: Bonus tasks (quantization profiling, structured output)

## License

MIT
