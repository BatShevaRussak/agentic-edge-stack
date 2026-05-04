# Agentic Edge Stack

A locally hosted, high-performance AI Assistant capable of retrieving
technical information and executing logic via an agentic loop.
Demonstrates model serving, RAG, agentic orchestration, and streaming APIs.

## Architecture

The project follows a layered architecture optimized for FastAPI applications:

```
app/
├── core/       # Cross-cutting concerns (config)
├── llm/        # Local Ollama LLM client
├── rag/        # Retrieval-Augmented Generation
├── agent/      # Agentic orchestrator
├── api/        # FastAPI endpoints
└── schemas/    # Pydantic models
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) for local inference

> **Model:** default `llama3.2:1b` (Llama 3.2 family), chosen for CPU-only hardware.

## Quick Start

### 1. Clone and environment

```bash
git clone <your-repo-url>
cd agentic-edge-stack
python -m venv venv
```

Activate the venv, then:

```bash
pip install -e .
```

Copy `.env.example` to `.env` and edit if needed (see **Configuration**).

### 2. Install Ollama and obtain the model

Install Ollama, ensure the daemon is running, then choose **one** of two paths
to obtain the model.

#### Path A - Online (`ollama pull`)

If your network allows direct downloads from Ollama's registry:

```bash
ollama pull llama3.2:1b
```

On Windows you can also use `.\scripts\deploy.ps1` (venv + `pip install` + pull).

#### Path B - Offline (import a local GGUF file)

If `ollama pull` is blocked by network filtering, import a manually
downloaded GGUF file. Place
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` (e.g. from
[bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF))
in your `Downloads` folder, then run:

```powershell
.\scripts\import_model.ps1
```

The script copies the GGUF into `models/`, runs `ollama create` against the
project's [`Modelfile`](Modelfile), and verifies registration. The Modelfile
embeds Meta's official Llama 3.2 chat template and stop tokens, so the imported
model behaves identically to one fetched via `ollama pull`.

> **Note:** `models/` and `*.gguf` are gitignored (model weights are large
> and not redistributable through this repo).

### 3. Verify

```bash
python tests/verify_ollama.py
```

You should see a successful short "hello" reply from the local model.

## Configuration

| Variable | Meaning |
|----------|---------|
| `OLLAMA_HOST` | Ollama base URL (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model tag (default `llama3.2:1b`) |
| `REQUEST_TIMEOUT` | HTTP timeout seconds (default `120`) |

See `.env.example` for the full template.

## Project Status

- [x] **Part 1: Model serving & deployment** - Ollama + Llama 3.2 1B (Q4_K_M) via Modelfile import; `verify_ollama.py` "Hello World" passes locally.
- [ ] Part 2: In-Memory RAG (FAISS + embeddings)
- [ ] Part 3: Agentic Orchestrator (LangChain tools)
- [ ] Part 4: Streaming API (FastAPI + SSE)
- [ ] Part 5: Bonus tasks (quantization profiling, structured output)

## License

MIT
