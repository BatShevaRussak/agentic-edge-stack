# Ollama Runtime, GGUF, and K-Quants

## What Ollama is

Ollama is an open-source local inference server that wraps the
`llama.cpp` C++ runtime behind a friendly HTTP API and CLI. It runs
on macOS, Linux, and Windows, supports CPU and GPU backends, and
hosts a model registry with one-line pulls. The server listens on
`http://localhost:11434` by default and exposes endpoints that
mirror the OpenAI Chat Completions shape:

- `POST /api/generate` - single-prompt completion.
- `POST /api/chat` - multi-turn conversation.
- `POST /api/embeddings` - embedding (when the model supports it).
- `GET  /api/tags` - list locally installed models.

Streaming is supported on `generate` and `chat` by setting
`"stream": true` and reading the response as a sequence of
newline-delimited JSON chunks.

## GGUF format

GGUF (GPT-Generated Unified Format) is a binary file format
introduced by the `llama.cpp` project to package quantized model
weights together with all metadata required to run them: tokenizer
vocabulary, special tokens, chat template, default generation
parameters. A single `.gguf` file is therefore self-contained and
portable across machines.

This project ships the model as
`llama-3.2-1b-instruct-q4_k_m.gguf` (~770 MB), imported into Ollama
via a project-level `Modelfile`.

## K-Quants and Q4_K_M

`llama.cpp` introduced the K-quant family in 2023. Unlike legacy
fixed-bit quantization, K-quants split each weight tensor into
super-blocks and quantize each block with its own scale and zero
point, yielding higher fidelity at the same bit budget.

Q4_K_M is the most common 4-bit K-quant: it stores most weights at
4 bits but uses 6 bits for a small subset of "important" tensors
(typically `attn_v` and `ffn_down`). The M suffix means "medium" - a
balanced choice between Q4_K_S (smaller, slightly lower quality) and
Q5_K_M (larger, slightly higher quality). On Llama-family models,
Q4_K_M typically loses fewer than two perplexity points compared to
the FP16 baseline while shrinking the file by roughly 3.5x.

## Modelfile and chat templates

A Modelfile is Ollama's equivalent of a Dockerfile. It declares the
base GGUF, the chat template, stop tokens, and default sampling
parameters. The template is a Go template string that maps the
abstract roles `system`, `user`, and `assistant` onto the special
tokens the model was trained with.

For Llama 3.2 the official template uses three pairs of header
tokens (`<|start_header_id|>`, `<|end_header_id|>`) and a
`<|eot_id|>` end-of-turn marker. Without the correct template, the
model produces malformed output and ignores stop tokens, so getting
the Modelfile right is critical for both chat reliability and tool
calling later in Part 3.

## Why Ollama in this project

Three engines were considered: vLLM, raw `llama.cpp`, and Ollama.

- vLLM is the highest-throughput option but requires an NVIDIA GPU
  with CUDA, which CPU-only edge hardware lacks.
- `llama.cpp` runs anywhere but exposes a low-level binary; building
  a stable HTTP wrapper around it takes meaningful effort.
- Ollama already provides the wrapper, ships precompiled Windows
  binaries, and exposes an OpenAI-compatible API surface. It is the
  pragmatic choice for a CPU-only edge stack and matches the
  assignment's "lightweight, local, easy to deploy" guidance.
