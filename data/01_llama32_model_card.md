# Llama 3.2 1B Instruct - Model Card Summary

## Overview

Llama 3.2 1B Instruct is a small, instruction-tuned causal language
model released by Meta in September 2024 as part of the Llama 3.2
collection. It is designed for on-device and edge deployments where
larger models in the family (3B, 8B, 70B) are impractical due to memory
or latency constraints. The model is distributed under the Llama 3.2
Community License.

## Architecture

The model is a decoder-only Transformer with the same backbone as the
larger Llama 3 models, scaled down. Key architectural facts:

- 1.23 billion parameters (hence the "1B" tag).
- 16 transformer blocks.
- Hidden size 2048, 32 attention heads with grouped-query attention (GQA)
  using 8 key/value heads.
- SwiGLU activations, RMSNorm, and rotary position embeddings (RoPE).
- Vocabulary of 128,000 tokens (the Llama 3 tokenizer based on tiktoken).
- Native context window of 128K tokens, although 1B is most reliable up
  to roughly 8K tokens in practice.

## Training

The 1B model was created by pruning and distilling logits from the
larger Llama 3.1 8B and 70B teachers. Pretraining used a mixture of
publicly available text and code (~9 trillion tokens for the family).
Post-training combined supervised fine-tuning (SFT), rejection sampling,
and Direct Preference Optimization (DPO) on instruction data covering
multilingual chat, summarization, code, and tool use.

## Intended Use

Meta lists three primary use cases for the 1B/3B variants:

1. **On-device assistants** that run locally on phones, laptops, or
   edge devices.
2. **Information retrieval and summarization** of short documents.
3. **Tool calling and structured output** in agentic pipelines.

## Limitations

- The model is English-first; non-English performance is weaker.
- Reasoning depth is limited compared with 8B+ models; multi-hop logic
  often requires retrieval support (which is exactly the role RAG plays
  in this project).
- The model can hallucinate facts not present in its training data.
  Grounding via retrieved context is therefore essential for any
  factual application.

## Why this model in this project

This project targets CPU-only edge hardware. On a typical dual-core
laptop CPU the 3B variant produces ~1-2 tokens per second, which is
unusable for an interactive agent that calls a tool two or three times
per turn. The 1B variant produces ~5-8 tokens per second under the same
conditions, keeping per-turn latency under a few seconds and making the
system genuinely usable. The assignment explicitly permits "or a
comparable lightweight instruct model," and 1B is a member of the same
Llama 3.2 family named in the brief.

## Quantization in deployment

The weights shipped through Ollama are in GGUF Q4_K_M format - a
4-bit K-quant that compresses the model from ~2.5 GB (FP16) down to
~770 MB while preserving most of the quality on standard benchmarks.
Q4_K_M is widely regarded as the best size/quality tradeoff for small
Llama models on CPU. See `04_ollama_runtime.md` for details on GGUF
and the K-quant scheme.
