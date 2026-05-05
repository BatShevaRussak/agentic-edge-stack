# Knowledge Corpus

This directory contains the source documents used by the in-memory RAG
system (Part 2 of the assignment). The corpus is **self-referential**:
each document describes a component of the stack the assistant is built
on, so the assistant can answer factual questions about its own design.

## Documents

| File | Topic | ~Pages |
|------|-------|--------|
| `01_llama32_model_card.md` | Meta Llama 3.2 1B Instruct - architecture, training, intended use, limitations | 1.5 |
| `02_faiss_overview.md` | FAISS index families (Flat / IVF / HNSW), exact vs approximate search, when to use each | 1.2 |
| `03_sentence_transformers_and_bge.md` | Sentence embeddings, BGE vs MiniLM, normalization, MTEB benchmark | 1.3 |
| `04_ollama_runtime.md` | Ollama runtime, GGUF format, K-quants (Q4_K_M), Modelfile templating | 1.2 |
| `05_rag_concepts.md` | RAG pipeline, chunking strategies, top-K retrieval, prompt augmentation | 1.5 |

Total: ~6.7 pages of dense technical text (well within the 2-10 page
target stated in the assignment).

## Why a self-referential corpus?

Three reasons:

1. **Demoability.** Reviewers can ask the assistant questions like *"What
   quantization scheme does this system use?"* or *"Why was BGE-small
   chosen over MiniLM?"* and verify retrieval quality against the source.
2. **Factual grounding.** Every chunk is cited and traceable to a file,
   making hallucinations easy to spot.
3. **Domain coherence.** All documents share vocabulary (vectors, tokens,
   indices), which exercises retrieval more meaningfully than mixing
   unrelated topics.

## Cache

Embeddings are computed once during ingestion and cached under
`data/cache/` (gitignored). The cache key is a SHA-256 hash of the
corpus contents, so any edit to a `.md` file triggers a re-embed.

## Sources

All technical claims are paraphrased from publicly available primary
sources: the Llama 3.2 model card on HuggingFace and Meta's blog, the
FAISS wiki and paper (Johnson et al., 2017), the BGE model card and
Sentence-BERT paper (Reimers & Gurevych, 2019), the Ollama documentation,
and the original RAG paper (Lewis et al., 2020).
