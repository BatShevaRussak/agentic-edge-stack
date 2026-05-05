# Sentence Embeddings - sentence-transformers and BGE

## What an embedding is

A text embedding is a fixed-length vector that represents the meaning
of a text. Two pieces of text with similar meaning produce vectors
that are close in the embedding space, where closeness is measured
by cosine similarity. Sentence embeddings are the foundation of
semantic search, RAG, clustering, and many classification pipelines.

## sentence-transformers

`sentence-transformers` is a Python library that wraps HuggingFace
Transformers with a high-level API tuned for producing sentence
vectors. It was introduced in the Sentence-BERT paper (Reimers &
Gurevych, EMNLP 2019), which showed that fine-tuning BERT with a
siamese architecture and contrastive loss produces dramatically better
sentence vectors than pooling a vanilla BERT. The library exposes
hundreds of pretrained models on the HuggingFace Hub.

A typical usage looks like:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vectors = model.encode(["hello world", "second sentence"])
```

`encode` runs tokenization, transformer forward pass, mean pooling,
and (optionally) L2 normalization in a single call.

## BGE - BAAI General Embedding

BGE is a family of embedding models published by the Beijing Academy
of Artificial Intelligence (BAAI) starting in mid-2023. The v1.5
revision improved retrieval quality and reduced sensitivity to query
phrasing. The "small" variant has roughly 33 million parameters and
produces 384-dimensional vectors, the same dimension as
`all-MiniLM-L6-v2` but with stronger benchmark performance.

## BGE-small vs MiniLM-L6

Both models are popular choices for local CPU-bound RAG. They are
compared on the MTEB (Massive Text Embedding Benchmark) leaderboard,
which aggregates 56 retrieval, classification, and clustering tasks:

| Model | Params | Dim | MTEB avg |
|-------|--------|-----|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | ~56.3 |
| `BAAI/bge-small-en-v1.5`                 | 33M | 384 | ~62.2 |

A six-point gap on MTEB is substantial - it corresponds roughly to
moving from a small model to a model 5-10x larger in older
benchmarks. Inference cost is similar (both fit in under 150 MB and
run at hundreds of sentences per second on a single CPU core), so
BGE-small is the better default unless backwards compatibility with
an existing index is required.

## Normalization and the cosine trick

Both models output raw vectors that are not unit-norm. To use cosine
similarity through a FAISS inner-product index, the vectors must be
L2-normalized before insertion. `sentence-transformers` exposes this
via `model.encode(..., normalize_embeddings=True)`, which divides
each vector by its L2 norm. After normalization, dot product equals
cosine similarity by definition.

## Optional query instruction

BGE models were trained with an optional natural-language instruction
prepended to queries (for example, *"Represent this sentence for
searching relevant passages: "*). For BGE v1.5 the instruction is
no longer required at inference, but adding it can give a small
recall boost on retrieval benchmarks. This project leaves it
configurable via settings and defaults to off, since the corpus is
small enough that the simpler form retrieves correctly.
