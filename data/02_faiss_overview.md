# FAISS - Overview and Index Selection

## What FAISS is

FAISS (Facebook AI Similarity Search) is an open-source library for
efficient similarity search and clustering of dense vectors. It is
written in C++ with Python bindings and is the de facto standard for
nearest-neighbor search in machine-learning systems. Released by
Meta in 2017 (Johnson, Douze & Jegou), FAISS powers vector search in
production at companies including Meta, Spotify, and OpenAI's
internal tools.

## Core abstraction: the Index

A FAISS `Index` is a data structure that stores a set of vectors and
supports a `search(query, k)` operation returning the k nearest
neighbors. Indexes differ in three dimensions:

- **Distance metric.** Inner product (`IndexFlatIP`) or L2
  (`IndexFlatL2`). Cosine similarity can be obtained by L2-normalizing
  vectors and using inner product.
- **Search exactness.** Exact search examines every vector; approximate
  search trades a small recall loss for large speedups.
- **Memory layout.** Vectors can be stored as full floats, scalar
  quantized, or product quantized for compression.

## When to use each index family

### IndexFlatIP / IndexFlatL2 (exact, brute force)

Computes the metric against every stored vector. Time complexity is
O(N * D) per query, where N is the number of vectors and D is the
dimension. Best when:

- N is small (under ~10,000 vectors).
- Recall must be 100%.
- Index build time must be near zero.

This is the right choice for small in-memory RAG corpora.

### IndexIVFFlat (approximate, inverted file)

Partitions the vector space into Voronoi cells using k-means, then
searches only the most relevant cells at query time. Sublinear in N
when tuned, but requires training on a representative sample. Used
when N is in the millions.

### IndexHNSW (approximate, hierarchical graph)

Builds a navigable small-world graph over the vectors. Excellent
recall-latency tradeoff for million-scale corpora. Higher memory
overhead than IVF but no training step.

### Product Quantization variants (IndexIVFPQ, IndexHNSWPQ)

Compress vectors to a few bytes each using product quantization. Cuts
memory by 32x or more at a small recall cost. Used when the corpus
exceeds RAM.

## Cosine similarity via inner product

Cosine similarity between two vectors x and y is defined as
x . y / (||x|| * ||y||). If both vectors are pre-normalized to unit
L2 norm, the denominators become 1 and cosine similarity reduces to
the plain inner product x . y. FAISS therefore does not provide a
dedicated cosine index - the standard pattern is to L2-normalize
vectors before insertion and use `IndexFlatIP`. This is the approach
this project uses.

## Why IndexFlatIP for this project

The corpus is ~50 to 200 chunks. At that scale, brute-force search
takes microseconds, so any approximation only adds complexity without
real benefit. `IndexFlatIP` over normalized BGE embeddings gives
exact cosine search with zero tuning and a single line of code per
operation. Approximate indexes (IVF, HNSW) start to pay off at
roughly 10,000+ vectors, well above the size of an in-memory RAG
corpus.
