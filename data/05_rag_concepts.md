# Retrieval-Augmented Generation (RAG) - Concepts

## Why RAG exists

A language model's knowledge is frozen at training time. To answer
questions about private documents, recent events, or any source it
did not see during pretraining, the model needs context supplied at
inference time. Retrieval-Augmented Generation, introduced by Lewis
et al. (Facebook AI, 2020), is the standard pattern for doing this:
relevant passages are fetched from an external store and injected
into the prompt before the model generates an answer.

RAG is preferred over fine-tuning for factual knowledge because it:

- updates instantly when documents change (no retraining),
- provides citable sources (each retrieved chunk has a known origin),
- keeps the model itself general-purpose and reusable across domains.

## The pipeline

A minimal RAG system has five stages:

1. **Ingest.** Load source documents (Markdown, PDF, HTML, ...).
2. **Chunk.** Split each document into smaller passages.
3. **Embed.** Convert each chunk into a fixed-length vector with an
   embedding model.
4. **Index.** Store vectors in a structure that supports nearest
   neighbor search (FAISS, Qdrant, ChromaDB, Pinecone, ...).
5. **Retrieve and generate.** At query time, embed the user question,
   look up the top-K nearest chunks, and inject them into the prompt
   sent to the LLM.

## Chunking strategies

Chunk quality dominates retrieval quality. The four common strategies,
from simplest to most sophisticated:

- **Fixed-size character chunks.** Split every N characters. Simple
  but cuts mid-sentence and loses semantic coherence.
- **Recursive character splitting.** Split on a hierarchy of
  separators (paragraphs, then sentences, then spaces) so chunks
  respect natural boundaries. This is the default in LangChain's
  `RecursiveCharacterTextSplitter` and the strategy this project
  uses.
- **Markdown / structural splitting.** First split by headings, then
  recursively within each section. Excellent for technical
  documentation where sections are self-contained.
- **Semantic chunking.** Embed sentences, then merge adjacent
  sentences whose embeddings are similar enough. Highest quality but
  significantly more expensive.

Chunk size is a tradeoff: smaller chunks improve precision (the
retrieved snippet is more on-topic) but lose surrounding context;
larger chunks include more context but dilute the embedding signal.
A common starting point is 300-700 characters with 10-15% overlap so
that information near boundaries is not lost.

## Top-K retrieval and the score threshold

After embedding the query, the index returns the K vectors with the
highest similarity. K is typically 3 to 10: too few risks missing
the right chunk, too many wastes the LLM's context window and
introduces distractors that hurt answer quality.

A score threshold filters out chunks that fall below a similarity
floor (for example, cosine < 0.3). This protects against off-topic
queries where every retrieved chunk is weakly related and would
mislead the LLM into hallucinating. When all candidates fall below
the threshold, the system can answer "I don't have that information"
instead of producing a confident wrong answer.

## Prompt augmentation

The retrieved chunks are stitched into the LLM prompt with three
elements:

1. **A system instruction** that constrains the model to answer
   from the provided context and to admit ignorance when the
   context is silent.
2. **The retrieved context**, formatted with source markers so the
   model can cite where each fact came from.
3. **The user question**, restated verbatim.

Llama 3.2 Instruct follows this pattern reliably. Common failure
modes are mitigated by explicit phrasing in the system message:
"Use ONLY the provided context. If the answer is not present, say
you do not know."

## Where RAG fits in this project

RAG is Part 2 of the assignment. It supplies the model with
domain-specific knowledge (in this case, technical facts about the
stack itself). In Part 3, the same retrieval call will be wrapped
as an Agent tool so the LLM can decide *when* to retrieve rather
than always retrieving. In Part 4, the full RAG-augmented response
is streamed back to the client over Server-Sent Events.
