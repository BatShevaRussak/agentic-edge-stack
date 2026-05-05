"""Exception hierarchy for the RAG layer."""


class RAGError(Exception):
    """Base class for failures inside the RAG layer."""


class IngestionError(RAGError):
    """Raised when reading, chunking, or indexing the corpus fails."""


class RetrievalError(RAGError):
    """Raised when embedding a query or searching the index fails."""


class EmptyCorpusError(IngestionError):
    """Raised when the data directory contains no ingestible documents."""
