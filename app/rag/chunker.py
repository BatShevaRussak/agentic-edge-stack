"""Document chunking via LangChain's recursive character splitter."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.errors import IngestionError
from app.rag.types import Chunk

_MARKDOWN_SEPARATORS: list[str] = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


class Chunker:
    """Splits text and files into ``Chunk`` objects on natural boundaries."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
        self.separators = separators or _MARKDOWN_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str, source: str) -> list[Chunk]:
        """Split a string into chunks tagged with the given source."""
        if not text or not text.strip():
            return []

        pieces = self._splitter.split_text(text)
        return [
            Chunk(text=piece, source=source, chunk_index=i)
            for i, piece in enumerate(pieces)
        ]

    def split_file(self, path: Path) -> list[Chunk]:
        """Read a UTF-8 file and split it into chunks.

        Raises:
            IngestionError: if the file cannot be read.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(f"Failed to read {path}: {exc}") from exc

        return self.split_text(text, source=path.as_posix())
