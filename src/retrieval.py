"""Search document chunks with BM25."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.indexing import build_index, load_chunks, load_index, tokenize
from src.structdata import DocumentChunk, MinimalSource


class BM25Retriever(BaseModel):
    """Retrieve source ranges ranked by their BM25 relevance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list[DocumentChunk] = Field(min_length=1)
    index: Any = None

    @model_validator(mode="after")
    def build_missing_index(self) -> BM25Retriever:
        """Build the BM25 index when none was supplied.

        Returns:
            The retriever, guaranteed to hold a usable index.
        """
        if self.index is None:
            self.index = build_index(self.chunks)

        return self

    @classmethod
    def load(cls, index_directory: Path) -> BM25Retriever:
        """Load a retriever with a persisted index and its chunks.

        Args:
            index_directory: Directory holding a saved BM25 index.

        Returns:
            A retriever ready to answer queries.
        """
        return cls(
            chunks=load_chunks(index_directory),
            index=load_index(index_directory),
        )

    def search(self, question: str, k: int) -> list[MinimalSource]:
        """Return at most k source ranges ranked by query relevance.

        Args:
            question: Natural language query.
            k: Maximum number of sources to return.

        Returns:
            The retrieved source ranges, most relevant first.

        Raises:
            ValueError: If k is not strictly positive.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        results = self.index.retrieve(
            tokenize(question),
            k=min(k, len(self.chunks)),
            show_progress=False,
        )

        return [
            self._to_source(self.chunks[int(document_id)])
            for document_id in results.documents[0]
        ]

    @staticmethod
    def _to_source(chunk: DocumentChunk) -> MinimalSource:
        """Convert an internal chunk into evaluator-compatible output.

        Args:
            chunk: Indexed chunk selected by the retriever.

        Returns:
            The minimal source expected by the evaluator.
        """
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
