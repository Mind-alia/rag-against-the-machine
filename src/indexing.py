"""Build, save, and load the BM25 index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

from src.structdata import DocumentChunk

CHUNKS_FILE_NAME = "chunks.json"

BM25_METHOD = "lucene"
STOPWORDS = "en"
STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: str | list[str], show_progress: bool = False) -> Any:
    """Tokenize a query or chunk corpus consistently for BM25.

    Documents and queries share this function so that a query term and
    the chunk term it should match always reduce to the same token.

    Args:
        texts: A single query or the whole chunk corpus.
        show_progress: Whether to display a progress bar.

    Returns:
        The tokenized corpus expected by ``bm25s``.
    """
    return bm25s.tokenize(
        texts,
        lower=True,
        stopwords=STOPWORDS,
        stemmer=STEMMER,
        show_progress=show_progress,
    )


def build_index(
    chunks: list[DocumentChunk],
    show_progress: bool = False,
) -> bm25s.BM25:
    """Build a BM25 index from non-empty document chunks.

    Args:
        chunks: Chunks to index.
        show_progress: Whether to display a progress bar.

    Returns:
        The BM25 index whose document IDs are chunk positions.

    Raises:
        ValueError: If no chunk is provided.
    """
    if not chunks:
        raise ValueError("cannot build an index from an empty chunk list")

    corpus_tokens = tokenize(
        [chunk.text for chunk in chunks],
        show_progress=show_progress,
    )
    index = bm25s.BM25(method=BM25_METHOD)
    index.index(corpus_tokens, show_progress=show_progress)

    return index


def save_index(
    index: bm25s.BM25,
    chunks: list[DocumentChunk],
    output_directory: Path,
) -> None:
    """Save a BM25 index and the chunks that its document IDs reference.

    Args:
        index: Index to persist.
        chunks: Chunks referenced by the index document IDs.
        output_directory: Directory receiving the index files.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    index.save(output_directory, show_progress=False)

    chunks_path = output_directory / CHUNKS_FILE_NAME
    chunk_data = [chunk.model_dump() for chunk in chunks]
    chunks_path.write_text(
        json.dumps(chunk_data, indent=2),
        encoding="utf-8",
    )


def load_index(index_directory: Path) -> bm25s.BM25:
    """Load a persisted BM25 index without its original corpus text.

    Args:
        index_directory: Directory holding a saved index.

    Returns:
        The loaded BM25 index.
    """
    return bm25s.BM25.load(index_directory, load_corpus=False)


def load_chunks(index_directory: Path) -> list[DocumentChunk]:
    """Load chunks stored alongside a BM25 index.

    Args:
        index_directory: Directory holding a saved index.

    Returns:
        The chunks referenced by the index document IDs.

    Raises:
        ValueError: If the saved chunk file is not a JSON list.
    """
    chunks_path = index_directory / CHUNKS_FILE_NAME
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not isinstance(raw_chunks, list):
        raise ValueError(f"{chunks_path} does not contain a list of chunks")

    return [
        DocumentChunk.model_validate(raw_chunk)
        for raw_chunk in raw_chunks
    ]
