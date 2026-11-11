"""Split loaded documents into traceable chunks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from tqdm import tqdm

from src.structdata import DocumentChunk, LoadedDocument


def chunk_documents(
    documents: list[LoadedDocument],
    max_chunk_size: int = 2000,
    overlap: int = 200,
    show_progress: bool = True,
) -> list[DocumentChunk]:
    """Split documents into bounded chunks with exact source ranges.

    Args:
        documents: Documents loaded from the repository.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by consecutive chunks.
        show_progress: Whether to display a progress bar.

    Returns:
        Chunks covering every document, each with its character range.
    """
    _validate_chunk_settings(max_chunk_size, overlap)

    chunks: list[DocumentChunk] = []

    for document in tqdm(
        documents,
        desc="Chunking documents",
        disable=not show_progress,
    ):
        chunks.extend(
            _chunk_document(
                document,
                max_chunk_size=max_chunk_size,
                overlap=overlap,
            )
        )

    return chunks


def _validate_chunk_settings(
    max_chunk_size: int,
    overlap: int,
) -> None:
    """Validate chunk size and overlap values.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by consecutive chunks.

    Raises:
        ValueError: If a setting is out of range.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to 0 and smaller "
            "than max_chunk_size"
        )


def _chunk_document(
    document: LoadedDocument,
    max_chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Split one document, preferring semantic boundaries when available."""
    text = document.text
    boundaries = _preferred_boundaries(document)
    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(text):
        hard_end = min(start + max_chunk_size, len(text))
        end = _choose_end(
            boundaries=boundaries,
            start=start,
            hard_end=hard_end,
            overlap=overlap,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.file_path}:{start}:{end}",
                file_path=document.file_path,
                first_character_index=start,
                last_character_index=end,
                text=text[start:end],
            )
        )
        if end == len(text):
            break

        start = end - overlap

    return chunks


def _choose_end(
    boundaries: list[int],
    start: int,
    hard_end: int,
    overlap: int,
) -> int:
    """Choose the furthest valid preferred boundary within one chunk."""
    candidates = [
        boundary
        for boundary in boundaries
        if start + overlap < boundary <= hard_end
    ]

    if candidates:
        return max(candidates)
    return hard_end


def _preferred_boundaries(document: LoadedDocument) -> list[int]:
    """Return preferred split positions for the document type."""
    text = document.text
    boundaries = {0, len(text)}

    if document.file_path.endswith(".py"):
        boundaries.update(_python_definition_offsets(text))

    if document.file_path.endswith(".md"):
        boundaries.update(_markdown_heading_offsets(text))

    return sorted(boundaries)


def _markdown_heading_offsets(text: str) -> list[int]:
    """Return offsets where Markdown headings begin."""
    return [
        match.start()
        for match in re.finditer(
            r"^#{1,6}\s+",
            text,
            flags=re.MULTILINE,
        )
    ]


def _python_definition_offsets(text: str) -> list[int]:
    """Return offsets where top-level Python classes and functions begin."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    line_offsets = _line_offsets(text)
    offsets: list[int] = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            offsets.append(line_offsets[node.lineno - 1])

    return offsets


def _line_offsets(text: str) -> list[int]:
    """Return the character offset at the beginning of every line."""
    offsets = [0]

    for index, character in enumerate(text):
        if character == "\n":
            offsets.append(index + 1)

    return offsets


def save_chunks(
    chunks: list[DocumentChunk],
    output_path: Path,
) -> None:
    """Save generated chunks as JSON.

    Args:
        chunks: Chunks to persist.
        output_path: JSON file receiving the chunks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [chunk.model_dump() for chunk in chunks]

    output_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
