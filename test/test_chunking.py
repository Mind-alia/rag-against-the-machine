"""Tests for traceable document chunking."""

import pytest

from src.chunking import chunk_documents
from src.structdata import LoadedDocument


def test_chunk_documents_keeps_a_short_document_intact() -> None:
    """A short document remains one exact, traceable chunk."""
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/docs/guide.md",
        text="A short vLLM guide.",
    )

    chunks = chunk_documents([document], max_chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].file_path == document.file_path
    assert chunks[0].first_character_index == 0
    assert chunks[0].last_character_index == len(document.text)
    assert chunks[0].text == document.text


def test_chunk_documents_preserves_ranges_and_overlap() -> None:
    """Long documents become bounded chunks with exact source ranges."""
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/vllm/engine.py",
        text=text,
    )

    chunks = chunk_documents([document], max_chunk_size=12, overlap=4)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 12 for chunk in chunks)
    assert all(
        chunk.text == text[
            chunk.first_character_index:chunk.last_character_index
        ]
        for chunk in chunks
    )
    assert all(
        current.first_character_index < previous.last_character_index
        for previous, current in zip(chunks, chunks[1:])
    )
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("max_chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10)],
)
def test_chunk_documents_rejects_invalid_size_settings(
    max_chunk_size: int,
    overlap: int,
) -> None:
    """Invalid size settings fail before any chunk is created."""
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/docs/guide.md",
        text="content",
    )

    with pytest.raises(ValueError):
        chunk_documents(
            [document],
            max_chunk_size=max_chunk_size,
            overlap=overlap,
        )


def test_chunk_documents_prefers_markdown_heading_boundaries() -> None:
    """A Markdown section heading begins a chunk when it fits the limit."""
    text = (
        "# Overview\n\n"
        "Short introduction.\n\n"
        "## Configuration\n\n"
        "Configuration details that belong to their own section."
    )
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/docs/guide.md",
        text=text,
    )

    chunks = chunk_documents([document], max_chunk_size=60, overlap=0)

    assert any(chunk.text.startswith("## Configuration") for chunk in chunks)


def test_chunk_documents_prefers_python_top_level_definitions() -> None:
    """Independent Python functions begin separate chunks when possible."""
    text = (
        "def create_engine():\n"
        "    return 'engine'\n\n"
        "def stop_engine():\n"
        "    return 'stopped'\n"
    )
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/vllm/engine.py",
        text=text,
    )

    chunks = chunk_documents([document], max_chunk_size=48, overlap=0)

    assert any(chunk.text.startswith("def create_engine") for chunk in chunks)
    assert any(chunk.text.startswith("def stop_engine") for chunk in chunks)
