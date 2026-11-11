"""Tests for persisted BM25 indexes."""

from pathlib import Path

from src.indexing import build_index, load_chunks, load_index, save_index
from src.retrieval import BM25Retriever
from src.structdata import DocumentChunk, MinimalSource


def test_saved_index_can_be_loaded_and_searched(
    tmp_path: Path,
) -> None:
    """An index reload keeps its chunk-to-source mapping intact."""
    chunks = [
        DocumentChunk(
            chunk_id="batching:0:42",
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=0,
            last_character_index=42,
            text="vLLM uses continuous batching for requests.",
        ),
        DocumentChunk(
            chunk_id="engine:0:40",
            file_path="data/raw/vllm-0.10.1/vllm/engine.py",
            first_character_index=0,
            last_character_index=40,
            text="The engine schedules GPU model execution.",
        ),
    ]
    index_directory = tmp_path / "bm25_index"

    index = build_index(chunks)
    save_index(index, chunks, index_directory)

    loaded_chunks = load_chunks(index_directory)
    loaded_index = load_index(index_directory)
    retriever = BM25Retriever(chunks=loaded_chunks, index=loaded_index)

    assert (index_directory / "chunks.json").is_file()
    assert retriever.search("continuous batching", k=1) == [
        MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=0,
            last_character_index=42,
        )
    ]
