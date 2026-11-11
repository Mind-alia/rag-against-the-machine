"""Tests defining the BM25 retrieval contract."""

import pytest

from src.retrieval import BM25Retriever
from src.structdata import DocumentChunk, MinimalSource


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    """Provide a small corpus with clearly distinct vocabulary."""
    return [
        DocumentChunk(
            chunk_id="docs:0:40",
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=0,
            last_character_index=40,
            text="vLLM uses continuous batching for requests.",
        ),
        DocumentChunk(
            chunk_id="engine:0:40",
            file_path="data/raw/vllm-0.10.1/vllm/engine.py",
            first_character_index=0,
            last_character_index=40,
            text="The engine schedules GPU model execution.",
        ),
        DocumentChunk(
            chunk_id="lora:0:40",
            file_path="data/raw/vllm-0.10.1/docs/lora.md",
            first_character_index=0,
            last_character_index=40,
            text="LoRA adapters customize a model efficiently.",
        ),
    ]


def test_search_returns_the_most_relevant_source(
    chunks: list[DocumentChunk],
) -> None:
    """A query with unique terms ranks the matching chunk first."""
    retriever = BM25Retriever(chunks=chunks)

    results = retriever.search("continuous batching", k=1)

    assert results == [
        MinimalSource(
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=0,
            last_character_index=40,
        )
    ]


def test_search_returns_at_most_k_sources(
    chunks: list[DocumentChunk],
) -> None:
    """Search never returns more sources than the caller requested."""
    retriever = BM25Retriever(chunks=chunks)

    results = retriever.search("vLLM model", k=2)

    assert len(results) <= 2


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_a_non_positive_k(
    chunks: list[DocumentChunk],
    k: int,
) -> None:
    """A caller receives a clear error for an invalid result count."""
    retriever = BM25Retriever(chunks=chunks)

    with pytest.raises(ValueError):
        retriever.search("continuous batching", k=k)
