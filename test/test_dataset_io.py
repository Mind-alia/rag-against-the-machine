"""Tests for dataset search-result generation."""

import json
from pathlib import Path

from src.dataset_io import search_dataset
from src.retrieval import BM25Retriever
from src.structdata import DocumentChunk, StudentSearchResults


def test_search_dataset_saves_moulinette_compatible_results(
    tmp_path: Path,
) -> None:
    """Each input question receives at most k retrieved sources."""
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "question-1",
                        "question": "How does continuous batching work?",
                    },
                    {
                        "question_id": "question-2",
                        "question": "What schedules GPU execution?",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    retriever = BM25Retriever(
        chunks=[
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
    )
    save_directory = tmp_path / "search_results"

    output_path = search_dataset(
        dataset_path=dataset_path,
        retriever=retriever,
        k=1,
        save_directory=save_directory,
        show_progress=False,
    )

    saved_results = StudentSearchResults.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert output_path == save_directory / "questions.json"
    assert saved_results.k == 1
    assert [result.question_id for result in saved_results.search_results] == [
        "question-1",
        "question-2",
    ]
    assert all(
        len(result.retrieved_sources) <= 1
        for result in saved_results.search_results
    )
