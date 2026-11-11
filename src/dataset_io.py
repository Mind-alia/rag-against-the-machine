"""Build moulinette-compatible search-result files from question datasets."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.retrieval import BM25Retriever
from src.structdata import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)


def search_dataset(
    dataset_path: Path,
    retriever: BM25Retriever,
    k: int,
    save_directory: Path,
    show_progress: bool = True,
) -> Path:
    """Search every dataset question and save evaluator-compatible JSON."""
    if k <= 0:
        raise ValueError("k must be positive")

    dataset = RagDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    search_results: list[MinimalSearchResults] = []
    questions = tqdm(
        dataset.rag_questions,
        desc="Searching questions",
        disable=not show_progress,
    )

    for question in questions:
        search_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=retriever.search(question.question, k),
            )
        )

    student_results = StudentSearchResults(
        search_results=search_results,
        k=k,
    )
    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / dataset_path.name
    output_path.write_text(
        student_results.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path
