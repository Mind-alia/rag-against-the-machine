"""Evaluate RAG search results against answered datasets."""

from __future__ import annotations

from pathlib import Path

from src.structdata import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

RECALL_LEVELS = (1, 3, 5, 10)


def evaluate_search_results(
    student_results_path: Path,
    dataset_path: Path,
    k: int = 10,
    minimal_iou_threshold: float = 0.05,
) -> dict[str, float | int]:
    """Calculate Recall@1, Recall@3, Recall@5, and Recall@10.

    The metric mirrors the official evaluator: a ground-truth source
    counts as found when its character range overlaps a retrieved range
    of the same file by more than ``minimal_iou_threshold``.

    Args:
        student_results_path: Saved search results to evaluate.
        dataset_path: Answered dataset used as ground truth.
        k: Maximum number of sources considered per question.
        minimal_iou_threshold: Minimum overlap counting as found.

    Returns:
        The number of evaluated questions and one recall per level.

    Raises:
        ValueError: If k is not positive or the dataset has no answers.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    student_results = StudentSearchResults.model_validate_json(
        student_results_path.read_text(encoding="utf-8")
    )
    dataset = RagDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )

    expected_questions = {
        question.question_id: question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    }

    if not expected_questions:
        raise ValueError(
            f"{dataset_path} contains no answered question; "
            "evaluation needs a dataset with ground-truth sources"
        )

    student_by_id = _index_by_question_id(student_results.search_results)

    metrics: dict[str, float | int] = {
        "questions_evaluated": len(expected_questions),
    }

    for level in RECALL_LEVELS:
        recalls = [
            _question_recall(
                expected_question.sources,
                _retrieved_for(student_by_id, question_id, min(level, k)),
                minimal_iou_threshold,
            )
            for question_id, expected_question in expected_questions.items()
        ]

        metrics[f"recall_at_{level}"] = sum(recalls) / len(recalls)

    return metrics


def _index_by_question_id(
    search_results: list[MinimalSearchResults],
) -> dict[str, MinimalSearchResults]:
    """Map every student result to its question identifier.

    Args:
        search_results: Results loaded from the student JSON file.

    Returns:
        The results keyed by question identifier.
    """
    return {result.question_id: result for result in search_results}


def _retrieved_for(
    student_by_id: dict[str, MinimalSearchResults],
    question_id: str,
    level: int,
) -> list[MinimalSource]:
    """Return the top sources a student returned for one question.

    Args:
        student_by_id: Student results keyed by question identifier.
        question_id: Question being evaluated.
        level: Number of sources to keep.

    Returns:
        The retained sources, or an empty list for a missing question.
    """
    student_result = student_by_id.get(question_id)

    if student_result is None:
        return []

    return student_result.retrieved_sources[:level]


def _question_recall(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    minimal_iou_threshold: float,
) -> float:
    """Calculate how many expected sources were found for one question.

    Args:
        expected_sources: Ground-truth sources for the question.
        retrieved_sources: Sources the system returned.
        minimal_iou_threshold: Minimum overlap counting as found.

    Returns:
        The ratio of ground-truth sources that were found.
    """
    if not expected_sources:
        return 1.0

    if not retrieved_sources:
        return 0.0

    found_sources = sum(
        any(
            _source_matches(
                expected_source,
                retrieved_source,
                minimal_iou_threshold,
            )
            for retrieved_source in retrieved_sources
        )
        for expected_source in expected_sources
    )

    return found_sources / len(expected_sources)


def _source_matches(
    expected_source: MinimalSource,
    retrieved_source: MinimalSource,
    minimal_iou_threshold: float,
) -> bool:
    """Return whether two source ranges overlap enough.

    Args:
        expected_source: Ground-truth source range.
        retrieved_source: Retrieved source range.
        minimal_iou_threshold: Minimum overlap counting as found.

    Returns:
        Whether both ranges describe the same passage.
    """
    if expected_source.file_path != retrieved_source.file_path:
        return False

    return _intersection_over_union(
        expected_source,
        retrieved_source,
    ) > minimal_iou_threshold


def _intersection_over_union(
    expected_source: MinimalSource,
    retrieved_source: MinimalSource,
) -> float:
    """Return the overlap ratio between two character ranges.

    Args:
        expected_source: Ground-truth source range.
        retrieved_source: Retrieved source range.

    Returns:
        The intersection over union of both ranges.
    """
    intersection = max(
        0,
        min(
            expected_source.last_character_index,
            retrieved_source.last_character_index,
        )
        - max(
            expected_source.first_character_index,
            retrieved_source.first_character_index,
        ),
    )

    expected_length = (
        expected_source.last_character_index
        - expected_source.first_character_index
    )
    retrieved_length = (
        retrieved_source.last_character_index
        - retrieved_source.first_character_index
    )
    union = expected_length + retrieved_length - intersection

    if union <= 0:
        return 0.0

    return intersection / union
