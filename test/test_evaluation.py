"""Tests defining the recall@k contract of the evaluation system."""

import json
from pathlib import Path

import pytest

from src.evaluation import evaluate_search_results

GUIDE = "data/raw/vllm-0.10.1/docs/guide.md"
ENGINE = "data/raw/vllm-0.10.1/vllm/engine.py"


def _source(
    file_path: str,
    first: int,
    last: int,
) -> dict[str, object]:
    """Build one minimal source range."""
    return {
        "file_path": file_path,
        "first_character_index": first,
        "last_character_index": last,
    }


def _write_dataset(
    path: Path,
    questions: list[dict[str, object]],
) -> Path:
    """Write an answered ground-truth dataset."""
    path.write_text(
        json.dumps({"rag_questions": questions}),
        encoding="utf-8",
    )
    return path


def _write_results(
    path: Path,
    results: list[dict[str, object]],
    k: int = 10,
) -> Path:
    """Write a student search-results file."""
    path.write_text(
        json.dumps({"search_results": results, "k": k}),
        encoding="utf-8",
    )
    return path


def test_a_perfect_run_scores_one_at_every_level(tmp_path: Path) -> None:
    """Retrieving the exact ground-truth range scores a full recall."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [_source(GUIDE, 0, 100)],
            }
        ],
    )
    results = _write_results(
        tmp_path / "results.json",
        [
            {
                "question_id": "q1",
                "question_str": "What is batching?",
                "retrieved_sources": [_source(GUIDE, 0, 100)],
            }
        ],
    )

    metrics = evaluate_search_results(results, dataset)

    assert metrics["questions_evaluated"] == 1
    assert metrics["recall_at_1"] == 1.0
    assert metrics["recall_at_10"] == 1.0


def test_recall_grows_with_the_rank_of_the_correct_source(
    tmp_path: Path,
) -> None:
    """A correct source ranked third counts from Recall@3 onwards."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [_source(GUIDE, 0, 100)],
            }
        ],
    )
    results = _write_results(
        tmp_path / "results.json",
        [
            {
                "question_id": "q1",
                "question_str": "What is batching?",
                "retrieved_sources": [
                    _source(ENGINE, 0, 100),
                    _source(ENGINE, 100, 200),
                    _source(GUIDE, 0, 100),
                ],
            }
        ],
    )

    metrics = evaluate_search_results(results, dataset)

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0


def test_a_partial_overlap_below_the_threshold_is_not_found(
    tmp_path: Path,
) -> None:
    """An overlap of five percent or less does not count as found."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [_source(GUIDE, 0, 1000)],
            }
        ],
    )
    results = _write_results(
        tmp_path / "results.json",
        [
            {
                "question_id": "q1",
                "question_str": "What is batching?",
                "retrieved_sources": [_source(GUIDE, 990, 1990)],
            }
        ],
    )

    metrics = evaluate_search_results(results, dataset)

    assert metrics["recall_at_1"] == 0.0


def test_only_part_of_the_expected_sources_scores_a_partial_recall(
    tmp_path: Path,
) -> None:
    """Finding one of two ground-truth sources scores one half."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [
                    _source(GUIDE, 0, 100),
                    _source(ENGINE, 0, 100),
                ],
            }
        ],
    )
    results = _write_results(
        tmp_path / "results.json",
        [
            {
                "question_id": "q1",
                "question_str": "What is batching?",
                "retrieved_sources": [_source(GUIDE, 0, 100)],
            }
        ],
    )

    metrics = evaluate_search_results(results, dataset)

    assert metrics["recall_at_1"] == 0.5


def test_a_question_without_student_results_scores_zero(
    tmp_path: Path,
) -> None:
    """A missing question counts as a miss instead of crashing."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [_source(GUIDE, 0, 100)],
            }
        ],
    )
    results = _write_results(tmp_path / "results.json", [])

    metrics = evaluate_search_results(results, dataset)

    assert metrics["recall_at_5"] == 0.0


def test_a_question_without_ground_truth_sources_scores_one(
    tmp_path: Path,
) -> None:
    """An empty ground truth cannot be missed, as in the evaluator."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What is batching?",
                "answer": "Continuous batching.",
                "sources": [],
            }
        ],
    )
    results = _write_results(tmp_path / "results.json", [])

    metrics = evaluate_search_results(results, dataset)

    assert metrics["recall_at_1"] == 1.0


def test_evaluation_rejects_a_dataset_without_answers(
    tmp_path: Path,
) -> None:
    """Evaluating unanswered questions produces a clear error."""
    dataset = _write_dataset(
        tmp_path / "dataset.json",
        [{"question_id": "q1", "question": "What is batching?"}],
    )
    results = _write_results(tmp_path / "results.json", [])

    with pytest.raises(ValueError, match="no answered question"):
        evaluate_search_results(results, dataset)


def test_evaluation_rejects_a_non_positive_k(tmp_path: Path) -> None:
    """A caller receives a clear error for an invalid result count."""
    dataset = _write_dataset(tmp_path / "dataset.json", [])
    results = _write_results(tmp_path / "results.json", [])

    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_search_results(results, dataset, k=0)
