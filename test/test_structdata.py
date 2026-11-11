from pathlib import Path

import pytest
from pydantic import ValidationError

from src.structdata import (
    AnsweredQuestion,
    DocumentChunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


DATASET_ROOT = Path("datasets_public/public")
DATASET_PATHS = sorted(DATASET_ROOT.glob("*/*.json"))
ANSWERED_DOCS = DATASET_ROOT / "AnsweredQuestions/dataset_docs_public.json"

# The public datasets ship with the subject rather than with the code, so
# these checks only run where they were unzipped.
needs_datasets = pytest.mark.skipif(
    not DATASET_PATHS,
    reason="public datasets not available",
)


@needs_datasets
@pytest.mark.parametrize(
    "dataset_path",
    DATASET_PATHS,
    ids=lambda path: path.name,
)
def test_public_datasets_are_valid(dataset_path: Path) -> None:
    """Every shipped dataset parses into the shared RagDataset model."""
    dataset = RagDataset.model_validate_json(dataset_path.read_text())

    assert len(dataset.rag_questions) == 100


@pytest.mark.skipif(
    not ANSWERED_DOCS.is_file(),
    reason="public datasets not available",
)
def test_answered_question_keeps_ground_truth_metadata() -> None:
    """An answered dataset keeps its sources and grading metadata."""
    dataset = RagDataset.model_validate_json(ANSWERED_DOCS.read_text())
    question = dataset.rag_questions[0]

    assert isinstance(question, AnsweredQuestion)
    assert question.answer
    assert question.sources
    assert question.difficulty == "synthetic"
    assert question.is_valid is True


def test_input_question_ignores_unknown_metadata() -> None:
    question = UnansweredQuestion.model_validate(
        {
            "question_id": "question-1",
            "question": "How does vLLM work?",
            "private_metadata": "ignored",
        }
    )

    assert question.model_dump() == {
        "question_id": "question-1",
        "question": "How does vLLM work?",
    }


@pytest.mark.parametrize(
    ("first_index", "last_index"),
    [(-1, 10), (10, 10), (11, 10)],
)
def test_source_rejects_invalid_character_range(
    first_index: int,
    last_index: int,
) -> None:
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="docs/index.md",
            first_character_index=first_index,
            last_character_index=last_index,
        )


def test_document_chunk_preserves_exact_text() -> None:
    text = "  def generate():\n    return True\n"

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        file_path="vllm/generate.py",
        first_character_index=20,
        last_character_index=20 + len(text),
        text=text,
    )

    assert chunk.text == text


def test_output_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MinimalSearchResults.model_validate(
            {
                "question_id": "question-1",
                "question_str": "How does vLLM work?",
                "retrieved_sources": [],
                "unexpected_score": 0.95,
            }
        )


def test_search_results_match_moulinette_json_structure() -> None:
    source = MinimalSource(
        file_path="docs/index.md",
        first_character_index=0,
        last_character_index=100,
    )
    result = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="question-1",
                question_str="How does vLLM work?",
                retrieved_sources=[source],
            )
        ],
        k=5,
    )

    assert result.model_dump() == {
        "search_results": [
            {
                "question_id": "question-1",
                "question_str": "How does vLLM work?",
                "retrieved_sources": [
                    {
                        "file_path": "docs/index.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
        "k": 5,
    }


def test_search_results_reject_more_sources_than_k() -> None:
    sources = [
        MinimalSource(
            file_path="docs/index.md",
            first_character_index=index * 10,
            last_character_index=(index + 1) * 10,
        )
        for index in range(2)
    ]

    with pytest.raises(ValidationError):
        StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="question-1",
                    question_str="How does vLLM work?",
                    retrieved_sources=sources,
                )
            ],
            k=1,
        )


def test_answer_results_include_generated_answer() -> None:
    answer = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="question-1",
                question_str="How does vLLM work?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="docs/index.md",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
                answer="vLLM uses continuous batching.",
            )
        ],
        k=1,
    )

    assert answer.search_results[0].answer == "vLLM uses continuous batching."


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k: int) -> None:
    with pytest.raises(ValidationError):
        StudentSearchResults(search_results=[], k=k)
