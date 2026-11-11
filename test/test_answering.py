"""Tests for answer assembly that do not require the Qwen model."""

import json
from pathlib import Path

from src.answering import (
    AnswerGenerator,
    _build_context,
    _strip_reasoning,
    answer_dataset,
)
from src.structdata import MinimalSource, StudentSearchResultsAndAnswer

GUIDE = "data/raw/vllm-0.10.1/docs/guide.md"


class StubGenerator(AnswerGenerator):
    """Answer generator that never loads a model."""

    def answer(
        self,
        question: str,
        sources: list[MinimalSource],
        repository_path: Path,
    ) -> str:
        """Return a deterministic answer for the question."""
        return f"Answer to {question}"


def test_strip_reasoning_removes_a_thinking_block() -> None:
    """A reasoning trace never reaches the generated answer."""
    raw = "<think>\nLet me check the docs.\n</think>\n\nvLLM batches."

    assert _strip_reasoning(raw) == "vLLM batches."


def test_strip_reasoning_removes_an_unclosed_thinking_block() -> None:
    """A truncated reasoning trace is still removed."""
    raw = "<think>Let me check.</think>vLLM batches.</think> Final."

    assert _strip_reasoning(raw) == "Final."


def test_strip_reasoning_keeps_a_plain_answer() -> None:
    """An answer without a reasoning trace is only trimmed."""
    assert _strip_reasoning("  vLLM batches requests.  ") == (
        "vLLM batches requests."
    )


def test_build_context_quotes_the_exact_character_range(
    tmp_path: Path,
) -> None:
    """The context holds the retrieved passage and cites its range."""
    documentation = tmp_path / "docs"
    documentation.mkdir()
    (documentation / "guide.md").write_text(
        "HEADERvLLM batches requests.FOOTER",
        encoding="utf-8",
    )

    context = _build_context(
        [
            MinimalSource(
                file_path=GUIDE,
                first_character_index=6,
                last_character_index=28,
            )
        ],
        tmp_path,
    )

    assert "vLLM batches requests." in context
    assert f"[{GUIDE}:6-28]" in context
    assert "FOOTER" not in context


def test_build_context_stays_within_the_character_budget(
    tmp_path: Path,
) -> None:
    """A long passage is truncated instead of overflowing the prompt."""
    documentation = tmp_path / "docs"
    documentation.mkdir()
    (documentation / "guide.md").write_text("x" * 5000, encoding="utf-8")

    context = _build_context(
        [
            MinimalSource(
                file_path=GUIDE,
                first_character_index=0,
                last_character_index=5000,
            )
        ],
        tmp_path,
        max_characters=200,
    )

    assert len(context) <= 200


def test_build_context_skips_sources_that_cannot_be_read(
    tmp_path: Path,
) -> None:
    """An unreadable source is ignored rather than raising."""
    context = _build_context(
        [
            MinimalSource(
                file_path=GUIDE,
                first_character_index=0,
                last_character_index=10,
            )
        ],
        tmp_path,
    )

    assert context == ""


def test_answer_dataset_writes_the_expected_output_model(
    tmp_path: Path,
) -> None:
    """Generated answers follow the StudentSearchResultsAndAnswer model."""
    results_path = tmp_path / "dataset_docs_public.json"
    results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question_str": "What is batching?",
                        "retrieved_sources": [
                            {
                                "file_path": GUIDE,
                                "first_character_index": 0,
                                "last_character_index": 40,
                            }
                        ],
                    }
                ],
                "k": 10,
            }
        ),
        encoding="utf-8",
    )
    save_directory = tmp_path / "answers"

    output_path = answer_dataset(
        student_search_results_path=results_path,
        generator=StubGenerator(),
        repository_path=tmp_path,
        save_directory=save_directory,
        show_progress=False,
    )

    saved = StudentSearchResultsAndAnswer.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert output_path == save_directory / results_path.name
    assert saved.k == 10
    assert saved.search_results[0].answer == "Answer to What is batching?"
