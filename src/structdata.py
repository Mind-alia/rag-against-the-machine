"""Validated data contracts used by the RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _InputModel(BaseModel):
    """Base model for datasets that may contain extra metadata."""

    model_config = ConfigDict(extra="ignore")


class _StrictModel(BaseModel):
    """Base model for internal data and evaluator output."""

    model_config = ConfigDict(extra="forbid")


class MinimalSource(_StrictModel):
    """Character range locating a retrieved passage in a source file."""

    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_character_range(self) -> MinimalSource:
        """Ensure that the source range is ordered and non-empty."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )
        return self


class UnansweredQuestion(_InputModel):
    """Question loaded from an unanswered RAG dataset."""

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class AnsweredQuestion(UnansweredQuestion):
    """Question with its reference answer and source passages."""

    answer: str = Field(min_length=1)
    sources: list[MinimalSource]
    difficulty: str | None = None
    is_valid: bool | None = None


class RagDataset(_InputModel):
    """Top-level structure shared by answered and unanswered datasets."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class DocumentChunk(MinimalSource):
    """Searchable text chunk and its exact location in the corpus."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MinimalSearchResults(_StrictModel):
    """Retrieved sources for one question, as expected by the evaluator."""

    question_id: str = Field(min_length=1)
    question_str: str = Field(min_length=1)
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources and generated answer for one question."""

    answer: str = Field(min_length=1)


def _validate_source_counts(
    search_results: Sequence[MinimalSearchResults],
    k: int,
) -> None:
    for result in search_results:
        if len(result.retrieved_sources) > k:
            raise ValueError(
                "retrieved_sources cannot contain more than k sources"
            )


class StudentSearchResults(_StrictModel):
    """Top-level JSON structure for search-only evaluation."""

    search_results: list[MinimalSearchResults]
    k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_source_counts(self) -> StudentSearchResults:
        """Ensure each question returns at most the requested top-k."""
        _validate_source_counts(self.search_results, self.k)
        return self


class StudentSearchResultsAndAnswer(_StrictModel):
    """Top-level JSON structure containing retrieval and LLM answers."""

    search_results: list[MinimalAnswer]
    k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_source_counts(self) -> StudentSearchResultsAndAnswer:
        """Ensure each question returns at most the requested top-k."""
        _validate_source_counts(self.search_results, self.k)
        return self


class LoadedDocument(_StrictModel):
    """Complete text loaded from one repository file."""

    file_path: str = Field(min_length=1)
    text: str = Field(min_length=1)


__all__ = [
    "AnsweredQuestion",
    "DocumentChunk",
    "MinimalAnswer",
    "MinimalSearchResults",
    "MinimalSource",
    "RagDataset",
    "StudentSearchResults",
    "StudentSearchResultsAndAnswer",
    "UnansweredQuestion",
    "LoadedDocument",
]
