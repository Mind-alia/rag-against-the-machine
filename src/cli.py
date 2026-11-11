"""Command-line actions for the RAG application."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from src.chunking import chunk_documents, save_chunks
from src.dataset_io import search_dataset as create_search_dataset
from src.errors import RagError
from src.evaluation import evaluate_search_results
from src.file_loader import find_repository, load_documents
from src.indexing import CHUNKS_FILE_NAME, build_index, save_index
from src.retrieval import BM25Retriever


def _describe_validation_error(error: ValidationError) -> str:
    """Summarise a pydantic failure as a single readable sentence.

    Args:
        error: The validation failure raised while reading a JSON file.

    Returns:
        A short message naming the field that caused the failure.
    """
    details = error.errors()

    if not details:
        return "the data does not match the expected format"

    first = next(
        (detail for detail in details if detail["type"] == "missing"),
        details[0],
    )

    if first["type"] == "json_invalid":
        return str(first["msg"])

    location = ".".join(str(part) for part in first["loc"])

    return f"{location or 'document'}: {first['msg']}"


@contextmanager
def _reported_as(action: str) -> Iterator[None]:
    """Translate low-level failures into clear command errors.

    Args:
        action: Short description of the step being performed.

    Yields:
        Nothing; the wrapped block runs inside the translation.
    """
    try:
        yield
    except RagError:
        raise
    except ValidationError as error:
        raise RagError(
            f"{action}: {_describe_validation_error(error)}"
        ) from error
    except json.JSONDecodeError as error:
        raise RagError(f"{action}: invalid JSON ({error})") from error
    except OSError as error:
        raise RagError(f"{action}: {error.strerror or error}") from error
    except ValueError as error:
        raise RagError(f"{action}: {error}") from error


def _as_text(value: object, name: str) -> str:
    """Return a non-empty string for a command-line argument."""
    text = value if isinstance(value, str) else str(value)

    if not text.strip():
        raise RagError(f"{name} must be a non-empty string")

    return text


def _as_positive_int(value: object, name: str) -> int:
    """Return a strictly positive integer for a command-line argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RagError(f"{name} must be a positive integer")

    if value <= 0:
        raise RagError(f"{name} must be a positive integer, got {value}")

    return value


def _as_non_negative_int(value: object, name: str) -> int:
    """Return a positive or null integer for a command-line argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RagError(f"{name} must be an integer greater than or equal to 0")

    if value < 0:
        raise RagError(f"{name} cannot be negative, got {value}")

    return value


def _as_ratio(value: object, name: str) -> float:
    """Return a ratio between 0 and 1 for a command-line argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagError(f"{name} must be a number between 0 and 1")

    if not 0 <= value <= 1:
        raise RagError(f"{name} must be between 0 and 1, got {value}")

    return float(value)


def _existing_file(value: object, name: str) -> Path:
    """Return an existing file path for a command-line argument."""
    path = Path(_as_text(value, name)).expanduser()

    if path.is_dir():
        raise RagError(f"{name} is a directory, not a file: {path}")

    if not path.is_file():
        raise RagError(f"{name}: file not found: {path}")

    return path


def _resolve_repository(repository_path: object) -> Path:
    """Locate the vLLM repository, searching the usual layouts.

    Args:
        repository_path: Explicit path, or None to search for it.

    Returns:
        The directory holding the vLLM sources.

    Raises:
        RagError: If the repository cannot be found.
    """
    explicit = (
        None
        if repository_path is None
        else _as_text(repository_path, "repository_path")
    )

    try:
        return find_repository(explicit)
    except ValueError as error:
        raise RagError(f"repository_path: {error}") from error


def _load_retriever(index_directory: object) -> BM25Retriever:
    """Load a saved index, explaining how to create it when missing."""
    directory = Path(
        _as_text(index_directory, "index_directory")
    ).expanduser()

    if not (directory / CHUNKS_FILE_NAME).is_file():
        raise RagError(
            f"No index found in {directory}. "
            "Run `uv run python -m src index` first."
        )

    with _reported_as(f"Loading the index from {directory}"):
        return BM25Retriever.load(directory)


class RagCli:
    """Expose RAG operations through Python Fire."""

    def index(
        self,
        repository_path: str | None = None,
        index_directory: str = "data/processed/bm25_index",
        chunks_path: str = "data/processed/chunks/chunks.json",
        max_chunk_size: int = 2000,
        overlap: int = 200,
    ) -> str:
        """Load vLLM, create chunks, and save a BM25 index.

        Args:
            repository_path: vLLM repository; searched when omitted.
            index_directory: Directory where the BM25 index is written.
            chunks_path: JSON file where the chunks are also written.
            max_chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters shared by consecutive chunks.

        Returns:
            A summary of how many documents and chunks were indexed.
        """
        repository = _resolve_repository(repository_path)
        output_directory = Path(
            _as_text(index_directory, "index_directory")
        ).expanduser()
        chunks_file = Path(_as_text(chunks_path, "chunks_path")).expanduser()
        chunk_size = _as_positive_int(max_chunk_size, "max_chunk_size")
        chunk_overlap = _as_non_negative_int(overlap, "overlap")

        if chunk_overlap >= chunk_size:
            raise RagError(
                "overlap must be smaller than max_chunk_size "
                f"({chunk_overlap} >= {chunk_size})"
            )

        with _reported_as(f"Reading the repository {repository}"):
            documents = load_documents(repository)

        if not documents:
            raise RagError(
                f"No supported file found in {repository}. "
                "Expected a vLLM repository with docs/ and vllm/ folders."
            )

        with _reported_as("Splitting the documents into chunks"):
            chunks = chunk_documents(
                documents,
                max_chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

        with _reported_as(f"Saving the chunks to {chunks_file}"):
            save_chunks(chunks, chunks_file)

        with _reported_as(f"Saving the index to {output_directory}"):
            index = build_index(chunks, show_progress=True)
            save_index(index, chunks, output_directory)

        return (
            f"Indexed {len(documents)} documents into "
            f"{len(chunks)} chunks. Index saved under {output_directory}."
        )

    def search(
        self,
        question: str,
        k: int = 5,
        index_directory: str = "data/processed/bm25_index",
    ) -> list[dict[str, object]]:
        """Search a saved BM25 index and return source locations.

        Args:
            question: Natural language query to search for.
            k: Maximum number of sources to return.
            index_directory: Directory holding a saved BM25 index.

        Returns:
            The retrieved source ranges, most relevant first.
        """
        query = _as_text(question, "question")
        top_k = _as_positive_int(k, "k")
        retriever = _load_retriever(index_directory)

        with _reported_as("Searching the index"):
            sources = retriever.search(query, top_k)

        return [source.model_dump() for source in sources]

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_directory: str = "data/processed/bm25_index",
    ) -> str:
        """Search every question in a JSON dataset using a saved index.

        Args:
            dataset_path: JSON dataset holding the questions to search.
            k: Maximum number of sources to return per question.
            save_directory: Directory where the results are written.
            index_directory: Directory holding a saved BM25 index.

        Returns:
            A message naming the file that was written.
        """
        dataset = _existing_file(dataset_path, "dataset_path")
        top_k = _as_positive_int(k, "k")
        output_directory = Path(
            _as_text(save_directory, "save_directory")
        ).expanduser()
        retriever = _load_retriever(index_directory)

        with _reported_as(f"Searching the dataset {dataset}"):
            output_path = create_search_dataset(
                dataset_path=dataset,
                retriever=retriever,
                k=top_k,
                save_directory=output_directory,
            )

        return f"Saved student_search_results to {output_path}"

    def evaluate(
        self,
        student_results_path: str,
        dataset_path: str,
        k: int = 10,
        minimal_iou_threshold: float = 0.05,
    ) -> dict[str, float | int]:
        """Evaluate saved search results against ground truth.

        Args:
            student_results_path: Saved search results to evaluate.
            dataset_path: Answered dataset used as ground truth.
            k: Maximum number of sources considered per question.
            minimal_iou_threshold: Minimum overlap counting as found.

        Returns:
            Recall@1, Recall@3, Recall@5, and Recall@10.
        """
        results_file = _existing_file(
            student_results_path,
            "student_results_path",
        )
        dataset_file = _existing_file(dataset_path, "dataset_path")
        top_k = _as_positive_int(k, "k")
        threshold = _as_ratio(minimal_iou_threshold, "minimal_iou_threshold")

        with _reported_as("Evaluating the search results"):
            return evaluate_search_results(
                student_results_path=results_file,
                dataset_path=dataset_file,
                k=top_k,
                minimal_iou_threshold=threshold,
            )

    def answer(
        self,
        question: str,
        k: int = 5,
        index_directory: str = "data/processed/bm25_index",
        repository_path: str | None = None,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> dict[str, object]:
        """Retrieve sources and generate a grounded answer.

        Args:
            question: Natural language question to answer.
            k: Maximum number of sources passed to the model.
            index_directory: Directory holding a saved BM25 index.
            repository_path: vLLM repository; searched when omitted.
            model_name: Hugging Face identifier of the answering model.

        Returns:
            The question, its retrieved sources, and the answer.
        """
        from src.answering import AnswerGenerator

        query = _as_text(question, "question")
        top_k = _as_positive_int(k, "k")
        repository = _resolve_repository(repository_path)
        model = _as_text(model_name, "model_name")
        retriever = _load_retriever(index_directory)

        with _reported_as("Searching the index"):
            sources = retriever.search(query, top_k)

        with _reported_as(f"Generating an answer with {model}"):
            generator = AnswerGenerator(model_name=model)
            generated_answer = generator.answer(query, sources, repository)

        return {
            "question": query,
            "retrieved_sources": [
                source.model_dump()
                for source in sources
            ],
            "answer": generated_answer,
        }

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        repository_path: str | None = None,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> str:
        """Generate answers for every question in saved search results.

        Args:
            student_search_results_path: Saved search results to answer.
            save_directory: Directory where the answers are written.
            repository_path: vLLM repository; searched when omitted.
            model_name: Hugging Face identifier of the answering model.

        Returns:
            A message naming the file that was written.
        """
        from src.answering import AnswerGenerator
        from src.answering import answer_dataset as create_answer_dataset

        results_file = _existing_file(
            student_search_results_path,
            "student_search_results_path",
        )
        output_directory = Path(
            _as_text(save_directory, "save_directory")
        ).expanduser()
        repository = _resolve_repository(repository_path)
        model = _as_text(model_name, "model_name")

        with _reported_as(f"Generating answers with {model}"):
            output_path = create_answer_dataset(
                student_search_results_path=results_file,
                generator=AnswerGenerator(model_name=model),
                repository_path=repository,
                save_directory=output_directory,
            )

        return f"Saved student_search_results_and_answer to {output_path}"
