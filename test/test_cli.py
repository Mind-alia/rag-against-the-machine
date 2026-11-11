"""Tests for command-line behaviour on valid and degenerate input."""

import json
from pathlib import Path
from typing import cast

import pytest

from src.cli import RagCli
from src.errors import RagError

GUIDE = "data/raw/vllm-0.10.1/docs/guide.md"


@pytest.fixture
def cli() -> RagCli:
    """Provide the command namespace exposed to Python Fire."""
    return RagCli()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Provide a minimal repository with one document and one module."""
    documentation = tmp_path / "repository" / "docs"
    documentation.mkdir(parents=True)
    (documentation / "guide.md").write_text(
        "# Batching\n\nvLLM uses continuous batching for requests.\n",
        encoding="utf-8",
    )

    package = tmp_path / "repository" / "vllm"
    package.mkdir(parents=True)
    (package / "engine.py").write_text(
        "def schedule() -> None:\n    \"\"\"Schedule GPU execution.\"\"\"\n",
        encoding="utf-8",
    )

    return tmp_path / "repository"


def test_cli_imports_without_torch() -> None:
    """Indexing and search commands must stay available without Qwen."""
    commands = RagCli()

    assert callable(commands.index)
    assert callable(commands.search)
    assert callable(commands.evaluate)


def test_index_then_search_returns_traceable_sources(
    cli: RagCli,
    repository: Path,
    tmp_path: Path,
) -> None:
    """A freshly built index answers a query with exact ranges."""
    index_directory = tmp_path / "index"

    summary = cli.index(
        repository_path=str(repository),
        index_directory=str(index_directory),
        chunks_path=str(tmp_path / "chunks.json"),
    )

    assert "Indexed 2 documents" in summary

    sources = cli.search(
        "continuous batching",
        k=1,
        index_directory=str(index_directory),
    )

    assert sources[0]["file_path"] == GUIDE
    assert int(str(sources[0]["last_character_index"])) > 0


@pytest.mark.parametrize("k", [0, -3, "abc", 2.5, None])
def test_search_rejects_an_invalid_result_count(
    cli: RagCli,
    k: object,
) -> None:
    """A degenerate --k value produces an error, never a traceback."""
    with pytest.raises(RagError, match="k must be a positive integer"):
        cli.search("batching", k=cast(int, k))


@pytest.mark.parametrize("question", ["", "   "])
def test_search_rejects_an_empty_question(
    cli: RagCli,
    question: str,
) -> None:
    """An empty query is refused instead of returning noise."""
    with pytest.raises(RagError, match="question must be a non-empty"):
        cli.search(question)


def test_search_explains_how_to_build_a_missing_index(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """Searching without an index points the user to the index command."""
    with pytest.raises(RagError, match="Run `uv run python -m src index`"):
        cli.search("batching", index_directory=str(tmp_path / "missing"))


def test_index_rejects_a_missing_repository(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """A typo in the repository path produces a clear message."""
    with pytest.raises(RagError, match="repository not found"):
        cli.index(repository_path=str(tmp_path / "missing"))


def test_index_finds_the_repository_under_data_raw(
    cli: RagCli,
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subject layout data/raw/vllm-0.10.1 is found without a flag."""
    workspace = tmp_path / "workspace"
    (workspace / "data" / "raw").mkdir(parents=True)
    (workspace / "data" / "raw" / "vllm-0.10.1").symlink_to(repository)
    monkeypatch.chdir(workspace)

    summary = cli.index(
        index_directory=str(tmp_path / "index"),
        chunks_path=str(tmp_path / "chunks.json"),
    )

    assert "Indexed 2 documents" in summary


def test_index_reports_where_it_looked_for_the_repository(
    cli: RagCli,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing repository names the layouts that were searched."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RagError, match="data/raw/vllm-0.10.1"):
        cli.index()


def test_index_rejects_an_empty_repository(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """A repository without supported files is refused clearly."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RagError, match="No supported file"):
        cli.index(
            repository_path=str(empty),
            index_directory=str(tmp_path / "index"),
            chunks_path=str(tmp_path / "chunks.json"),
        )


@pytest.mark.parametrize(
    ("max_chunk_size", "overlap", "message"),
    [
        (0, 100, "max_chunk_size must be a positive integer"),
        (-5, 100, "max_chunk_size must be a positive integer"),
        (100, -1, "overlap cannot be negative"),
        (100, 500, "overlap must be smaller than max_chunk_size"),
    ],
)
def test_index_rejects_invalid_chunk_settings(
    cli: RagCli,
    repository: Path,
    max_chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    """Degenerate chunk settings are refused before any file is read."""
    with pytest.raises(RagError, match=message):
        cli.index(
            repository_path=str(repository),
            max_chunk_size=max_chunk_size,
            overlap=overlap,
        )


def test_evaluate_rejects_a_missing_file(cli: RagCli) -> None:
    """A missing results file produces a clear message."""
    with pytest.raises(RagError, match="file not found"):
        cli.evaluate(
            student_results_path="does-not-exist.json",
            dataset_path="does-not-exist.json",
        )


def test_evaluate_rejects_malformed_json(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """A truncated JSON file produces a clear message."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(RagError, match="Invalid JSON"):
        cli.evaluate(
            student_results_path=str(broken),
            dataset_path=str(broken),
        )


def test_evaluate_rejects_json_missing_required_fields(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """A JSON file with the wrong shape names the missing field."""
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"unexpected": 1}), encoding="utf-8")

    with pytest.raises(RagError, match="search_results"):
        cli.evaluate(
            student_results_path=str(wrong_shape),
            dataset_path=str(wrong_shape),
        )


def test_evaluate_rejects_an_out_of_range_threshold(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """An overlap threshold outside zero and one is refused."""
    existing = tmp_path / "results.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(RagError, match="must be between 0 and 1"):
        cli.evaluate(
            student_results_path=str(existing),
            dataset_path=str(existing),
            minimal_iou_threshold=42,
        )


def test_search_dataset_rejects_a_directory_as_dataset(
    cli: RagCli,
    tmp_path: Path,
) -> None:
    """Passing a directory where a file is expected is refused."""
    with pytest.raises(RagError, match="is a directory, not a file"):
        cli.search_dataset(dataset_path=str(tmp_path))
