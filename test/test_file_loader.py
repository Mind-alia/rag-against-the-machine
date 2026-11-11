from pathlib import Path

import pytest

from src.file_loader import load_documents


def test_load_documents_reads_markdown_and_python(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "vllm-0.10.1"

    docs_directory = repository / "docs"
    code_directory = repository / "vllm"

    docs_directory.mkdir(parents=True)
    code_directory.mkdir(parents=True)

    (docs_directory / "guide.md").write_text(
        "# vLLM guide",
        encoding="utf-8",
    )
    (code_directory / "engine.py").write_text(
        "class Engine:\n    pass\n",
        encoding="utf-8",
    )

    documents = load_documents(repository)

    loaded_files = {
        document.file_path: document.text
        for document in documents
    }

    assert loaded_files == {
        "data/raw/vllm-0.10.1/docs/guide.md": "# vLLM guide",
        "data/raw/vllm-0.10.1/vllm/engine.py": (
            "class Engine:\n    pass\n"
        ),
    }


def test_load_documents_includes_supported_root_files(
    tmp_path: Path,
) -> None:
    """Root files used by vLLM documentation stay searchable."""
    repository = tmp_path / "vllm-0.10.1"
    repository.mkdir()

    for filename in (
        "README.md",
        "RELEASE.md",
        "SECURITY.md",
        "CMakeLists.txt",
    ):
        (repository / filename).write_text(filename, encoding="utf-8")

    documents = load_documents(repository)

    assert [document.file_path for document in documents] == [
        "data/raw/vllm-0.10.1/CMakeLists.txt",
        "data/raw/vllm-0.10.1/README.md",
        "data/raw/vllm-0.10.1/RELEASE.md",
        "data/raw/vllm-0.10.1/SECURITY.md",
    ]


def test_load_documents_ignores_empty_and_unsupported_files(
    tmp_path: Path,
) -> None:
    """Only non-empty Markdown documentation and Python files are loaded."""
    repository = tmp_path / "vllm-0.10.1"
    (repository / "docs").mkdir(parents=True)
    (repository / "vllm").mkdir()
    (repository / "examples").mkdir()

    (repository / "docs" / "empty.md").write_text("", encoding="utf-8")
    (repository / "docs" / "guide.txt").write_text("ignore", encoding="utf-8")
    (repository / "vllm" / "engine.py").write_text("pass", encoding="utf-8")
    (repository / "examples" / "demo.py").write_text("pass", encoding="utf-8")

    documents = load_documents(repository)

    assert [document.file_path for document in documents] == [
        "data/raw/vllm-0.10.1/vllm/engine.py"
    ]


def test_load_documents_rejects_a_missing_repository(tmp_path: Path) -> None:
    """A typo in the repository path produces a clear caller-facing error."""
    missing_repository = tmp_path / "missing-vllm"

    with pytest.raises(ValueError, match="repository not found"):
        load_documents(missing_repository)
