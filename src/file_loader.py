"""Load useful text files from the vLLM repository."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field
from tqdm import tqdm

from src.structdata import LoadedDocument

LOGICAL_REPOSITORY_PATH = PurePosixPath("data/raw/vllm-0.10.1")

REPOSITORY_CANDIDATES = (
    "data/raw/vllm-0.10.1",
    "vllm-0.10.1",
    "data/raw/vllm",
    "vllm",
    "../data/raw/vllm-0.10.1",
    "../vllm-0.10.1",
)

ROOT_FILES = {
    "README.md",
    "RELEASE.md",
    "SECURITY.md",
    "CMakeLists.txt",
}


class RepositoryLoader(BaseModel):
    """Load searchable files from the vLLM repository."""

    repository_path: Path
    logical_repository_path: str = Field(
        default=str(LOGICAL_REPOSITORY_PATH),
        min_length=1,
    )
    show_progress: bool = True

    def is_supported_file(self, file_path: Path) -> bool:
        """Return whether a file should be loaded.

        Args:
            file_path: Candidate file inside the repository.

        Returns:
            Whether the file belongs to the searchable knowledge base.
        """
        relative_path = file_path.relative_to(self.repository_path)

        if len(relative_path.parts) == 1:
            return relative_path.name in ROOT_FILES

        first_directory = relative_path.parts[0]

        if first_directory == "docs":
            return file_path.suffix == ".md"

        if first_directory == "vllm":
            return file_path.suffix == ".py"

        return False

    def load_documents(self) -> list[LoadedDocument]:
        """Load every supported, non-empty file.

        Returns:
            Searchable documents with evaluator-compatible logical paths.

        Raises:
            ValueError: If the repository directory does not exist.
        """
        if not self.repository_path.is_dir():
            raise ValueError(
                f"repository not found: {self.repository_path}"
            )

        candidates = sorted(self.repository_path.rglob("*"))
        documents: list[LoadedDocument] = []

        for file_path in tqdm(
            candidates,
            desc="Loading repository files",
            disable=not self.show_progress,
        ):
            document = self._load_document(file_path)

            if document is not None:
                documents.append(document)

        return documents

    def _load_document(self, file_path: Path) -> LoadedDocument | None:
        """Read one repository file, skipping anything unusable."""
        if not file_path.is_file() or not self.is_supported_file(file_path):
            return None

        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        if not text:
            return None

        relative_path = file_path.relative_to(self.repository_path)

        return LoadedDocument(
            file_path=(
                f"{self.logical_repository_path}/"
                f"{relative_path.as_posix()}"
            ),
            text=text,
        )


def looks_like_repository(path: Path) -> bool:
    """Return whether a directory holds the vLLM sources to index.

    Args:
        path: Candidate directory.

    Returns:
        Whether the directory has the folders the loader reads.
    """
    return path.is_dir() and (
        (path / "vllm").is_dir() or (path / "docs").is_dir()
    )


def find_repository(repository_path: str | None = None) -> Path:
    """Locate the vLLM repository to index.

    The subject places the repository under ``data/raw/vllm-0.10.1``,
    but it is often unzipped at the project root instead, so both
    layouts are searched when no explicit path is given.

    Args:
        repository_path: Explicit location, or None to search.

    Returns:
        The directory holding the vLLM sources.

    Raises:
        ValueError: If no candidate directory holds a vLLM repository.
    """
    if repository_path is not None:
        explicit = Path(repository_path).expanduser()

        if not explicit.is_dir():
            raise ValueError(f"repository not found: {explicit}")

        return explicit

    candidates = [Path(name) for name in REPOSITORY_CANDIDATES]
    candidates.extend(sorted(Path("data/raw").glob("vllm-*")))
    candidates.extend(sorted(Path("../data/raw").glob("vllm-*")))

    for candidate in candidates:
        if looks_like_repository(candidate):
            return candidate

    raise ValueError(
        "no vLLM repository found in "
        + ", ".join(REPOSITORY_CANDIDATES)
        + "; unzip it there or pass --repository_path"
    )


def load_documents(
    repository_path: Path,
    show_progress: bool = True,
) -> list[LoadedDocument]:
    """Load the supported files from a vLLM repository.

    Args:
        repository_path: Filesystem location of the vLLM repository.
        show_progress: Whether to display a progress bar.

    Returns:
        Searchable documents with evaluator-compatible logical paths.
    """
    loader = RepositoryLoader(
        repository_path=repository_path,
        logical_repository_path=str(LOGICAL_REPOSITORY_PATH),
        show_progress=show_progress,
    )
    return loader.load_documents()
