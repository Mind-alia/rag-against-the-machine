"""Generate grounded answers with Qwen."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from src.errors import RagError
from src.structdata import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
LOGICAL_REPOSITORY = PurePosixPath("data/raw/vllm-0.10.1")

MAX_CONTEXT_CHARACTERS = 8000
MAX_PASSAGE_CHARACTERS = 1200
MAX_NEW_TOKENS = 512
THINKING_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

NO_ANSWER = "I do not know: the retrieved context does not answer this."
UNREADABLE_SOURCES = (
    "I do not know: none of the retrieved sources could be read from "
    "the repository."
)

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the vLLM "
    "repository using only the context you are given."
)
INSTRUCTIONS = (
    "Answer the question using only the context below.\n"
    "Write a self-contained answer that can be read without seeing the "
    "question.\n"
    "Use only the context: never invent information. If the context "
    "does not hold the answer, say that you do not know."
)


class AnswerGenerator(BaseModel):
    """Generate answers from retrieved source passages."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    model_name: str = Field(default=DEFAULT_MODEL, min_length=1)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, gt=0)
    max_context_characters: int = Field(
        default=MAX_CONTEXT_CHARACTERS,
        gt=0,
    )
    max_passage_characters: int = Field(
        default=MAX_PASSAGE_CHARACTERS,
        gt=0,
    )
    tokenizer: Any = None
    model: Any = None

    def answer(
        self,
        question: str,
        sources: list[MinimalSource],
        repository_path: Path,
    ) -> str:
        """Generate one source-grounded answer.

        Args:
            question: Question to answer.
            sources: Retrieved source ranges used as context.
            repository_path: Directory holding the vLLM repository.

        Returns:
            The generated answer, without the model reasoning trace.
        """
        self._load_model()

        passages = _select_passages(
            sources,
            repository_path,
            self.max_context_characters,
            self.max_passage_characters,
        )

        if not passages:
            return UNREADABLE_SOURCES

        context = _format_context(passages)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{INSTRUCTIONS}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ]

        answer = self._generate(messages) or NO_ANSWER

        return f"{answer}\n\n{_format_citations(passages)}"

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run the model on a prepared chat prompt.

        Args:
            messages: Chat messages sent to the model.

        Returns:
            The decoded answer, stripped of any reasoning block.

        Raises:
            RagError: If the model fails to produce an answer.
        """
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        except Exception as error:
            raise RagError(
                f"the model {self.model_name} failed to answer: {error}"
            ) from error

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        decoded_answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return _strip_reasoning(cast(str, decoded_answer))

    def _load_model(self) -> None:
        """Load Qwen only once for the current command.

        Raises:
            RagError: If the model cannot be downloaded or loaded.
        """
        if self.model is not None:
            return

        torch, auto_model, auto_tokenizer = _load_llm_dependencies()
        device, dtype = _get_device_and_dtype(torch)

        try:
            self.tokenizer = auto_tokenizer.from_pretrained(self.model_name)
            self.model = auto_model.from_pretrained(
                self.model_name,
                dtype=dtype,
            )
            self.model.to(device)
            self.model.eval()
        except Exception as error:
            raise RagError(
                f"could not load the model {self.model_name}: {error}"
            ) from error


def answer_dataset(
    student_search_results_path: Path,
    generator: AnswerGenerator,
    repository_path: Path,
    save_directory: Path,
    show_progress: bool = True,
) -> Path:
    """Generate answers for every result in a search-results JSON file.

    Args:
        student_search_results_path: Saved search results to answer.
        generator: Configured answer generator.
        repository_path: Directory holding the vLLM repository.
        save_directory: Directory where the answers are written.
        show_progress: Whether to display a progress bar.

    Returns:
        The path of the written answer file.
    """
    search_results = StudentSearchResults.model_validate_json(
        student_search_results_path.read_text(encoding="utf-8")
    )

    answers = [
        MinimalAnswer(
            question_id=result.question_id,
            question_str=result.question_str,
            retrieved_sources=result.retrieved_sources,
            answer=generator.answer(
                result.question_str,
                result.retrieved_sources,
                repository_path,
            ),
        )
        for result in tqdm(
            search_results.search_results,
            desc="Generating answers",
            disable=not show_progress,
        )
    ]

    output = StudentSearchResultsAndAnswer(
        search_results=answers,
        k=search_results.k,
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / student_search_results_path.name
    output_path.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def _strip_reasoning(answer: str) -> str:
    """Remove the reasoning trace some chat models emit.

    Args:
        answer: Raw text decoded from the model.

    Returns:
        The user-facing answer only.
    """
    without_block = THINKING_BLOCK.sub("", answer)

    if "</think>" in without_block:
        without_block = without_block.rsplit("</think>", 1)[-1]

    return without_block.replace("<think>", "").strip()


def _load_llm_dependencies() -> tuple[Any, Any, Any]:
    """Load Qwen dependencies only when answer generation is requested.

    Returns:
        The torch module, the model class, and the tokenizer class.

    Raises:
        RagError: If the optional LLM dependencies are missing.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RagError(
            "answer generation requires torch and transformers. "
            "Run `make install` (or `uv sync`) first."
        ) from error

    return torch, AutoModelForCausalLM, AutoTokenizer


def _get_device_and_dtype(torch: Any) -> tuple[str, Any]:
    """Choose the best available local PyTorch device.

    Args:
        torch: The imported torch module.

    Returns:
        The device name and the matching floating point type.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16

    if torch.backends.mps.is_available():
        return "mps", torch.float16

    return "cpu", torch.float32


def _select_passages(
    sources: list[MinimalSource],
    repository_path: Path,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
    max_passage_characters: int = MAX_PASSAGE_CHARACTERS,
) -> list[tuple[MinimalSource, str]]:
    """Read the retrieved passages that fit in the prompt budget.

    Two limits keep the prompt usable by a small model: the whole
    context stays under ``max_characters``, and no single passage may
    take more than ``max_passage_characters`` of it. Without the second
    limit the first two chunks fill the budget and a lower-ranked but
    relevant passage never reaches the model.

    Args:
        sources: Retrieved source ranges.
        repository_path: Directory holding the vLLM repository.
        max_characters: Maximum size of the assembled context.
        max_passage_characters: Maximum size of one passage.

    Returns:
        Each readable source with the passage text kept for the prompt.
    """
    passages: list[tuple[MinimalSource, str]] = []
    remaining = max_characters

    for source in sources:
        text = _read_source_text(source, repository_path)

        if not text:
            continue

        header_length = len(_citation(source)) + 1
        budget = min(
            max_passage_characters,
            remaining - header_length,
        )

        if budget <= 0:
            break

        passages.append((source, text[:budget]))
        remaining -= header_length + min(len(text), budget)

    return passages


def _citation(source: MinimalSource) -> str:
    """Return the traceable tag of one source range.

    Args:
        source: Source range to describe.

    Returns:
        The tag in the form ``[file_path:start-end]``.
    """
    return (
        f"[{source.file_path}:"
        f"{source.first_character_index}-"
        f"{source.last_character_index}]"
    )


def _format_context(passages: list[tuple[MinimalSource, str]]) -> str:
    """Assemble the prompt context from the selected passages.

    Args:
        passages: Sources and their text, as selected for the prompt.

    Returns:
        The formatted context sent to the model.
    """
    return "\n\n".join(
        f"{_citation(source)}\n{text}" for source, text in passages
    )


def _format_citations(passages: list[tuple[MinimalSource, str]]) -> str:
    """List the sources the answer was grounded on.

    The citations come from the retrieved ranges rather than from the
    model, so an answer can never cite a source that was not supplied.

    Args:
        passages: Sources and their text, as selected for the prompt.

    Returns:
        A single ``Sources:`` line naming every supplied range.
    """
    return "Sources: " + ", ".join(
        _citation(source) for source, _ in passages
    )


def _build_context(
    sources: list[MinimalSource],
    repository_path: Path,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
    max_passage_characters: int = MAX_PASSAGE_CHARACTERS,
) -> str:
    """Read and format retrieved passages for the model prompt.

    Args:
        sources: Retrieved source ranges.
        repository_path: Directory holding the vLLM repository.
        max_characters: Maximum size of the assembled context.
        max_passage_characters: Maximum size of one passage.

    Returns:
        The formatted context, empty when no source is readable.
    """
    return _format_context(
        _select_passages(
            sources,
            repository_path,
            max_characters,
            max_passage_characters,
        )
    )


def _read_source_text(
    source: MinimalSource,
    repository_path: Path,
) -> str | None:
    """Read the exact passage a source points to.

    Args:
        source: Retrieved source range.
        repository_path: Directory holding the vLLM repository.

    Returns:
        The passage text, or None when the file cannot be read.
    """
    try:
        relative_path = PurePosixPath(source.file_path).relative_to(
            LOGICAL_REPOSITORY
        )
    except ValueError:
        relative_path = PurePosixPath(source.file_path)

    source_path = repository_path / Path(*relative_path.parts)

    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    return text[
        source.first_character_index:source.last_character_index
    ]
