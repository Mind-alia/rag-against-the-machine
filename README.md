*This project has been created as part of the 42 curriculum by allasser*

# RAG against the machine

[![CI](https://github.com/Mind-alia/rag-against-the-machine/actions/workflows/ci.yml/badge.svg)](https://github.com/Mind-alia/rag-against-the-machine/actions/workflows/ci.yml)

## Description

This project builds a Retrieval-Augmented Generation (RAG) system that answers
questions about the vLLM codebase. It ingests the vLLM documentation and Python
sources, splits them into chunks that keep their exact character ranges, ranks
those chunks with BM25, and feeds the retrieved passages to `Qwen/Qwen3-0.6B` to
produce grounded answers.

Every retrieved result is a triple `(file_path, first_character_index,
last_character_index)`, so any answer can be traced back to the exact bytes of
the repository it came from.

## Instructions

Install the dependencies (Python 3.10 and `uv` are required):

```bash
make install   # equivalent to: uv sync
```

`uv sync` alone installs everything the project needs, including the
`transformers` and `torch` stack used by the answer commands.

Every rule checks that `uv` is on `PATH` first and explains how to install it
otherwise. If the sources were copied from another machine, run
`make reinstall`: a `.venv` built on a different operating system holds
binaries that cannot run here, and rebuilding it is faster than diagnosing
it.

Run the automated checks:

```bash
make test        # pytest
make lint        # flake8 + mypy with the mandatory flags
make lint-strict # flake8 + mypy --strict
```

`.python-version` pins Python 3.10, so `uv` picks (or downloads) the right
interpreter by itself.

Unzip the vLLM repository (the `vllm-0.10.1` attachment) into either layout:

```
data/raw/vllm-0.10.1/     # layout used by the subject
vllm-0.10.1/              # or directly at the project root
```

Both are found automatically, so `--repository_path` is only needed for a
repository kept somewhere else. Build the index once before searching.

## Example usage

### 1. Index the repository

```bash
uv run python -m src index --max_chunk_size 2000
# Indexed 974 documents into 9347 chunks.
# Index saved under data/processed/bm25_index.
```

Pass `--repository_path` to index a repository kept outside the two default
layouts.

### 2. Search a single question

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

### 3. Search a whole dataset

```bash
uv run python -m src search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
# Saved student_search_results to data/output/search_results/dataset_docs_public.json
```

### 4. Evaluate the retrieval quality

```bash
uv run python -m src evaluate \
  --student_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10
# questions_evaluated: 100
# recall_at_1: 0.6   recall_at_3: 0.81   recall_at_5: 0.88   recall_at_10: 0.9
```

### 5. Answer a single question

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

### 6. Answer a whole dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

The first run of `answer` or `answer_dataset` downloads `Qwen/Qwen3-0.6B`.
Every command validates its arguments and reports failures as a single
`Error: ...` line, never as a traceback:

```bash
uv run python -m src search "batching" --k -1
# Error: k must be a positive integer, got -1
```

## System architecture

The pipeline is a chain of five modules, each one owning a single stage:

| Stage | Module | Role |
|---|---|---|
| Ingestion | [`src/file_loader.py`](src/file_loader.py) | Reads the useful files of the repository |
| Chunking | [`src/chunking.py`](src/chunking.py) | Splits documents while keeping character ranges |
| Indexing | [`src/indexing.py`](src/indexing.py) | Builds and persists the BM25 index |
| Retrieval | [`src/retrieval.py`](src/retrieval.py) | Ranks chunks and returns minimal sources |
| Generation | [`src/answering.py`](src/answering.py) | Builds the context and prompts Qwen |

The generation stage keeps the prompt inside the limits of a 0.6B model with
two budgets: the whole context stays under 8 000 characters, and no single
passage may take more than 1 200 of them. The per-passage cap matters as much
as the total: without it the two top-ranked chunks fill the budget and a
lower-ranked but relevant passage never reaches the model.

[`src/evaluation.py`](src/evaluation.py) scores saved search results against a
ground-truth dataset, [`src/structdata.py`](src/structdata.py) holds every
pydantic model, and [`src/cli.py`](src/cli.py) exposes the six commands through
Python Fire.

The ingestion stage keeps only what can answer a question: Markdown under
`docs/`, Python under `vllm/`, and the useful root files (`README.md`,
`RELEASE.md`, `SECURITY.md`, `CMakeLists.txt`). Each document is stored under
its logical path `data/raw/vllm-0.10.1/...`, which is the path the evaluation
datasets use.

## Chunking strategy

Chunks are built by walking a document from its first character to its last, so
consecutive chunks tile the whole file and every chunk knows its exact
`[first_character_index, last_character_index)` range.

The chunker prefers semantic boundaries, which differ per file type:

- **Python**: top-level `class`, `def`, and `async def` offsets, found by
  parsing the file with `ast`, so a chunk rarely cuts a definition in half. A
  file that does not parse falls back to fixed-size chunking.
- **Markdown**: heading offsets (`#` to `######`), so a chunk tends to hold one
  section.

Within a chunk the furthest boundary that still fits the size limit is chosen;
when no boundary fits, the chunk is cut at the size limit. Consecutive chunks
share an overlap (200 characters by default) so an answer that straddles a
boundary still appears whole in at least one chunk. `--max_chunk_size` is a CLI
argument and defaults to the 2000-character maximum required by the subject.

## Retrieval method

Retrieval uses **BM25** (`bm25s`, Lucene variant) over the 9347 chunks.

Documents and queries go through the exact same `tokenize` function, so a query
term and the chunk term it should match always reduce to the same token:

1. lowercasing,
2. English stopword removal,
3. Snowball stemming (`PyStemmer`), so *configuring*, *configured*, and
   *configuration* collapse to one token.

Ranking is the standard BM25 score with the library defaults (`k1 = 1.5`,
`b = 0.75`). BM25 scores chunk ids; the retriever maps the top-k ids back to
their chunks and returns only the minimal source ranges, which is exactly what
the evaluator compares.

The index is persisted under `data/processed/bm25_index/` together with the
chunks it references, so `search` never rebuilds it.

## Performance analysis

Measured on an Apple M-series laptop over the public datasets, against the
minimum performances required by the subject:

| Metric | Requirement | Measured |
|---|---|---|
| Indexing time | ≤ 5 min | **2.9 s** (974 documents, 9347 chunks) |
| Cold start latency | ≤ 60 s | **0.28 s** |
| Retrieval throughput | ≤ 90 s / 1000 questions | **0.3 s / 100 questions** |

Recall@k, using the same metric as the evaluator (a ground-truth source counts
as found when its character range overlaps a retrieved range of the same file
by more than 5%):

| Dataset | R@1 | R@3 | **R@5** | R@10 | Threshold |
|---|---|---|---|---|---|
| docs | 0.60 | 0.81 | **0.88** | 0.90 | ≥ 0.80 |
| code | 0.34 | 0.51 | **0.56** | 0.63 | ≥ 0.50 |

Both datasets pass. Documentation retrieval is comfortably ahead because a
question about the docs usually reuses the vocabulary of the page that answers
it. Code retrieval is harder: questions are asked in natural language
("what method returns the maximum number of input items?") while the chunk
holds identifiers (`get_supported_mm_limits`), so the lexical overlap BM25
relies on is much thinner. Adding stemming and stopword removal moved code
Recall@5 from 0.53 to 0.56 and docs Recall@5 from 0.85 to 0.88; closing the
remaining gap would need semantic embeddings rather than better lexical tuning.

## Design decisions

- **Character ranges over file paths.** The evaluator scores the overlap of
  character ranges, not files, so a range is carried end to end: chunking
  records it, the index references it, and the answer prompt quotes exactly the
  passage it points to.
- **Pydantic everywhere.** Every structure crossing a boundary is a validated
  model, including the retriever and the answer generator. The output models
  mirror the evaluator field for field (`question_str` included), so a malformed
  submission fails locally instead of at correction time.
- **One tokenizer.** Indexing and querying call the same function, which makes
  a stopword or stemming change impossible to apply to only one side.
- **Persisted index.** Indexing runs once; `search` reloads the saved index and
  chunks, which is what keeps the cold start under a second.
- **Lazy LLM import.** `torch` and `transformers` are imported only inside the
  `answer` commands, so indexing, searching, and evaluating start instantly and
  keep working even if the LLM stack is unavailable.
- **Two context budgets instead of one.** Capping only the total context lets
  the first chunks crowd out the rest; capping each passage as well raised the
  share of substantive answers from 3/7 to 6/7 on a sample of questions.
- **Deterministic citations.** The `Sources:` line appended to an answer is
  built from the retrieved ranges, not asked from the model. A 0.6B model asked
  to cite its sources invents plausible file names; building the line in code
  makes a fabricated citation impossible.
- **Errors as messages.** Commands validate their arguments up front and raise
  `RagError`; the entry point prints one `Error: ...` line and exits non-zero,
  so no degenerate input can produce a traceback.

## Challenges faced

- **Retrieving ranges, not files.** Returning the right file is easy; returning
  a range that overlaps the annotated one is not. Solved by making the chunks
  tile each document with an overlap, so any annotated passage is fully inside
  at least one chunk.
- **Natural language against identifiers.** Code questions share few literal
  tokens with the code that answers them. Stemming and stopword removal
  recovered part of the gap; the rest is a known limit of lexical retrieval.
- **Qwen3 reasoning traces.** The model emitted a `<think>...</think>` block
  that consumed the whole generation budget and truncated the answer mid
  sentence. Disabling thinking in the chat template, raising the token budget,
  and stripping any residual block fixed both the leak and the truncation.
- **Crash-free CLI.** The first version propagated `FileNotFoundError`,
  `ValueError`, and pydantic `ValidationError` straight to the terminal. Every
  command now validates its input and every failure is reported as one line.
- **Context size, not just retrieval quality.** Passing all ten retrieved
  chunks to Qwen3-0.6B made it answer "the answer is not available" on
  questions whose sources had been correctly retrieved. The fix was on the
  prompt side, not the retrieval side: fewer characters per passage so more
  distinct passages fit.

## Resources

- [vLLM documentation](https://docs.vllm.ai/)
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [`bm25s` documentation](https://github.com/xhluca/bm25s)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire documentation](https://google.github.io/python-fire/)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)

### Use of AI

AI was used to review the project structure, propose an incremental
implementation plan, explain the trade-offs between chunking and retrieval
options, and audit the project against the subject (error handling, evaluation
metric parity with the moulinette, and test coverage). Every generated change
was read, tested, and measured locally before being kept; the recall and
latency figures above come from runs on this machine, not from the model.

## License

[MIT](LICENSE) for the code. The question datasets under `datasets_public/` were
provided with the 42 subject and are not covered by this license.
