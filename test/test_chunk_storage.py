"""Tests for chunk persistence."""

import json
from pathlib import Path

from src.chunking import save_chunks
from src.structdata import DocumentChunk


def test_save_chunks_creates_json_with_all_chunk_fields(
    tmp_path: Path,
) -> None:
    """Saved chunks preserve the data needed to rebuild an index."""
    chunks = [
        DocumentChunk(
            chunk_id="guide:0:19",
            file_path="data/raw/vllm-0.10.1/docs/guide.md",
            first_character_index=0,
            last_character_index=19,
            text="A short vLLM guide.",
        )
    ]
    output_path = tmp_path / "processed" / "chunks" / "chunks.json"

    save_chunks(chunks, output_path)

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        chunks[0].model_dump()
    ]
