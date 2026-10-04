"""Unit tests for ingestion idempotency, manifest handling, and model mismatch rejection."""

import hashlib
import json
from pathlib import Path
from typing import List
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.ingest import (
    ingest_strategy,
    load_ingest_manifest,
    run_ingest,
    save_ingest_manifest,
)


class MockHashEmbeddings(Embeddings):
    """Deterministic hash-based embedding for testing without network/API."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Take bytes and normalize to float in [0.0, 1.0]
        return [float(b) / 255.0 for b in digest[: self.dim]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def test_ingest_idempotency(tmp_path, monkeypatch):
    """Running ingest twice should not duplicate chunks."""
    # Setup mock data directory
    mock_data_dir = tmp_path / "data"
    mock_chroma_dir = tmp_path / "chroma_db"
    mock_data_dir.mkdir(parents=True)
    mock_chroma_dir.mkdir(parents=True)

    monkeypatch.setattr("src.ingest.DATA_DIR", mock_data_dir)
    monkeypatch.setattr("src.ingest.CHROMA_DIR", mock_chroma_dir)

    dummy_chunks = [
        {
            "chunk_id": f"test-A-{i:03d}",
            "doc_id": "test",
            "title": "Tài liệu kiểm tra",
            "source_url": "https://momo.vn/test",
            "section": "",
            "strategy": "A",
            "text": f"Nội dung đoạn văn số {i}",
        }
        for i in range(1, 6)
    ]

    chunks_file = mock_data_dir / "chunks_A.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in dummy_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    embeddings = MockHashEmbeddings()

    # First run
    count1 = ingest_strategy("A", embedding_function=embeddings, batch_size=2)
    assert count1 == 5

    # Verify count in vector store
    store = Chroma(
        collection_name="momo_A",
        persist_directory=str(mock_chroma_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    assert len(store.get()["ids"]) == 5

    # Second run (should be completely idempotent)
    count2 = ingest_strategy("A", embedding_function=embeddings, batch_size=2)
    assert count2 == 5
    assert len(store.get()["ids"]) == 5


def test_ingest_manifest_creation_and_rejection(tmp_path, monkeypatch):
    """Test manifest creation and rejection when model differs without --rebuild."""
    mock_chroma_dir = tmp_path / "chroma_db"
    mock_manifest_path = mock_chroma_dir / "ingest_manifest.json"
    mock_chroma_dir.mkdir(parents=True)

    monkeypatch.setattr("src.ingest.CHROMA_DIR", mock_chroma_dir)
    monkeypatch.setattr("src.ingest.MANIFEST_PATH", mock_manifest_path)

    # Save manifest with model_v1
    save_ingest_manifest("model_v1", {"momo_A": 10, "momo_B": 10})
    assert mock_manifest_path.exists()

    loaded = load_ingest_manifest()
    assert loaded["embedding_model"] == "model_v1"
    assert loaded["collections"]["momo_A"] == 10

    # Mock configs and run_ingest with a different model_v2
    monkeypatch.setattr("src.ingest.get_api_key", lambda: "fake_key")
    monkeypatch.setattr(
        "src.ingest.load_models_config",
        lambda: {"embedding_model": "model_v2"},
    )

    with pytest.raises(ValueError, match="Vector không tương thích"):
        run_ingest(rebuild=False)
