"""Unit tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_configs_endpoint():
    """Verify /configs endpoint returns list of available config IDs."""
    response = client.get("/configs")
    assert response.status_code == 200
    data = response.json()
    assert "configs" in data
    assert "A" in data["configs"]
    assert "B" in data["configs"]


@patch("src.api.ask")
def test_ask_endpoint_success(mock_ask):
    """Verify /ask endpoint handles valid request and returns contract schema."""
    mock_ask.return_value = {
        "question": "Hạn mức MoMo là bao nhiêu?",
        "config_id": "A",
        "answer": "Hạn mức tối đa là 50 triệu đồng.",
        "sources": [
            {
                "index": 1,
                "chunk_id": "terms-A-001",
                "doc_id": "terms",
                "title": "Điều khoản sử dụng",
                "source_url": "https://momo.vn/terms",
                "score": 0.92,
                "text": "Hạn mức tối đa là 50 triệu đồng.",
                "section": "Hạn mức",
            }
        ],
        "final_prompt": "Prompt text...",
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "cost_usd": 0.00015,
        "latency_ms": {"retrieval": 15.2, "generation": 450.1, "total": 465.3},
        "error": None,
    }

    payload = {
        "question": "Hạn mức MoMo là bao nhiêu?",
        "config_id": "A",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hạn mức tối đa là 50 triệu đồng."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "terms-A-001"
    assert data["usage"]["total_tokens"] == 120
    assert data["latency_ms"]["total"] == 465.3


def test_ask_endpoint_invalid_config():
    """Verify /ask returns 400 for unknown config_id."""
    payload = {
        "question": "Câu hỏi test",
        "config_id": "INVALID_CONFIG",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "INVALID_CONFIG" in detail

