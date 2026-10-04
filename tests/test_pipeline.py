"""Unit tests for RAG pipeline module."""

import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import RAGPipeline, ask


@pytest.fixture
def mock_pipeline_dependencies():
    """Mock external dependencies (Retriever and ChatGoogleGenerativeAI) for RAGPipeline."""
    with patch("src.pipeline.Retriever") as MockRetriever, \
         patch("src.pipeline.ChatGoogleGenerativeAI") as MockLLM:
        yield MockRetriever, MockLLM


def test_pipeline_ask_normal_flow(mock_pipeline_dependencies):
    """Verify normal ask execution flow with mock retriever and mock LLM."""
    MockRetriever, MockLLM = mock_pipeline_dependencies

    # Setup retriever mock
    retriever_instance = MockRetriever.return_value
    retriever_instance.retrieve.return_value = [
        {
            "chunk_id": "test-doc-001",
            "doc_id": "test-doc",
            "title": "Hướng dẫn sử dụng",
            "section": "Mục 1",
            "source_url": "https://momo.vn/guide",
            "score": 0.88,
            "text": "Để liên kết ngân hàng, mở ứng dụng và chọn Ví của tôi.",
        }
    ]

    # Setup LLM mock
    llm_instance = MockLLM.return_value
    mock_response = MagicMock()
    mock_response.content = "Bạn có thể liên kết ngân hàng từ mục Ví của tôi."
    mock_response.usage_metadata = {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}
    llm_instance.invoke.return_value = mock_response

    pipeline = RAGPipeline("A")
    result = pipeline.ask("Làm sao để liên kết ngân hàng?")

    assert result["config_id"] == "A"
    assert result["question"] == "Làm sao để liên kết ngân hàng?"
    assert result["answer"] == "Bạn có thể liên kết ngân hàng từ mục Ví của tôi."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "test-doc-001"
    assert result["usage"]["total_tokens"] == 180
    assert result["cost_usd"] >= 0.0
    assert result["latency_ms"]["retrieval"] >= 0.0
    assert result["latency_ms"]["generation"] >= 0.0
    assert result["latency_ms"]["total"] >= 0.0
    assert result["error"] is None


def test_pipeline_ask_short_circuit_empty_context(mock_pipeline_dependencies):
    """Verify short-circuit when empty context and short_circuit_on_empty_context is enabled."""
    MockRetriever, MockLLM = mock_pipeline_dependencies

    retriever_instance = MockRetriever.return_value
    retriever_instance.retrieve.return_value = []  # No chunks found

    pipeline = RAGPipeline("A")
    pipeline.config.short_circuit_on_empty_context = True

    result = pipeline.ask("Một câu hỏi ngoài lề không có tài liệu nào liên quan?")

    # LLM should NOT have been invoked
    pipeline.llm.invoke.assert_not_called()

    assert result["answer"] == "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
    assert result["sources"] == []
    assert result["usage"]["total_tokens"] == 0
    assert result["cost_usd"] == 0.0
    assert result["error"] is None


def test_pipeline_ask_exception_handling(mock_pipeline_dependencies):
    """Verify pipeline catches exceptions, records error string, and does not crash."""
    MockRetriever, MockLLM = mock_pipeline_dependencies

    retriever_instance = MockRetriever.return_value
    retriever_instance.retrieve.side_effect = RuntimeError("Chroma connection failed")

    pipeline = RAGPipeline("A")
    result = pipeline.ask("Câu hỏi test lỗi")

    assert result["error"] == "Chroma connection failed"
    assert result["answer"] == ""
    assert result["latency_ms"]["total"] >= 0.0

