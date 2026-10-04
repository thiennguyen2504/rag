"""Unit tests for evaluation metrics module."""

import pytest
from eval.metrics import (
    analyze_citations,
    check_chunk_contains_evidence,
    compute_metrics_for_items,
    compute_retrieval_stats,
    normalize_text,
    safe_div,
)


def test_normalize_text():
    """Verify whitespace collapsing and lowercasing."""
    raw = "  Điều   Khoản \n\t Sử DỤNG  MoMo  "
    expected = "điều khoản sử dụng momo"
    assert normalize_text(raw) == expected


def test_evidence_matching():
    """Verify exact match, fuzzy match (>=90%), and miss."""
    chunk = normalize_text("Người dùng có thể liên kết tài khoản ngân hàng với ví MoMo qua mục Ví của tôi.")

    evidence_exact = normalize_text("liên kết tài khoản ngân hàng với ví MoMo")
    assert check_chunk_contains_evidence(chunk, evidence_exact) is True

    evidence_fuzzy = normalize_text("liên kết tài khoán ngân hàng với ví MoMo")
    assert check_chunk_contains_evidence(chunk, evidence_fuzzy) is True

    evidence_miss = normalize_text("rút tiền mặt tại quầy giao dịch bưu điện")
    assert check_chunk_contains_evidence(chunk, evidence_miss) is False


def test_retrieval_stats_and_mrr():
    """Verify doc_hit, evidence_hit, recall, and MRR calculations."""
    chunks_lookup = {
        "chk-1": {"chunk_id": "chk-1", "doc_id": "doc_other", "text": "Nội dung không liên quan."},
        "chk-2": {"chunk_id": "chk-2", "doc_id": "doc_terms", "text": "Hạn mức tối đa là 50 triệu đồng mỗi ngày."},
        "chk-3": {"chunk_id": "chk-3", "doc_id": "doc_faq", "text": "Phí duy trì tài khoản là hoàn toàn miễn phí."},
    }

    retrieved = [
        {"chunk_id": "chk-1", "doc_id": "doc_other", "score": 0.9},
        {"chunk_id": "chk-2", "doc_id": "doc_terms", "score": 0.8},
    ]
    gold_doc_ids = ["doc_terms"]
    gold_evidences = ["hạn mức tối đa là 50 triệu đồng"]

    stats = compute_retrieval_stats(retrieved, gold_doc_ids, gold_evidences, chunks_lookup)
    assert stats["doc_hit"] is True
    assert stats["evidence_hit"] is True
    assert stats["evidence_recall"] == 1.0
    assert stats["mrr"] == 0.5

    gold_evidences_multi = ["hạn mức tối đa là 50 triệu đồng", "cần xác thực khuôn mặt"]
    stats_multi = compute_retrieval_stats(retrieved, gold_doc_ids, gold_evidences_multi, chunks_lookup)
    assert stats_multi["evidence_recall"] == 0.5

    stats_miss = compute_retrieval_stats(
        [{"chunk_id": "chk-1", "doc_id": "doc_other", "score": 0.9}],
        ["doc_other"],
        ["thông tin không có"],
        chunks_lookup,
    )
    assert stats_miss["doc_hit"] is True
    assert stats_miss["evidence_hit"] is False
    assert stats_miss["evidence_recall"] == 0.0
    assert stats_miss["mrr"] == 0.0


def test_analyze_citations():
    """Verify citation marker extraction and validity against total sources."""
    answer_valid = "Theo quy định [1], hạn mức là 50 triệu và miễn phí [2]."
    res_valid = analyze_citations(answer_valid, num_sources=2)
    assert res_valid["has_citation"] is True
    assert res_valid["valid_citations"] is True
    assert res_valid["total_citations"] == 2

    answer_invalid = "Theo quy định [1] và chi tiết tại [3]."
    res_invalid = analyze_citations(answer_invalid, num_sources=2)
    assert res_invalid["has_citation"] is True
    assert res_invalid["valid_citations"] is False

    answer_none = "Không có số hiệu trích dẫn nào trong câu trả lời."
    res_none = analyze_citations(answer_none, num_sources=2)
    assert res_none["has_citation"] is False
    assert res_none["valid_citations"] is True


def test_compute_metrics_and_hallucination_rates():
    """Verify aggregate metrics including hallucination and refusal rates."""
    items = [
        {
            "question_id": "q1",
            "type": "direct",
            "answerable": True,
            "doc_hit": True,
            "evidence_hit": True,
            "evidence_recall": 1.0,
            "mrr": 1.0,
            "judgement": {"correctness": "correct", "faithful": True, "refused": False},
            "latency_ms": {"retrieval": 100.0, "generation": 400.0, "total": 500.0},
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "cost_usd": 0.0001,
            "citation_info": {"has_citation": True, "valid_citations": True},
            "error": None,
        },
        {
            "question_id": "q2",
            "type": "direct",
            "answerable": True,
            "doc_hit": False,
            "evidence_hit": False,
            "evidence_recall": 0.0,
            "mrr": 0.0,
            "judgement": {"correctness": "partial", "faithful": True, "refused": True},
            "latency_ms": {"retrieval": 200.0, "generation": 300.0, "total": 500.0},
            "usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            "cost_usd": 0.0001,
            "citation_info": {"has_citation": False, "valid_citations": True},
            "error": None,
        },
        {
            "question_id": "q3",
            "type": "unanswerable",
            "answerable": False,
            "doc_hit": None,
            "evidence_hit": None,
            "evidence_recall": None,
            "mrr": None,
            "judgement": {"correctness": "correct", "faithful": True, "refused": True},
            "latency_ms": {"retrieval": 50.0, "generation": 100.0, "total": 150.0},
            "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            "cost_usd": 0.00005,
            "citation_info": {"has_citation": False, "valid_citations": True},
            "error": None,
        },
        {
            "question_id": "q4",
            "type": "unanswerable",
            "answerable": False,
            "doc_hit": None,
            "evidence_hit": None,
            "evidence_recall": None,
            "mrr": None,
            "judgement": {"correctness": "incorrect", "faithful": False, "refused": False},
            "latency_ms": {"retrieval": 80.0, "generation": 500.0, "total": 580.0},
            "usage": {"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
            "cost_usd": 0.0002,
            "citation_info": {"has_citation": True, "valid_citations": True},
            "error": None,
        },
    ]

    metrics = compute_metrics_for_items(items)

    assert metrics["count"] == 4
    assert metrics["answerable_count"] == 2
    assert metrics["unanswerable_count"] == 2

    assert metrics["retrieval"]["doc_hit_rate"] == 0.5
    assert metrics["retrieval"]["evidence_hit_rate"] == 0.5
    assert metrics["retrieval"]["mrr"] == 0.5

    assert metrics["generation"]["correct_rate"] == 0.5
    assert metrics["generation"]["partial_rate"] == 0.25
    assert metrics["generation"]["incorrect_rate"] == 0.25
    assert metrics["generation"]["faithful_rate"] == 0.75

    assert metrics["hallucination"]["unanswerable_hallucination_rate"] == 0.5
    assert metrics["hallucination"]["answerable_refusal_rate"] == 0.5
    assert metrics["hallucination"]["unfaithful_rate"] == 0.25

    assert metrics["operational"]["latency_p50_ms"]["total"] == 500.0
    assert safe_div(10, 0, 0.0) == 0.0

