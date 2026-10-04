"""Unit tests for context assembly module."""

import pytest
from src.context import build_context


def test_build_context_empty():
    """Verify build_context returns empty string and list for empty chunks."""
    context_str, sources = build_context([])
    assert context_str == ""
    assert sources == []


def test_build_context_numbering_and_formatting():
    """Verify 1-indexed numbering and source header formatting."""
    chunks = [
        {
            "chunk_id": "doc1-A-001",
            "doc_id": "doc1",
            "title": "Điều khoản MoMo",
            "source_url": "https://momo.vn/terms",
            "section": "Điều 1",
            "text": "Nội dung điều 1.",
            "score": 0.85,
        },
        {
            "chunk_id": "doc2-A-001",
            "doc_id": "doc2",
            "title": "Hỏi đáp MoMo",
            "source_url": "https://momo.vn/faq",
            "section": "",
            "text": "Nội dung câu hỏi thường gặp.",
            "score": 0.75,
        },
    ]

    context_str, sources = build_context(chunks, max_context_chars=5000)

    assert len(sources) == 2
    assert sources[0]["index"] == 1
    assert sources[0]["chunk_id"] == "doc1-A-001"
    assert sources[1]["index"] == 2
    assert sources[1]["chunk_id"] == "doc2-A-001"

    expected_part1 = "[1] (Điều khoản MoMo - Điều 1)\nNội dung điều 1."
    expected_part2 = "[2] (Hỏi đáp MoMo)\nNội dung câu hỏi thường gặp."

    assert expected_part1 in context_str
    assert expected_part2 in context_str
    assert context_str == f"{expected_part1}\n\n{expected_part2}"


def test_build_context_deduplication():
    """Verify chunks with duplicate text content are skipped."""
    chunks = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc1",
            "title": "Doc 1",
            "source_url": "url1",
            "text": "Nội dung trùng lặp hoàn toàn.",
            "score": 0.9,
        },
        {
            "chunk_id": "chunk-2",
            "doc_id": "doc2",
            "title": "Doc 2",
            "source_url": "url2",
            "text": "Nội dung trùng lặp hoàn toàn.",
            "score": 0.8,
        },
        {
            "chunk_id": "chunk-3",
            "doc_id": "doc3",
            "title": "Doc 3",
            "source_url": "url3",
            "text": "Nội dung khác duy nhất.",
            "score": 0.7,
        },
    ]

    context_str, sources = build_context(chunks, max_context_chars=5000)

    assert len(sources) == 2
    assert sources[0]["chunk_id"] == "chunk-1"
    assert sources[0]["index"] == 1
    assert sources[1]["chunk_id"] == "chunk-3"
    assert sources[1]["index"] == 2
    assert "Doc 2" not in context_str


def test_build_context_max_chars_boundary():
    """Verify context stops adding chunks when exceeding max_context_chars without cutting mid-chunk."""
    chunk1 = {
        "chunk_id": "c1",
        "title": "Title 1",
        "text": "A" * 100,
    }
    chunk2 = {
        "chunk_id": "c2",
        "title": "Title 2",
        "text": "B" * 100,
    }

    # Block 1 is approx 112 chars: "[1] (Title 1)\n" + 100 A's
    # Setting max_context_chars to 150 allows chunk1 but disallows chunk2
    context_str, sources = build_context([chunk1, chunk2], max_context_chars=150)

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "c1"
    assert "Title 2" not in context_str
    assert len(context_str) <= 150

