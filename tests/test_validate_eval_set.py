"""Unit tests for validate_eval_set.py using mock corpus and eval files."""

import json
from pathlib import Path
import pytest

from eval.validate_eval_set import validate_eval_set


@pytest.fixture
def mock_corpus_and_eval(tmp_path):
    """Create mock processed documents and eval files."""
    corpus_dir = tmp_path / "processed"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: terms
    terms_file = corpus_dir / "momo_terms.md"
    terms_file.write_text(
        "# Điều khoản MoMo\n\nHạn mức giao dịch tối đa là 50 triệu đồng mỗi ngày.\nKhách hàng cần xác thực KYC.",
        encoding="utf-8",
    )

    # Document 2: faq
    faq_file = corpus_dir / "momo_faq.md"
    faq_file.write_text(
        "# Hỏi đáp MoMo\n\nPhí duy trì tài khoản MoMo là hoàn toàn miễn phí.",
        encoding="utf-8",
    )

    return corpus_dir, tmp_path


def test_valid_eval_items(mock_corpus_and_eval):
    """Verify that correctly formatted items produce 0 errors."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_valid.jsonl"

    items = [
        {
            "id": "q1",
            "question": "Hạn mức giao dịch MoMo là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "50 triệu đồng mỗi ngày",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": ["Hạn mức giao dịch tối đa là 50 triệu đồng mỗi ngày."],
            "notes": "Direct query",
        },
        {
            "id": "q2",
            "question": "MoMo có thể dùng để chuyển tiền ra sao Hỏa không?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Out of scope",
        },
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert len(errors) == 0
    assert stats["total_questions"] == 2
    assert stats["type_counts"]["direct"] == 1
    assert stats["type_counts"]["unanswerable"] == 1


def test_invalid_evidence_and_suggestion(mock_corpus_and_eval):
    """Verify error reporting and RapidFuzz suggestion when evidence doesn't match verbatim."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_invalid_evidence.jsonl"

    items = [
        {
            "id": "q1",
            "question": "Hạn mức MoMo là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "50 triệu",
            "gold_doc_ids": ["momo_terms"],
            # Altered text: "tối đa là 100 triệu" instead of "50 triệu"
            "gold_evidence": ["Hạn mức giao dịch tối đa là 100 triệu đồng mỗi ngày."],
            "notes": "Invalid evidence",
        }
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert len(errors) == 1
    assert "KHÔNG tìm thấy nguyên văn" in errors[0]
    assert "Gợi ý đoạn gần giống nhất" in errors[0]


def test_unanswerable_constraints(mock_corpus_and_eval):
    """Verify error when unanswerable questions violate empty doc_ids or evidence rules."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_unans_violation.jsonl"

    items = [
        {
            "id": "q_err",
            "question": "Câu hỏi không có trong tài liệu?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": ["momo_terms"],  # Violation: should be empty
            "gold_evidence": ["Một câu nào đó"],  # Violation: should be empty
            "notes": "",
        }
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert any("answerable=false nhưng gold_doc_ids không rỗng" in e for e in errors)
    assert any("answerable=false nhưng gold_evidence không rỗng" in e for e in errors)


def test_duplicate_id_detection(mock_corpus_and_eval):
    """Verify duplicate IDs are caught."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_dup.jsonl"

    items = [
        {
            "id": "q_same",
            "question": "Câu hỏi số 1",
            "type": "unanswerable",
            "answerable": False,
        },
        {
            "id": "q_same",
            "question": "Câu hỏi số 2",
            "type": "unanswerable",
            "answerable": False,
        },
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert any("Trùng lặp ID 'q_same'" in e for e in errors)


def test_missing_doc_id_detection(mock_corpus_and_eval):
    """Verify missing doc_id is caught."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_missing_doc.jsonl"

    items = [
        {
            "id": "q_missing",
            "question": "Câu hỏi tham chiếu tài liệu không tồn tại?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Answer",
            "gold_doc_ids": ["doc_khong_ton_tai"],
            "gold_evidence": ["Some text"],
        }
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert any("không tồn tại trong corpus" in e for e in errors)


def test_warnings_for_near_duplicates_and_long_evidence(mock_corpus_and_eval):
    """Verify quality warnings for near-duplicate questions and long evidence (>200 chars)."""
    corpus_dir, tmp_path = mock_corpus_and_eval
    eval_file = tmp_path / "eval_warnings.jsonl"

    long_evidence = "A" * 205
    # Write long evidence into doc
    doc_path = corpus_dir / "momo_terms.md"
    doc_path.write_text(long_evidence, encoding="utf-8")

    items = [
        {
            "id": "q1",
            "question": "Làm thế nào để đổi mật khẩu ví MoMo của tôi?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "ans",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [long_evidence],
        },
        {
            "id": "q2",
            "question": "Làm thế nào để đổi mật khẩu ví MoMo?",  # Very high similarity to q1
            "type": "direct",
            "answerable": True,
            "gold_answer": "ans",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [long_evidence],
        },
    ]

    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    errors, warnings, stats = validate_eval_set(eval_file, corpus_dir=corpus_dir)
    assert any("> 200 ký tự" in w for w in warnings)
    assert any("gần trùng lặp" in w for w in warnings)

