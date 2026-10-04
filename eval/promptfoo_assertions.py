"""Custom Promptfoo Python Assertions for MoMo RAG testing.

Provides programmatic assertions for testing RAG response quality:
1. assert_vietnamese_refusal: Checks if unanswerable queries are properly refused without hallucination.
2. assert_gold_evidence_present: Checks if ground truth keywords/facts appear in the answer.
3. assert_retrieval_doc_match: Checks if expected document IDs were retrieved.
"""

from typing import Any, Dict, Union


def assert_vietnamese_refusal(output: str, context: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
    """Assert that the model politely declines to answer when information is not in context."""
    answerable = context.get("vars", {}).get("answerable", True)
    if answerable:
        return {"pass": True, "score": 1.0, "reason": "Câu hỏi hợp lệ có thể trả lời."}

    text_lower = output.lower()
    refusal_keywords = [
        "không có thông tin",
        "không được đề cập",
        "không thấy thông tin",
        "không tìm thấy thông tin",
        "chưa có thông tin",
        "không có đề cập",
        "không chứa thông tin",
        "tôi không biết",
        "tài liệu không",
        "ngữ cảnh không",
    ]
    has_refusal = any(kw in text_lower for kw in refusal_keywords)
    if has_refusal:
        return {
            "pass": True,
            "score": 1.0,
            "reason": "Mô hình từ chối chính xác khi thiếu thông tin trong ngữ cảnh.",
        }
    return {
        "pass": False,
        "score": 0.0,
        "reason": f"Mô hình không từ chối rõ ràng đối với câu hỏi ngoài ngữ cảnh. Output: {output[:100]}",
    }


def assert_gold_evidence_present(output: str, context: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
    """Assert that key phrases from gold_answer are present in the response."""
    gold_answer = context.get("vars", {}).get("gold_answer", "")
    if not gold_answer:
        return {"pass": True, "score": 1.0, "reason": "Không có gold_answer để đối chiếu."}

    output_lower = output.lower()
    words = [w for w in gold_answer.lower().replace(",", " ").replace(".", " ").split() if len(w) >= 3]
    if not words:
        return {"pass": True, "score": 1.0, "reason": "Gold answer quá ngắn."}

    matched_words = [w for w in words if w in output_lower]
    overlap_ratio = len(matched_words) / len(words)

    passed = overlap_ratio >= 0.35
    return {
        "pass": passed,
        "score": round(overlap_ratio, 2),
        "reason": f"Độ trùng khớp từ khóa với đáp án chuẩn: {overlap_ratio:.1%} ({len(matched_words)}/{len(words)} từ).",
    }


def assert_retrieval_doc_match(output: str, context: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
    """Assert that at least one expected document ID was retrieved in pipeline metadata."""
    expected_docs = context.get("vars", {}).get("gold_doc_ids", [])
    if not expected_docs:
        return {"pass": True, "score": 1.0, "reason": "Không yêu cầu doc_ids cụ thể."}

    provider_metadata = context.get("provider", {})
    if isinstance(provider_metadata, dict):
        retrieved_docs = provider_metadata.get("metadata", {}).get("retrieved_docs", [])
    else:
        retrieved_docs = []

    if not retrieved_docs:
        return {"pass": True, "score": 1.0, "reason": "Metadata retrieval không có sẵn trong context."}

    overlap = set(expected_docs).intersection(set(retrieved_docs))
    if overlap:
        return {
            "pass": True,
            "score": 1.0,
            "reason": f"Truy xuất đúng tài liệu: {list(overlap)}",
        }
    return {
        "pass": False,
        "score": 0.0,
        "reason": f"Không truy xuất được tài liệu mong đợi: kỳ vọng {expected_docs}, thực tế {retrieved_docs}",
    }


def get_assert(output: str, context: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
    """Default entrypoint called by Promptfoo python assertion."""
    answerable = context.get("vars", {}).get("answerable", True)
    if not answerable:
        return assert_vietnamese_refusal(output, context)
    return assert_gold_evidence_present(output, context)

