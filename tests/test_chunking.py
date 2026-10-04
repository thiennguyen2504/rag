"""Unit tests for chunking strategies A and B (offline, no API key)."""

import pytest
from src.chunking import chunk_strategy_a, chunk_strategy_b

SYNTHETIC_DOC = """# Hướng Dẫn Dịch Vụ Ví MoMo

Đây là phần mở đầu giới thiệu tổng quan dịch vụ của chúng tôi.

## I. Quy định chung

Quy định này áp dụng cho toàn thể khách hàng đăng ký mở và sử dụng tài khoản Ví MoMo trên toàn quốc theo quy định hiện hành của pháp luật.

### Điều 1. Đăng ký tài khoản
Khách hàng có thể đăng ký tài khoản Ví MoMo bằng số điện thoại di động chính chủ. Mỗi số điện thoại chỉ được đăng ký một tài khoản duy nhất.

### Điều 2. Xác thực tài khoản
Khách hàng cần cung cấp CCCD gắn chip để xác thực danh tính điện tử theo quy định của Ngân hàng Nhà nước.

## II. Câu hỏi thường gặp

Làm sao để nạp tiền vào MoMo?
Bạn có thể nạp tiền từ tài khoản ngân hàng liên kết, thẻ ATM nội địa hoặc thẻ thanh toán quốc tế đã xác thực.

Rút tiền từ MoMo có mất phí không?
Rút tiền về tài khoản ngân hàng liên kết là hoàn toàn miễn phí theo hạn mức ưu đãi hàng tháng của MoMo.

Điều khoản ngắn.
Ok.
"""


def test_chunk_ids_unique_and_deterministic():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    chunks_a1 = chunk_strategy_a("doc_test", title, url, SYNTHETIC_DOC)
    chunks_a2 = chunk_strategy_a("doc_test", title, url, SYNTHETIC_DOC)

    ids_a1 = [c["chunk_id"] for c in chunks_a1]
    ids_a2 = [c["chunk_id"] for c in chunks_a2]

    # Deterministic
    assert ids_a1 == ids_a2
    # Unique
    assert len(ids_a1) == len(set(ids_a1))

    chunks_b1 = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)
    chunks_b2 = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)

    ids_b1 = [c["chunk_id"] for c in chunks_b1]
    ids_b2 = [c["chunk_id"] for c in chunks_b2]

    assert ids_b1 == ids_b2
    assert len(ids_b1) == len(set(ids_b1))


def test_strategy_b_context_line_prefix():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    chunks_b = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)

    for c in chunks_b:
        text = c["text"]
        lines = text.splitlines()
        first_line = lines[0]
        # Must start with document title
        assert first_line.startswith(title), f"Chunk B does not start with title: {first_line}"
        assert c["strategy"] == "B"
        assert len(c["text"].strip()) > 0


def test_faq_pair_not_separated_in_strategy_b():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    chunks_b = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)

    # Check if the question and answer about "Làm sao để nạp tiền vào MoMo?" are together
    found_faq = False
    for c in chunks_b:
        if "Làm sao để nạp tiền vào MoMo?" in c["text"]:
            found_faq = True
            # Both question and answer in same chunk
            assert "ngân hàng liên kết" in c["text"]
    assert found_faq, "FAQ question was not found in any chunk"


def test_short_sections_are_merged():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    chunks_b = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)

    # In SYNTHETIC_DOC, "Điều khoản ngắn.\nOk." is < 100 characters.
    # It must be merged and not exist as its own isolated chunk.
    for c in chunks_b:
        body_without_context = "\n".join(c["text"].splitlines()[1:])
        # Standalone chunk should not be just "Ok."
        assert body_without_context.strip() != "Ok."


def test_no_empty_chunks():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    chunks_a = chunk_strategy_a("doc_test", title, url, SYNTHETIC_DOC)
    chunks_b = chunk_strategy_b("doc_test", title, url, SYNTHETIC_DOC)

    for c in chunks_a:
        assert c["text"].strip() != ""
        assert len(c["text"].strip()) > 0

    for c in chunks_b:
        assert c["text"].strip() != ""
        assert len(c["text"].strip()) > 0


def test_chunk_a_length_constraint():
    title = "Hướng Dẫn Dịch Vụ Ví MoMo"
    url = "https://momo.vn/huong-dan"
    long_text = "MoMo là ví điện tử hàng đầu Việt Nam với hàng chục triệu người dùng. " * 50
    chunks_a = chunk_strategy_a("doc_test", title, url, long_text)

    for c in chunks_a:
        # Max chunk size is 500 characters, allow reasonable margin for whitespace
        assert len(c["text"]) <= 550, f"Chunk A exceeded chunk size: {len(c['text'])}"
