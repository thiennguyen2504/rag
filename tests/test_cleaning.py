"""Unit tests for text cleaning without network or API calls."""

import unicodedata
import pytest
from src.fetch_docs import clean_text


def test_nfc_normalization():
    # NFD: 'e' + combining acute accent
    decomposed = "To\u0300i te\u0302n la\u0300 MoMo"
    cleaned = clean_text(decomposed)
    # Result must be NFC
    assert cleaned == unicodedata.normalize("NFC", decomposed)
    assert "\u0300" not in cleaned


def test_nbsp_replacement():
    raw = "Khoản\u00a0tiền\u00a0thanh\u00a0toán\u00a0MoMo"
    cleaned = clean_text(raw)
    assert "\u00a0" not in cleaned
    assert "Khoản tiền thanh toán MoMo" == cleaned


def test_boilerplate_removal():
    raw = (
        "Nội dung chính sách quan trọng.\n"
        "Tải ứng dụng ngay hôm nay!\n"
        "Hotline CSKH: 1900 5454 41\n"
        "Điều khoản áp dụng cho người dùng.\n"
        "Navigation\n"
        "GỌI NGAY\n"
        "Bản quyền thuộc về Công ty MoMo."
    )
    cleaned = clean_text(raw)
    assert "Nội dung chính sách quan trọng." in cleaned
    assert "Điều khoản áp dụng cho người dùng." in cleaned
    assert "Tải ứng dụng" not in cleaned
    assert "Hotline CSKH" not in cleaned
    assert "Navigation" not in cleaned
    assert "GỌI NGAY" not in cleaned
    assert "Bản quyền thuộc về" not in cleaned


def test_consecutive_blank_lines_collapse():
    raw = "Dòng 1\n\n\n\n\n\nDòng 2\n\n\n\nDòng 3"
    cleaned = clean_text(raw)
    assert cleaned == "Dòng 1\n\nDòng 2\n\nDòng 3"
    assert "\n\n\n" not in cleaned


def test_consecutive_duplicate_lines_removal():
    raw = (
        "Tiêu đề chính\n"
        "Nội dung lặp lại\n"
        "Nội dung lặp lại\n"
        "Nội dung tiếp theo\n"
        "Nội dung tiếp theo\n"
        "Nội dung tiếp theo\n"
        "Kết thúc"
    )
    cleaned = clean_text(raw)
    expected = "Tiêu đề chính\nNội dung lặp lại\nNội dung tiếp theo\nKết thúc"
    assert cleaned == expected


def test_trailing_whitespace_stripped():
    raw = "Dòng có khoảng trắng thừa cuối    \nDòng thứ hai   \t"
    cleaned = clean_text(raw)
    lines = cleaned.splitlines()
    assert lines[0] == "Dòng có khoảng trắng thừa cuối"
    assert lines[1] == "Dòng thứ hai"
