"""Unit tests for fetch_docs with mock requests, BeautifulSoup fallback, and discovery."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import yaml

from src.fetch_docs import (
    discover_url,
    extract_content,
    extract_with_bs4,
    fetch_and_process,
)


def test_discover_privacy_url_from_fake_footer():
    fake_html = """
    <html>
      <body>
        <main><h1>Trang chủ MoMo</h1></main>
        <footer>
          <div class="footer-links">
            <a href="/dieu-khoan">Điều khoản</a>
            <a href="/chinh-sach-quyen-rieng-tu">Chính sách Quyền riêng tư</a>
            <a href="/lien-he">Liên hệ</a>
          </div>
        </footer>
      </body>
    </html>
    """
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.text = fake_html
    mock_response.apparent_encoding = "utf-8"
    mock_session.get.return_value = mock_response

    discovered = discover_url(
        discover_from="https://momo.vn/",
        link_text_regex="quyền riêng tư",
        session=mock_session,
    )
    assert discovered == "https://momo.vn/chinh-sach-quyen-rieng-tu"


def test_fallback_to_bs4_when_trafilatura_returns_none(monkeypatch):
    html = """
    <html>
      <body>
        <main>
          <h1>Tiêu đề tài liệu MoMo</h1>
          <p>Nội dung đoạn văn thứ nhất về dịch vụ ví điện tử.</p>
          <h2>Quy định bảo mật</h2>
          <ul>
            <li>Điều kiện một</li>
            <li>Điều kiện hai</li>
          </ul>
        </main>
      </body>
    </html>
    """
    # Force trafilatura to return None to trigger BS4 fallback
    monkeypatch.setattr("trafilatura.extract", lambda *args, **kwargs: None)

    extracted = extract_content(html)
    assert "# Tiêu đề tài liệu MoMo" in extracted
    assert "## Quy định bảo mật" in extracted
    assert "- Điều kiện một" in extracted


def test_fetch_prioritizes_manual_files(tmp_path, monkeypatch):
    mock_data = tmp_path / "data"
    mock_raw = mock_data / "raw"
    mock_manual = mock_data / "manual"
    mock_processed = mock_data / "processed"

    for d in [mock_data, mock_raw, mock_manual, mock_processed]:
        d.mkdir(parents=True)

    monkeypatch.setattr("src.fetch_docs.DATA_DIR", mock_data)
    monkeypatch.setattr("src.fetch_docs.RAW_DATA_DIR", mock_raw)
    monkeypatch.setattr("src.fetch_docs.MANUAL_DATA_DIR", mock_manual)
    monkeypatch.setattr("src.fetch_docs.PROCESSED_DATA_DIR", mock_processed)

    # Write sources.yaml
    sources_content = {
        "docs": [
            {
                "doc_id": "test_manual_doc",
                "title": "Tài liệu thủ công",
                "url": "https://momo.vn/fake",
            }
        ]
    }
    with open(mock_data / "sources.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sources_content, f)

    # Write manual override file with > 500 chars
    manual_content = "Đây là nội dung người dùng tự dán tay vào thư mục manual. " * 20
    with open(mock_manual / "test_manual_doc.md", "w", encoding="utf-8") as f:
        f.write(manual_content)

    # Mock requests to make sure no web calls are made
    mock_session = MagicMock()
    mock_session.get.side_effect = RuntimeError("Không được gọi mạng khi có bản manual!")
    monkeypatch.setattr("requests.Session", lambda: mock_session)

    summary = fetch_and_process()
    assert "test_manual_doc" in summary
    assert "OK" in summary["test_manual_doc"]

    # Verify processed file matches manual content
    with open(mock_processed / "test_manual_doc.md", "r", encoding="utf-8") as f:
        saved_text = f.read()
    assert "tự dán tay vào thư mục manual" in saved_text


def test_short_content_warning(tmp_path, monkeypatch):
    mock_data = tmp_path / "data"
    mock_raw = mock_data / "raw"
    mock_manual = mock_data / "manual"
    mock_processed = mock_data / "processed"

    for d in [mock_data, mock_raw, mock_manual, mock_processed]:
        d.mkdir(parents=True)

    monkeypatch.setattr("src.fetch_docs.DATA_DIR", mock_data)
    monkeypatch.setattr("src.fetch_docs.RAW_DATA_DIR", mock_raw)
    monkeypatch.setattr("src.fetch_docs.MANUAL_DATA_DIR", mock_manual)
    monkeypatch.setattr("src.fetch_docs.PROCESSED_DATA_DIR", mock_processed)

    sources_content = {
        "docs": [
            {
                "doc_id": "short_doc",
                "title": "Tài liệu ngắn",
                "url": "https://momo.vn/short",
            }
        ]
    }
    with open(mock_data / "sources.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sources_content, f)

    # Manual content with only 50 characters (< 500)
    with open(mock_manual / "short_doc.md", "w", encoding="utf-8") as f:
        f.write("Nội dung quá ngắn để cấu thành văn bản đầy đủ.")

    summary = fetch_and_process()
    assert "CẢNH BÁO: < 500 ký tự" in summary["short_doc"]
