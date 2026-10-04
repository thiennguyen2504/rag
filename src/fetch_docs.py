"""Fetch documents from MoMo website or manual inputs, clean and build corpus manifest."""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
import trafilatura
import yaml

from src.config import (
    BASE_DIR,
    DATA_DIR,
    MANUAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.llm_utils import retry_with_backoff

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_docs")

# Configurable regex patterns for boilerplate lines to discard
BOILERPLATE_PATTERNS = [
    r"^Tải ứng dụng.*",
    r"^\[?Tải ứng dụng\]?.*",
    r"^Kết nối với MoMo.*",
    r"^Chia sẻ.*",
    r"^Bản quyền thuộc về.*",
    r"^Công ty Cổ phần Dịch vụ Di động Trực tuyến.*",
    r"^Địa chỉ:.*",
    r"^Hotline:.*",
    r"^Hotline CSKH.*",
    r"^Tổng đài CSKH:.*",
    r"^Email:.*",
    r"^Theo dõi chúng tôi.*",
    r"^Trang chủ\s*>\s*.*",
    r"^Menu\b.*",
    r"^Navigation$",
    r"^Câu hỏi theo chủ đề$",
    r"^Liên hệ với MoMo$",
    r"^Hướng dẫn trợ giúp trên$",
    r"^24/07$",
    r"^\(1000 đ/phút\)$",
    r"^GỌI NGAY$",
    r"^Đánh giá\s*:\s*$",
    r"^\d+/\d+$",
    r"^Xem thêm.*",
    r"^Đăng nhập\s*\|\s*Đăng ký.*",
    r"^.*Giấy phép cung ứng dịch vụ trung gian thanh toán.*",
    r"^.*Chính sách bảo mật.*Điều khoản và điều kiện.*",
    r"^©\s*\d{4}.*",
    r"^Quét mã để tải ứng dụng.*",
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


def clean_text(raw_text: str) -> str:
    """Clean raw extracted markdown or HTML text.

    Steps:
    1. Unicode NFC normalization.
    2. Replace NBSP (\\u00a0) with regular space.
    3. Strip trailing whitespace from each line.
    4. Remove boilerplate lines matching BOILERPLATE_PATTERNS.
    5. Remove consecutive duplicate lines.
    6. Merge >2 consecutive newlines into 2 (one blank line).
    """
    if not raw_text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", raw_text)

    # 2. Replace NBSP
    text = text.replace("\u00a0", " ")

    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    prev_line: Optional[str] = None

    for line in lines:
        line_stripped = line.rstrip()

        # Check boilerplate pattern
        stripped_content = line_stripped.strip()
        if stripped_content:
            if any(pattern.match(stripped_content) for pattern in compiled_patterns):
                continue

        # Check consecutive duplicates (allow empty lines to collapse later)
        if line_stripped and line_stripped == prev_line:
            continue

        cleaned_lines.append(line_stripped)
        if line_stripped:
            prev_line = line_stripped

    # Merge >2 consecutive newlines into 2
    merged_text = "\n".join(cleaned_lines)
    merged_text = re.sub(r"\n{3,}", "\n\n", merged_text)
    return merged_text.strip()


def extract_with_bs4(html: str) -> str:
    """Fallback text extractor using BeautifulSoup preserving structure."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, navigation and footer tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
        tag.decompose()

    # Prioritize main content containers
    content_root = soup.find("main") or soup.find("article") or soup.body
    if not content_root:
        return ""

    lines: List[str] = []

    def walk_element(element: Any) -> None:
        if element.name in ["h1", "h2", "h3", "h4"]:
            level = int(element.name[1])
            heading_text = element.get_text(separator=" ", strip=True)
            if heading_text:
                lines.append(f"\n{'#' * level} {heading_text}\n")
        elif element.name == "li":
            li_text = element.get_text(separator=" ", strip=True)
            if li_text:
                lines.append(f"- {li_text}")
        elif element.name in ["p", "div", "section"]:
            # Recurse or take direct text if no children with headings
            has_subheadings = bool(element.find(["h1", "h2", "h3", "h4", "p", "div", "li"]))
            if has_subheadings:
                for child in element.children:
                    if hasattr(child, "name") and child.name:
                        walk_element(child)
            else:
                txt = element.get_text(separator=" ", strip=True)
                if txt:
                    lines.append(txt)
        else:
            for child in element.children:
                if hasattr(child, "name") and child.name:
                    walk_element(child)

    walk_element(content_root)
    return "\n\n".join(lines)


def extract_content(html: str) -> str:
    """Extract readable markdown from HTML with Trafilatura, fallback to BS4."""
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
    )
    if extracted and len(extracted) >= 500:
        return extracted

    logger.info("Trafilatura trả về kết quả ngắn hoặc rỗng, kích hoạt fallback BeautifulSoup.")
    return extract_with_bs4(html)


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def fetch_url(url: str, session: Optional[requests.Session] = None) -> str:
    """Fetch URL with browser user agent and timeout."""
    s = session or requests.Session()
    resp = s.get(url, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    # Try UTF-8 encoding
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def discover_url(discover_from: str, link_text_regex: str, session: requests.Session) -> Optional[str]:
    """Discover URL by crawling a page and matching link anchor text."""
    logger.info("Đang tự động tìm link từ trang: %s với regex: '%s'", discover_from, link_text_regex)
    html = fetch_url(discover_from, session=session)
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(link_text_regex, re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        text = a.get_text(separator=" ", strip=True)
        if pattern.search(text):
            found_url = urljoin(discover_from, a["href"])
            logger.info("Đã tìm thấy URL: %s (anchor text: '%s')", found_url, text)
            return found_url

    return None


def compute_sha256(content: str) -> str:
    """Calculate SHA256 hexadecimal digest of text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_manifest() -> Dict[str, Any]:
    """Load corpus_manifest.json if exists."""
    manifest_path = DATA_DIR / "corpus_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Save corpus_manifest.json."""
    manifest_path = DATA_DIR / "corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def fetch_and_process(force: bool = False, only_doc_id: Optional[str] = None) -> Dict[str, str]:
    """Main orchestrator for fetching and processing corpus documents."""
    sources_file = DATA_DIR / "sources.yaml"
    if not sources_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {sources_file}")

    with open(sources_file, "r", encoding="utf-8") as f:
        sources_data = yaml.safe_load(f) or {}

    docs = sources_data.get("docs", [])
    manifest = load_manifest()
    results_summary: Dict[str, str] = {}

    session = requests.Session()

    for doc in docs:
        doc_id = doc.get("doc_id")
        title = doc.get("title", "")
        url = doc.get("url", "").strip()
        discover_from = doc.get("discover_from", "").strip()
        link_text_regex = doc.get("link_text_regex", "").strip()

        if only_doc_id and doc_id != only_doc_id:
            continue

        print(f"\n---> Đang xử lý tài liệu: [{doc_id}] - {title}")

        manual_file = MANUAL_DATA_DIR / f"{doc_id}.md"
        raw_file = RAW_DATA_DIR / f"{doc_id}.html"
        processed_file = PROCESSED_DATA_DIR / f"{doc_id}.md"

        source_kind = "web"
        raw_html = ""
        content_text = ""

        # Check manual override
        if manual_file.exists():
            logger.info("Phát hiện file thủ công tại %s -> Ưu tiên sử dụng bản manual.", manual_file)
            source_kind = "manual"
            with open(manual_file, "r", encoding="utf-8") as f:
                content_text = f.read()
            raw_sha256 = compute_sha256(content_text)
        else:
            # Need to fetch via web
            if not url and discover_from:
                try:
                    discovered = discover_url(discover_from, link_text_regex, session)
                    if discovered:
                        url = discovered
                    else:
                        warning_msg = (
                            f"CẢNH BÁO: Không tìm thấy link khớp '{link_text_regex}' trên {discover_from}. "
                            f"Bỏ qua doc_id: {doc_id}."
                        )
                        print(f"[CẢNH BÁO] {warning_msg}")
                        logger.warning(warning_msg)
                        results_summary[doc_id] = "CẢNH BÁO: Không tìm thấy link tự động"
                        continue
                except Exception as exc:
                    err_msg = f"Lỗi khi tìm link tự động cho {doc_id}: {exc}"
                    logger.error(err_msg)
                    results_summary[doc_id] = f"LỖI: {exc}"
                    continue

            if not url:
                warning_msg = f"URL của {doc_id} trống và không có discover_from. Bỏ qua."
                print(f"[CẢNH BÁO] {warning_msg}")
                results_summary[doc_id] = "CẢNH BÁO: Thiếu URL"
                continue

            # Fetch HTML or load existing raw
            if raw_file.exists() and not force:
                logger.info("File raw HTML đã tồn tại: %s (dùng lại, dùng --force để tải lại).", raw_file)
                with open(raw_file, "r", encoding="utf-8") as f:
                    raw_html = f.read()
            else:
                try:
                    logger.info("Đang tải HTML từ: %s ...", url)
                    raw_html = fetch_url(url, session=session)
                    with open(raw_file, "w", encoding="utf-8") as f:
                        f.write(raw_html)
                    time.sleep(1.0)  # Politeness delay
                except Exception as exc:
                    err_msg = f"Lỗi tải mạng cho {doc_id} ({url}): {exc}"
                    logger.error(err_msg)
                    print(f"[LỖI] {err_msg}")
                    results_summary[doc_id] = f"LỖI TẢI: {exc}"
                    continue

            raw_sha256 = compute_sha256(raw_html)
            content_text = extract_content(raw_html)

        # Clean text
        processed_text = clean_text(content_text)
        n_chars = len(processed_text)
        n_words = len(processed_text.split())
        processed_sha256 = compute_sha256(processed_text)

        # Save processed text
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(processed_text)

        # Manifest entry
        manifest[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "url": url,
            "source_kind": source_kind,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": raw_sha256,
            "processed_sha256": processed_sha256,
            "n_chars": n_chars,
            "n_words": n_words,
        }

        # Check threshold
        if n_chars < 500:
            print("\n" + "!" * 80)
            print(
                f"[CẢNH BÁO QUAN TRỌNG] {doc_id}: Văn bản sau xử lý quá ngắn ({n_chars} ký tự < 500).\n"
                f"Trang có thể render bằng JavaScript hoặc nội dung accordion bị ẩn.\n"
                f"Hãy sao chép nội dung đầy đủ và dán vào data/manual/{doc_id}.md rồi chạy lại lệnh này!"
            )
            print("!" * 80 + "\n")
            results_summary[doc_id] = f"CẢNH BÁO: < 500 ký tự ({n_chars} chars)"
        else:
            print(f"[OK] Đã lưu {processed_file} ({n_chars} ký tự, {n_words} từ)")
            results_summary[doc_id] = f"OK ({n_chars} chars, {n_words} words)"

    save_manifest(manifest)
    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean corpus documents.")
    parser.add_argument("--force", action="store_true", help="Bắt buộc tải lại HTML dù file raw đã tồn tại")
    parser.add_argument("--only", type=str, default=None, help="Chỉ xử lý một doc_id cụ thể")
    args = parser.parse_args()

    print("=" * 80)
    print("BẮT ĐẦU THU THẬP VÀ XỬ LÝ CORPUS MOMO (src/fetch_docs.py)")
    print("=" * 80)

    summary = fetch_and_process(force=args.force, only_doc_id=args.only)

    print("\n" + "=" * 80)
    print("TỔNG KẾT THU THẬP TÀI LIỆU:")
    print("=" * 80)
    for doc_id, status in summary.items():
        print(f"  - {doc_id:<20}: {status}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
