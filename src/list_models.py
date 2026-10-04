"""List available Gemini models from Google AI Studio REST API."""

import sys
from typing import Any, Dict, List
import requests
from src.config import get_api_key
from src.llm_utils import retry_with_backoff

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def fetch_models_page(api_key: str, page_token: str = "") -> Dict[str, Any]:
    """Fetch one page of models from Generative Language API."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"x-goog-api-key": api_key}
    params: Dict[str, Any] = {"pageSize": 100}
    if page_token:
        params["pageToken"] = page_token

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_all_models(api_key: str) -> List[Dict[str, Any]]:
    """Retrieve all available models through pagination."""
    all_models: List[Dict[str, Any]] = []
    page_token = ""

    while True:
        data = fetch_models_page(api_key, page_token)
        models = data.get("models", [])
        all_models.extend(models)

        page_token = data.get("nextPageToken", "")
        if not page_token:
            break

    return all_models


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format tabular data into an aligned ASCII table."""
    if not rows:
        return "(Không có dữ liệu)"

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]

    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def main() -> None:
    try:
        api_key = get_api_key()
    except Exception as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(1)

    print("Đang truy vấn danh sách models từ Google AI Studio...")
    try:
        models = get_all_models(api_key)
    except Exception as exc:
        print(f"[LỖI] Không thể kết nối tới Google AI Studio: {exc}", file=sys.stderr)
        sys.exit(1)

    gen_models: List[Dict[str, Any]] = []
    embed_models: List[Dict[str, Any]] = []

    for m in models:
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            gen_models.append(m)
        if "embedContent" in methods:
            embed_models.append(m)

    headers = ["Model Name", "Display Name", "Input Limit", "Supported Methods"]

    print("\n" + "=" * 90)
    print(" 1. CÁC MODEL HỖ TRỢ SINH VĂN BẢN (generateContent) - Dùng cho generator & judge")
    print("=" * 90)
    gen_rows = [
        [
            m.get("name", "").replace("models/", ""),
            m.get("displayName", "")[:30],
            str(m.get("inputTokenLimit", "N/A")),
            ", ".join(m.get("supportedGenerationMethods", []))[:35],
        ]
        for m in gen_models
    ]
    print(format_table(headers, gen_rows))

    print("\n" + "=" * 90)
    print(" 2. CÁC MODEL HỖ TRỢ EMBEDDING (embedContent) - Dùng cho embedding_model")
    print("=" * 90)
    embed_rows = [
        [
            m.get("name", "").replace("models/", ""),
            m.get("displayName", "")[:30],
            str(m.get("inputTokenLimit", "N/A")),
            ", ".join(m.get("supportedGenerationMethods", []))[:35],
        ]
        for m in embed_models
    ]
    print(format_table(headers, embed_rows))

    print("\n" + "=" * 90)
    print("HƯỚNG DẪN:")
    print("Hãy chọn tên model phù hợp ở trên và điền vào configs/models.yaml:")
    print("  - generator_model: ví dụ gemini-2.5-flash (hoặc gemini-1.5-flash / gemini-2.0-flash)")
    print("  - judge_model: ví dụ gemini-2.5-flash (hoặc gemini-2.5-pro)")
    print("  - embedding_model: ví dụ text-embedding-004 (hoặc gemini-embedding-001)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
