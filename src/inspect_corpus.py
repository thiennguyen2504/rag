"""Inspect processed corpus, report statistics, check boilerplate and detect quality issues."""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from src.config import DATA_DIR, PROCESSED_DATA_DIR

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def detect_issues(lines: List[str]) -> List[Dict[str, Any]]:
    """Detect text quality issues: replacement char, strange sequences, long lines."""
    issues = []
    strange_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    for idx, line in enumerate(lines, 1):
        # 1. Replacement char (\ufffd)
        if "\ufffd" in line:
            issues.append({
                "line_no": idx,
                "type": "Ký tự thay thế (\\ufffd)",
                "snippet": line[:100],
            })

        # 2. Control / strange characters
        if strange_pattern.search(line):
            issues.append({
                "line_no": idx,
                "type": "Ký tự điều khiển / lạ",
                "snippet": repr(line[:100]),
            })

        # 3. Excessively long line (> 1500 chars)
        if len(line) > 1500:
            issues.append({
                "line_no": idx,
                "type": f"Dòng quá dài ({len(line)} ký tự > 1500)",
                "snippet": line[:120] + "...",
            })

    return issues


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format aligned ASCII table."""
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


def inspect_corpus() -> None:
    if not PROCESSED_DATA_DIR.exists():
        print(f"[LỖI] Thư mục {PROCESSED_DATA_DIR} không tồn tại. Hãy chạy 'python -m src.fetch_docs' trước.", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(PROCESSED_DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"[LỖI] Không có file .md nào trong {PROCESSED_DATA_DIR}. Hãy chạy 'python -m src.fetch_docs' trước.", file=sys.stderr)
        sys.exit(1)

    print("=" * 100)
    print("THỐNG KÊ CHI TIẾT CORPUS MOMO (src/inspect_corpus.py)")
    print("=" * 100)

    headers = ["doc_id", "n_chars", "n_words", "Số dòng", "Dòng '#'", "Mẫu 'Điều N'", "Dòng '?' (FAQ)"]
    rows = []
    total_words = 0
    total_chars = 0
    file_contents = {}

    dieu_pattern = re.compile(r"\bĐiều\s+\d+", re.IGNORECASE)

    for file_path in md_files:
        doc_id = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        file_contents[doc_id] = lines

        n_chars = len(content)
        n_words = len(content.split())
        n_lines = len(lines)
        n_hash_lines = sum(1 for line in lines if line.strip().startswith("#"))
        n_dieu = len(dieu_pattern.findall(content))
        n_question_lines = sum(1 for line in lines if line.strip().endswith("?"))

        total_words += n_words
        total_chars += n_chars

        rows.append([
            doc_id,
            f"{n_chars:,}",
            f"{n_words:,}",
            str(n_lines),
            str(n_hash_lines),
            str(n_dieu),
            str(n_question_lines),
        ])

    print(format_table(headers, rows))
    print("-" * 100)
    print(f"TỔNG CỘNG: {len(md_files)} tài liệu | {total_chars:,} ký tự | {total_words:,} từ.")

    if total_words < 15000:
        print("\n" + "!" * 80)
        print(f"[CẢNH BÁO] Tổng số từ ({total_words:,} từ) < 15.000 từ. Cân nhắc thêm tài liệu trong data/sources.yaml.")
        print("!" * 80)
    else:
        print(f"\n[OK] Tổng số từ đạt chuẩn: {total_words:,} từ (>= 15.000 từ).")

    # Inspect first and last 15 lines per doc
    print("\n" + "=" * 100)
    print("KIỂM TRA 15 DÒNG ĐẦU VÀ 15 DÒNG CUỐI MỖI TÀI LIỆU (Phát hiện boilerplate)")
    print("=" * 100)

    for doc_id, lines in file_contents.items():
        print(f"\n--- [ {doc_id} ] (Tổng {len(lines)} dòng) ---")
        first_15 = lines[:15]
        last_15 = lines[-15:] if len(lines) > 15 else lines

        print("  >> 15 DÒNG ĐẦU:")
        for idx, line in enumerate(first_15, 1):
            print(f"    {idx:02d}: {line}")

        print("  >> 15 DÒNG CUỐI:")
        start_idx = max(1, len(lines) - len(last_15) + 1)
        for idx, line in enumerate(last_15, start_idx):
            print(f"    {idx:02d}: {line}")

    # Inspect quality issues
    print("\n" + "=" * 100)
    print("PHÁT HIỆN CHẤT LƯỢNG VĂN BẢN (Ký tự thay thế , ký tự lạ, dòng quá dài > 1500 ký tự)")
    print("=" * 100)

    any_issues = False
    for doc_id, lines in file_contents.items():
        issues = detect_issues(lines)
        if issues:
            any_issues = True
            print(f"\n[!] Phát hiện {len(issues)} vấn đề trong '{doc_id}':")
            for issue in issues[:10]:  # limit 10 per doc
                print(f"    - Dòng {issue['line_no']}: {issue['type']} -> {issue['snippet']}")
            if len(issues) > 10:
                print(f"    ... và {len(issues) - 10} vấn đề khác.")

    if not any_issues:
        print("\n[OK] Không phát hiện ký tự lạ, ký tự lỗi , hay dòng dài bất thường nào trong toàn bộ corpus!")

    print("\n" + "=" * 100)


def main() -> None:
    inspect_corpus()


if __name__ == "__main__":
    main()
