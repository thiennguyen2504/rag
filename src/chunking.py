"""Document chunking module implementing Strategy A and Strategy B."""

import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, PROCESSED_DATA_DIR

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chunking")

# ==============================================================================
# Configurable Heading Recognition Patterns for Strategy B
# ==============================================================================
# Patterns are evaluated in order. Each returns (level, heading_text).
# Levels:
#   1: Top-level document title (#)
#   2: Major sections (##, Roman numerals I-X, "Điều N", numbered "1.", bold titles)
#   3: Subsections (###, "1.1", short FAQ questions ending in '?')
#   4: Sub-subsections (####, "1.1.1")
#   5-6: Deep headings
HEADING_PATTERNS = [
    # 1. Markdown headings (# through ######)
    (r"^(#{1,6})\s+(.+)$", "markdown"),
    # 2. Roman numerals e.g., "##### **I. Định nghĩa**" or "I. Định nghĩa"
    (r"^(?:#{1,6}\s+)?\*{0,2}([IVXLCDM]+\.\s+[^*]+)\*{0,2}$", "roman"),
    # 3. "Điều N" e.g., "Điều 1.", "Điều 25. Kênh liên hệ..."
    (r"^(?:#{1,6}\s+)?\*{0,2}(Điều\s+\d+[:\.]?(?:\s+[^*]+)?)\*{0,2}$", "dieu"),
    # 4. Numbered outline e.g., "**1. Giải thích từ ngữ**" or "**1.1. MoMo:**" or "1. Mục tiêu"
    (r"^(?:#{1,6}\s+)?\*{0,2}(\d+(?:\.\d+)*)\.?\s+([^*]+)\*{0,2}$", "numbered"),
    # 5. Standalone bold title lines like "**Phạm vi áp dụng**", "**Mục đích xử lý dữ liệu**"
    (r"^\*\*([A-ZÀ-Ỹ0-9][^*]{2,80})\*\*[:]?$", "bold_title"),
    # 6. FAQ question lines ending with '?' (short lines <= 120 chars)
    (r"^([A-ZÀ-Ỹ0-9][^\n]{3,120}\?)$", "faq_question"),
]


def match_heading(line: str) -> Optional[Tuple[int, str]]:
    """Determine whether a line is a structural heading, and return its hierarchy level and clean title."""
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 150:
        return None

    # 1. Markdown heading
    md_match = re.match(r"^(#{1,6})\s+(.+)$", clean_line)
    if md_match:
        level = len(md_match.group(1))
        title = md_match.group(2).strip().strip("*").strip()
        return (level, title)

    # 2. Roman numeral
    roman_match = re.match(r"^(?:#{1,6}\s+)?\*{0,2}([IVXLCDM]+\.\s+[^*]+)\*{0,2}$", clean_line)
    if roman_match:
        title = roman_match.group(1).strip().strip("*").strip()
        return (2, title)

    # 3. Điều N
    dieu_match = re.match(r"^(?:#{1,6}\s+)?\*{0,2}(Điều\s+\d+[:\.]?(?:\s+[^*]+)?)\*{0,2}$", clean_line, re.IGNORECASE)
    if dieu_match:
        title = dieu_match.group(1).strip().strip("*").strip()
        return (2, title)

    # 4. Numbered outline (1., 1.1, 1.1.1)
    num_match = re.match(r"^(?:#{1,6}\s+)?\*{0,2}(\d+(?:\.\d+)*)\.?\s+([^*]+)\*{0,2}$", clean_line)
    if num_match:
        dots = num_match.group(1).count(".")
        level = min(2 + dots, 5)
        full_title = f"{num_match.group(1)}. {num_match.group(2).strip().strip('*').strip()}"
        # Avoid matching plain sentences that happen to start with a number
        if len(clean_line) < 120 and not (clean_line.endswith(".") and dots == 0 and len(clean_line) > 60):
            return (level, full_title)

    # 5. Standalone bold title
    bold_match = re.match(r"^\*\*([A-ZÀ-Ỹ0-9][^*]{2,80})\*\*[:]?$", clean_line)
    if bold_match:
        title = bold_match.group(1).strip()
        return (2, title)

    # 6. FAQ Question ending in '?'
    faq_match = re.match(r"^([A-ZÀ-Ỹ0-9][^\n]{3,120}\?)$", clean_line)
    if faq_match:
        title = faq_match.group(1).strip()
        return (3, title)

    return None


def chunk_strategy_a(doc_id: str, title: str, source_url: str, text: str) -> List[Dict[str, Any]]:
    """Strategy A: Naive chunking with RecursiveCharacterTextSplitter(500, 50).

    section = ""
    text = chunk content, no prepended context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_pieces = splitter.split_text(text)

    chunks: List[Dict[str, Any]] = []
    for idx, piece in enumerate(raw_pieces, 1):
        content = piece.strip()
        if not content:
            continue
        chunks.append({
            "chunk_id": f"{doc_id}-A-{idx:03d}",
            "doc_id": doc_id,
            "title": title,
            "source_url": source_url,
            "section": "",
            "strategy": "A",
            "text": content,
        })
    return chunks


def parse_sections(text: str) -> List[Dict[str, Any]]:
    """Parse document into structural sections while maintaining heading hierarchy."""
    lines = text.splitlines()
    raw_sections: List[Dict[str, Any]] = []

    # Stack holds tuples of (level, heading_text)
    stack: List[Tuple[int, str]] = []
    current_body_lines: List[str] = []
    current_section_path = ""

    for line in lines:
        h_match = match_heading(line)
        if h_match:
            # Save prior section if it had content or prior heading
            if current_body_lines or current_section_path:
                body_content = "\n".join(current_body_lines).strip()
                raw_sections.append({
                    "section_path": current_section_path,
                    "body": body_content,
                })
                current_body_lines = []

            level, heading_title = h_match
            # Pop stack elements with level >= new heading level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_title))
            current_section_path = " > ".join(h[1] for h in stack)
        else:
            current_body_lines.append(line)

    # Flush last section
    if current_body_lines or current_section_path:
        body_content = "\n".join(current_body_lines).strip()
        raw_sections.append({
            "section_path": current_section_path,
            "body": body_content,
        })

    return raw_sections


def merge_short_sections(raw_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge sections with body < 100 characters into subsequent (or prior) section."""
    if not raw_sections:
        return []

    merged: List[Dict[str, Any]] = []
    buffered_body = ""
    buffered_path = ""

    for i, sec in enumerate(raw_sections):
        body = sec["body"].strip()
        path = sec["section_path"]

        if buffered_body:
            body = (buffered_body + "\n\n" + body).strip()
            # Retain the most specific or combined path
            if not path:
                path = buffered_path
            buffered_body = ""
            buffered_path = ""

        is_last = (i == len(raw_sections) - 1)

        if len(body) < 100 and not is_last:
            # Buffer to merge into next section
            buffered_body = body
            buffered_path = path
        else:
            # If it's the last section and still < 100 chars, merge into previous if available
            if len(body) < 100 and is_last and merged:
                merged[-1]["body"] = (merged[-1]["body"] + "\n\n" + body).strip()
            else:
                merged.append({
                    "section_path": path,
                    "body": body,
                })

    # If any trailing buffered body remained
    if buffered_body:
        if merged:
            merged[-1]["body"] = (merged[-1]["body"] + "\n\n" + buffered_body).strip()
        else:
            merged.append({
                "section_path": buffered_path,
                "body": buffered_body,
            })

    return [s for s in merged if s["body"].strip()]


def chunk_strategy_b(doc_id: str, title: str, source_url: str, text: str) -> List[Dict[str, Any]]:
    """Strategy B: Hierarchical structure-aware chunking.

    1. Parse into sections with heading stack (Tiêu đề cha > Tiêu đề con).
    2. Body <= 1000 chars -> 1 chunk.
       Body > 1000 chars -> RecursiveCharacterTextSplitter(800, 80).
    3. Body < 100 chars -> Merged into adjacent sections.
    4. text starts with: "{title} > {section}\\n{content}".
    5. Fallback if no headings: 800/80 with title prefix.
    """
    raw_sections = parse_sections(text)

    # Check if any non-empty headings were discovered
    has_headings = any(s["section_path"].strip() for s in raw_sections)

    splitter_b = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []

    if not has_headings:
        # Fallback: split entire text with 800/80 and prepend title
        pieces = splitter_b.split_text(text)
        for idx, piece in enumerate(pieces, 1):
            content = piece.strip()
            if not content:
                continue
            chunk_text = f"{title}\n{content}"
            chunks.append({
                "chunk_id": f"{doc_id}-B-{idx:03d}",
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url,
                "section": "",
                "strategy": "B",
                "text": chunk_text,
            })
        return chunks

    # Merge short sections
    sections = merge_short_sections(raw_sections)

    chunk_idx = 1
    for sec in sections:
        sec_path = sec["section_path"].strip()
        body = sec["body"].strip()
        if not body:
            continue

        context_prefix = f"{title} > {sec_path}" if sec_path else title

        if len(body) <= 1000:
            chunk_text = f"{context_prefix}\n{body}"
            chunks.append({
                "chunk_id": f"{doc_id}-B-{chunk_idx:03d}",
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url,
                "section": sec_path,
                "strategy": "B",
                "text": chunk_text,
            })
            chunk_idx += 1
        else:
            pieces = splitter_b.split_text(body)
            for piece in pieces:
                p_clean = piece.strip()
                if not p_clean:
                    continue
                chunk_text = f"{context_prefix}\n{p_clean}"
                chunks.append({
                    "chunk_id": f"{doc_id}-B-{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "title": title,
                    "source_url": source_url,
                    "section": sec_path,
                    "strategy": "B",
                    "text": chunk_text,
                })
                chunk_idx += 1

    return chunks


def calculate_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistical summaries for a set of chunks."""
    if not chunks:
        return {
            "count": 0,
            "min_len": 0,
            "mean_len": 0,
            "median_len": 0,
            "max_len": 0,
            "short_count": 0,
            "doc_counts": {},
        }

    lengths = [len(c["text"]) for c in chunks]
    doc_counts: Dict[str, int] = {}
    for c in chunks:
        d = c["doc_id"]
        doc_counts[d] = doc_counts.get(d, 0) + 1

    return {
        "count": len(chunks),
        "min_len": min(lengths),
        "mean_len": round(statistics.mean(lengths), 1),
        "median_len": round(statistics.median(lengths), 1),
        "max_len": max(lengths),
        "short_count": sum(1 for l in lengths if l < 100),
        "doc_counts": doc_counts,
    }


def print_stats(strategy_name: str, stats: Dict[str, Any]) -> None:
    """Print aligned statistical summary table for a chunking strategy."""
    print("\n" + "=" * 80)
    print(f"THỐNG KÊ CHI TIẾT STRATEGY {strategy_name}")
    print("=" * 80)
    print(f"  - Tổng số chunks         : {stats['count']}")
    print(f"  - Độ dài nhỏ nhất (min)  : {stats['min_len']} ký tự")
    print(f"  - Độ dài trung bình (mean): {stats['mean_len']} ký tự")
    print(f"  - Độ dài trung vị (median): {stats['median_len']} ký tự")
    print(f"  - Độ dài lớn nhất (max)  : {stats['max_len']} ký tự")
    print(f"  - Số chunk < 100 ký tự   : {stats['short_count']}")
    print("  - Phân bổ chunk theo tài liệu:")
    for doc_id, count in stats["doc_counts"].items():
        print(f"      * {doc_id:<20}: {count} chunks")
    print("=" * 80)


def build_chunks() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute chunking for all documents in manifest for both strategies A and B."""
    manifest_path = DATA_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_chunks_a: List[Dict[str, Any]] = []
    all_chunks_b: List[Dict[str, Any]] = []

    for doc_id, info in manifest.items():
        doc_path = PROCESSED_DATA_DIR / f"{doc_id}.md"
        if not doc_path.exists():
            logger.warning("Không tìm thấy file processed cho %s tại %s", doc_id, doc_path)
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        title = info.get("title", "")
        url = info.get("url", "")

        chunks_a = chunk_strategy_a(doc_id, title, url, text)
        all_chunks_a.extend(chunks_a)

        chunks_b = chunk_strategy_b(doc_id, title, url, text)
        all_chunks_b.extend(chunks_b)

    # Save chunks_A.jsonl
    chunks_a_file = DATA_DIR / "chunks_A.jsonl"
    with open(chunks_a_file, "w", encoding="utf-8") as f:
        for c in all_chunks_a:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Save chunks_B.jsonl
    chunks_b_file = DATA_DIR / "chunks_B.jsonl"
    with open(chunks_b_file, "w", encoding="utf-8") as f:
        for c in all_chunks_b:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return all_chunks_a, all_chunks_b


def main() -> None:
    print("=" * 80)
    print("BẮT ĐẦU CHUNKING CORPUS (src/chunking.py)")
    print("=" * 80)

    chunks_a, chunks_b = build_chunks()

    stats_a = calculate_stats(chunks_a)
    stats_b = calculate_stats(chunks_b)

    print_stats("A (RecursiveCharacter 500/50)", stats_a)
    print_stats("B (Cấu trúc phân cấp + Breadcrumb)", stats_b)

    print(f"\n[OK] Đã ghi thành công {len(chunks_a)} chunks vào data/chunks_A.jsonl")
    print(f"[OK] Đã ghi thành công {len(chunks_b)} chunks vào data/chunks_B.jsonl\n")


if __name__ == "__main__":
    main()
