"""Check retrieval performance on Phase 1 benchmark queries across Strategy A and B."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    ANALYSIS_DIR,
    CHROMA_DIR,
    DATA_DIR,
    get_api_key,
    load_models_config,
)
from src.ingest import load_ingest_manifest
from src.llm_utils import retry_with_backoff

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def query_chroma_with_score(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> List[Tuple[Any, float]]:
    """Query Chroma and return (Document, similarity_score).

    In Chroma with hnsw:space=cosine:
    Chroma returns cosine distance in [0, 2].
    Cosine similarity = 1.0 - distance.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    converted: List[Tuple[Any, float]] = []
    for doc, dist in results:
        similarity = 1.0 - float(dist)
        converted.append((doc, similarity))
    return converted


def load_queries(file_path: Path) -> List[str]:
    """Load query strings from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {file_path}")
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                queries.append(q)
    return queries


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format an ASCII table."""
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


def run_retrieval_check(queries_path: Path, k: int = 5) -> None:
    api_key = get_api_key()
    models_cfg = load_models_config()
    embedding_model = models_cfg["embedding_model"]

    manifest = load_ingest_manifest()
    if manifest and manifest.get("embedding_model") != embedding_model:
        print(
            f"[CẢNH BÁO] Ingest manifest dùng model '{manifest.get('embedding_model')}' "
            f"nhưng models.yaml đang cấu hình '{embedding_model}'.",
            file=sys.stderr,
        )

    queries = load_queries(queries_path)
    if not queries:
        print(f"[LỖI] File câu hỏi {queries_path} không có câu hỏi nào.", file=sys.stderr)
        sys.exit(1)

    print("=" * 90)
    print(f"KIỂM TRA TRUY XUẤT CORPUS MOMO (A vs B) - k = {k}")
    print(f"Embedding model: {embedding_model}")
    print("=" * 90)

    embedding_function = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key,
    )

    store_a = Chroma(
        collection_name="momo_A",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store_b = Chroma(
        collection_name="momo_B",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )

    eval_data: List[Dict[str, Any]] = []

    for idx, query in enumerate(queries, 1):
        # By contract: first 6 are answerable, last 4 are unanswerable
        category = "Có đáp án" if idx <= 6 else "Không có đáp án"

        res_a = query_chroma_with_score(store_a, query, k=k)
        res_b = query_chroma_with_score(store_b, query, k=k)

        top1_a = res_a[0][1] if res_a else 0.0
        top1_b = res_b[0][1] if res_b else 0.0

        eval_data.append({
            "idx": idx,
            "query": query,
            "category": category,
            "top1_a": top1_a,
            "top1_b": top1_b,
            "res_a": res_a,
            "res_b": res_b,
        })

        print(f"\n[{idx:02d}] ({category}) Câu hỏi: {query}")
        print("  - STRATEGY A (momo_A):")
        for rank, (doc, sim) in enumerate(res_a, 1):
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            sec = doc.metadata.get("section", "")
            preview = doc.page_content.replace("\n", " ")[:120]
            sec_info = f" | Sec: {sec}" if sec else ""
            print(f"     #{rank} [sim: {sim:.4f}] {chunk_id}{sec_info} -> {preview}...")

        print("  - STRATEGY B (momo_B):")
        for rank, (doc, sim) in enumerate(res_b, 1):
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            sec = doc.metadata.get("section", "")
            preview = doc.page_content.replace("\n", " ")[:120]
            sec_info = f" | Sec: {sec}" if sec else ""
            print(f"     #{rank} [sim: {sim:.4f}] {chunk_id}{sec_info} -> {preview}...")

    # Compute Group Averages
    ans_a = [d["top1_a"] for d in eval_data if d["category"] == "Có đáp án"]
    ans_b = [d["top1_b"] for d in eval_data if d["category"] == "Có đáp án"]
    unans_a = [d["top1_a"] for d in eval_data if d["category"] == "Không có đáp án"]
    unans_b = [d["top1_b"] for d in eval_data if d["category"] == "Không có đáp án"]

    avg_ans_a = sum(ans_a) / len(ans_a) if ans_a else 0.0
    avg_ans_b = sum(ans_b) / len(ans_b) if ans_b else 0.0
    avg_unans_a = sum(unans_a) / len(unans_a) if unans_a else 0.0
    avg_unans_b = sum(unans_b) / len(unans_b) if unans_b else 0.0

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP TOP-1 SIMILARITY")
    print("=" * 90)
    summary_headers = ["STT", "Nhóm", "Top-1 Sim (A)", "Top-1 Sim (B)", "Câu hỏi"]
    summary_rows = [
        [
            f"Q{d['idx']:02d}",
            d["category"],
            f"{d['top1_a']:.4f}",
            f"{d['top1_b']:.4f}",
            d["query"][:50] + ("..." if len(d["query"]) > 50 else ""),
        ]
        for d in eval_data
    ]
    print(format_table(summary_headers, summary_rows))

    print("\n" + "-" * 90)
    print(f"TRUNG BÌNH TOP-1 NHÓM 'CÓ ĐÁP ÁN'      : A = {avg_ans_a:.4f} | B = {avg_ans_b:.4f}")
    print(f"TRUNG BÌNH TOP-1 NHÓM 'KHÔNG CÓ ĐÁP ÁN': A = {avg_unans_a:.4f} | B = {avg_unans_b:.4f}")
    print(f"ĐỘ LỆCH (GAP = Ans - Unans)            : A = {(avg_ans_a - avg_unans_a):.4f} | B = {(avg_ans_b - avg_unans_b):.4f}")
    print("=" * 90)

    # Save to analysis/phase1_retrieval_check.md
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = ANALYSIS_DIR / "phase1_retrieval_check.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ TRUY XUẤT RETRIEVAL PHASE 1 (A vs B)\n\n")
        f.write(f"- **Thời gian thực hiện**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"- **Embedding Model**: `{embedding_model}`\n")
        f.write(f"- **Số lượng câu truy vấn kiểm tra**: {len(queries)} (6 câu 'Có đáp án', 4 câu 'Không có đáp án')\n")
        f.write(f"- **Top-k**: {k}\n\n")

        f.write("## 1. Bảng Tổng Hợp Top-1 Cosine Similarity\n\n")
        f.write("| STT | Nhóm | Câu hỏi | Top-1 Sim (A) | Top-1 Sim (B) | Chênh lệch (B - A) |\n")
        f.write("|---|---|---|:---:|:---:|:---:|\n")
        for d in eval_data:
            diff = d["top1_b"] - d["top1_a"]
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            f.write(f"| Q{d['idx']:02d} | {d['category']} | {d['query']} | {d['top1_a']:.4f} | {d['top1_b']:.4f} | {diff_str} |\n")

        f.write("\n## 2. Thống Kê So Sánh Theo Nhóm Câu Hỏi\n\n")
        f.write("| Chỉ số | Strategy A (Naive 500/50) | Strategy B (Phân cấp + Breadcrumb) |\n")
        f.write("|---|:---:|:---:|\n")
        f.write(f"| **Trung bình Top-1 (Có đáp án)** | `{avg_ans_a:.4f}` | `{avg_ans_b:.4f}` |\n")
        f.write(f"| **Trung bình Top-1 (Không có đáp án)** | `{avg_unans_a:.4f}` | `{avg_unans_b:.4f}` |\n")
        f.write(f"| **Khoảng cách phân tách (Gap)** | `{avg_ans_a - avg_unans_a:.4f}` | `{avg_ans_b - avg_unans_b:.4f}` |\n\n")

        f.write("## 3. Nhận Xét Tự Động & Gợi Ý Cấu Hình Phase 2\n\n")
        f.write("- **Khả năng phân tách (Discrimination Gap)**:\n")
        if (avg_ans_b - avg_unans_b) >= (avg_ans_a - avg_ans_a):
            f.write("  * Strategy B thể hiện khoảng cách phân tách tốt hơn hoặc tương đương giữa câu hỏi hợp lệ và câu hỏi ngoại phạm vi.\n")
        f.write(f"  * Điểm trung bình câu 'Có đáp án' đạt `{avg_ans_b:.4f}` (B) và `{avg_ans_a:.4f}` (A).\n")
        f.write(f"  * Điểm trung bình câu 'Không có đáp án' là `{avg_unans_b:.4f}` (B) và `{avg_unans_a:.4f}` (A).\n\n")

        rec_threshold_a = round((avg_ans_a + avg_unans_a) / 2, 2)
        rec_threshold_b = round((avg_ans_b + avg_unans_b) / 2, 2)
        f.write("- **Gợi ý ngưỡng lọc `similarity_threshold` cho Phase 2**:\n")
        f.write(f"  * Đối với Strategy A: Khuyến nghị thử nghiệm ngưỡng `similarity_threshold: {rec_threshold_a}`.\n")
        f.write(f"  * Đối với Strategy B: Khuyến nghị thử nghiệm ngưỡng `similarity_threshold: {rec_threshold_b}`.\n")
        f.write("  * Ngưỡng này giúp pipeline kích hoạt tính năng `short_circuit_on_empty_context` để từ chối các câu hỏi nằm ngoài tài liệu mà không tốn chi phí gọi LLM sinh văn bản.\n\n")

        f.write("## 4. Chi Tiết Top-k Retrieved Chunks\n\n")
        for d in eval_data:
            f.write(f"### Q{d['idx']:02d}: {d['query']} (`{d['category']}`)\n\n")
            f.write("#### Strategy A:\n")
            for rank, (doc, sim) in enumerate(d["res_a"], 1):
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                sec = doc.metadata.get("section", "")
                text_snippet = doc.page_content.replace("\n", " ")[:200]
                f.write(f"{rank}. **[{chunk_id}]** (Sim: `{sim:.4f}`) {text_snippet}...\n")

            f.write("\n#### Strategy B:\n")
            for rank, (doc, sim) in enumerate(d["res_b"], 1):
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                sec = doc.metadata.get("section", "")
                text_snippet = doc.page_content.replace("\n", " ")[:200]
                sec_md = f"*(Mục: {sec})* " if sec else ""
                f.write(f"{rank}. **[{chunk_id}]** (Sim: `{sim:.4f}`) {sec_md}{text_snippet}...\n")
            f.write("\n---\n\n")

    print(f"\n[OK] Đã ghi chi tiết báo cáo đánh giá vào: {report_file}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 retrieval check.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DATA_DIR / "phase1_queries.txt",
        help="Đường dẫn file chứa danh sách câu hỏi kiểm tra",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Số lượng chunks truy xuất top-k",
    )
    args = parser.parse_args()

    run_retrieval_check(args.queries, k=args.k)


if __name__ == "__main__":
    main()
