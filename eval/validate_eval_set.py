"""Validation tool for evaluation sets enforcing data contracts, verbatim evidence, and quality metrics."""

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError
import rapidfuzz

from src.config import BASE_DIR

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_eval_set")

PROCESSED_DIR = BASE_DIR / "data" / "processed"

TARGET_CATEGORY_DISTRIBUTION = {
    "direct": 12,
    "multi": 6,
    "unanswerable": 8,
    "trap": 6,
    "casual": 4,
}


class EvalItem(BaseModel):
    """Schema representing an evaluation question record."""

    id: str = Field(..., description="ID duy nhất")
    question: str = Field(..., min_length=5, description="Câu hỏi người dùng")
    type: Literal["direct", "multi", "unanswerable", "trap", "casual"]
    answerable: bool
    gold_answer: str = Field(default="", description="Câu trả lời chuẩn")
    gold_doc_ids: List[str] = Field(default_factory=list)
    gold_evidence: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default="", description="Ghi chú")


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and converting to lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_processed_corpus(corpus_dir: Optional[Path] = None) -> Dict[str, str]:
    """Load and normalize all processed documents into memory."""
    if corpus_dir is None:
        corpus_dir = PROCESSED_DIR

    corpus: Dict[str, str] = {}
    if not corpus_dir.exists():
        return corpus

    for file_path in corpus_dir.glob("*.md"):
        doc_id = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            corpus[doc_id] = normalize_text(content)
    return corpus


def find_best_verbatim_match(
    evidence: str,
    doc_text_norm: str,
    original_doc_text: str,
) -> Tuple[float, str]:
    """Search for the best matching text window in document using RapidFuzz."""
    ev_norm = normalize_text(evidence)
    ev_len = len(ev_norm)
    if ev_len == 0 or len(doc_text_norm) == 0:
        return 0.0, ""

    window_size = max(ev_len, int(ev_len * 1.1))
    step = max(10, ev_len // 4)

    best_score = 0.0
    best_segment = ""

    for i in range(0, len(doc_text_norm) - ev_len + 1, step):
        chunk = doc_text_norm[i : i + window_size]
        score = rapidfuzz.fuzz.ratio(ev_norm, chunk)
        if score > best_score:
            best_score = score
            best_segment = original_doc_text[i : i + window_size]
            if best_score >= 95.0:
                break

    return best_score, best_segment.replace("\n", " ").strip()


def validate_eval_set(
    eval_file_path: Path,
    corpus_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Validate eval set file and return errors, warnings, and distribution stats."""
    if not eval_file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {eval_file_path}")

    corpus = load_processed_corpus(corpus_dir)
    original_docs: Dict[str, str] = {}
    actual_dir = corpus_dir or PROCESSED_DIR
    if actual_dir.exists():
        for fp in actual_dir.glob("*.md"):
            with open(fp, "r", encoding="utf-8") as f:
                original_docs[fp.stem] = f.read()

    errors: List[str] = []
    warnings: List[str] = []

    seen_ids: Set[str] = set()
    questions_list: List[Tuple[str, str]] = []
    type_counts: Dict[str, int] = {k: 0 for k in TARGET_CATEGORY_DISTRIBUTION}
    doc_counts: Dict[str, int] = {}

    with open(eval_file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                raw_item = json.loads(line_str)
            except json.JSONDecodeError as exc:
                errors.append(f"Dòng {line_idx}: Định dạng JSON không hợp lệ ({exc})")
                continue

            try:
                item = EvalItem(**raw_item)
            except ValidationError as exc:
                errors.append(f"Dòng {line_idx} [{raw_item.get('id', 'N/A')}]: Lỗi schema Pydantic: {exc}")
                continue

            # Check duplicate ID
            if item.id in seen_ids:
                errors.append(f"Dòng {line_idx}: Trùng lặp ID '{item.id}'")
            seen_ids.add(item.id)

            # Record question for near-duplicate check
            questions_list.append((item.id, item.question))

            # Count type
            type_counts[item.type] = type_counts.get(item.type, 0) + 1

            # Check unanswerable constraints
            if not item.answerable:
                if item.gold_doc_ids:
                    errors.append(
                        f"Câu hỏi [{item.id}]: answerable=false nhưng gold_doc_ids không rỗng: {item.gold_doc_ids}"
                    )
                if item.gold_evidence:
                    errors.append(
                        f"Câu hỏi [{item.id}]: answerable=false nhưng gold_evidence không rỗng: {item.gold_evidence}"
                    )
            else:
                # Answerable=true requires doc_ids and evidence
                if not item.gold_doc_ids:
                    errors.append(f"Câu hỏi [{item.id}]: answerable=true nhưng gold_doc_ids rỗng.")
                if not item.gold_evidence:
                    errors.append(f"Câu hỏi [{item.id}]: answerable=true nhưng gold_evidence rỗng.")

            # Record doc distribution
            for doc in item.gold_doc_ids:
                doc_counts[doc] = doc_counts.get(doc, 0) + 1

            # Check doc existence in processed/
            for doc in item.gold_doc_ids:
                if doc not in corpus:
                    errors.append(
                        f"Câu hỏi [{item.id}]: doc_id '{doc}' không tồn tại trong corpus data/processed/"
                    )

            # Check verbatim evidence against docs
            for ev_idx, evidence in enumerate(item.gold_evidence, 1):
                ev_norm = normalize_text(evidence)
                if len(evidence) > 200:
                    warnings.append(
                        f"Câu hỏi [{item.id}]: Đoạn trích evidence #{ev_idx} dài {len(evidence)} ký tự (> 200 ký tự)."
                    )

                found = False
                for doc in item.gold_doc_ids:
                    doc_norm = corpus.get(doc, "")
                    if ev_norm in doc_norm:
                        found = True
                        break

                if not found and item.gold_doc_ids:
                    # Provide suggestion using RapidFuzz
                    best_doc = item.gold_doc_ids[0]
                    score, suggestion = find_best_verbatim_match(
                        evidence,
                        corpus.get(best_doc, ""),
                        original_docs.get(best_doc, ""),
                    )
                    err_msg = (
                        f"Câu hỏi [{item.id}]: Đoạn trích evidence #{ev_idx} KHÔNG tìm thấy nguyên văn trong {item.gold_doc_ids}.\n"
                        f'       Evidence: "{evidence}"'
                    )
                    if score >= 60.0:
                        err_msg += f'\n       Gợi ý đoạn gần giống nhất trong \'{best_doc}\' (độ khớp {score:.1f}%): "{suggestion[:150]}"'
                    errors.append(err_msg)

    # Check for near duplicate questions (RapidFuzz >= 85%)
    for i in range(len(questions_list)):
        qid_1, q_text_1 = questions_list[i]
        norm_1 = normalize_text(q_text_1)
        for j in range(i + 1, len(questions_list)):
            qid_2, q_text_2 = questions_list[j]
            norm_2 = normalize_text(q_text_2)
            sim = rapidfuzz.fuzz.ratio(norm_1, norm_2)
            if sim >= 85.0:
                warnings.append(
                    f"Cảnh báo câu hỏi gần trùng lặp ({sim:.1f}% tương đồng): [{qid_1}] vs [{qid_2}]: '{q_text_1}' <-> '{q_text_2}'"
                )

    stats = {
        "total_questions": len(questions_list),
        "type_counts": type_counts,
        "doc_counts": doc_counts,
    }

    return errors, warnings, stats


def print_validation_report(
    eval_file_path: Path,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """Print aligned, readable terminal report."""
    print("=" * 90)
    print(f"KẾT QUẢ KIỂM TRA TẬP ĐÁNH GIÁ (eval_set): {eval_file_path}")
    print("=" * 90)

    if errors:
        print(f"\n[LỖI NGHIÊM TRỌNG] Phát hiện {len(errors)} lỗi:")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx:02d}. {err}")
    else:
        print("\n[OK] Không có lỗi nghiêm trọng nào. Tập dữ liệu tuân thủ đúng hợp đồng!")

    if warnings:
        print(f"\n[CẢNH BÁO] Có {len(warnings)} cảnh báo chất lượng:")
        for idx, warn in enumerate(warnings, 1):
            print(f"  {idx:02d}. {warn}")
    else:
        print("\n[OK] Không có cảnh báo chất lượng nào.")

    print("\n" + "-" * 90)
    print(f"THỐNG KÊ PHÂN BỔ LOẠI CÂU HỎI (Tổng số câu: {stats['total_questions']})")
    print("-" * 90)
    print(f"{'Loại câu hỏi':<18} | {'Thực tế':<10} | {'Mục tiêu':<10} | {'Trạng thái'}")
    print(f"{'-'*19}+{'-'*12}+{'-'*12}+{'-'*16}")

    for cat, target in TARGET_CATEGORY_DISTRIBUTION.items():
        actual = stats["type_counts"].get(cat, 0)
        diff = actual - target
        if diff >= 0:
            status = f"Đạt (+{diff})" if diff > 0 else "Đạt"
        else:
            status = f"Thiếu {abs(diff)} câu"
        print(f"{cat:<18} | {actual:<10} | {target:<10} | {status}")

    print("\n" + "-" * 90)
    print("PHÂN BỔ THEO TÀI LIỆU (gold_doc_ids):")
    for doc, cnt in sorted(stats["doc_counts"].items(), key=lambda x: -x[1]):
        print(f"  - {doc:<25}: {cnt} câu hỏi")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate eval_set dataset contract and quality.")
    parser.add_argument(
        "--file",
        dest="file_path",
        type=str,
        default="data/eval_set.jsonl",
        help="Đường dẫn file eval_set (mặc định: data/eval_set.jsonl)",
    )
    args = parser.parse_args()

    eval_path = Path(args.file_path)
    if not eval_path.is_absolute():
        eval_path = BASE_DIR / eval_path

    try:
        errors, warnings, stats = validate_eval_set(eval_path)
        print_validation_report(eval_path, errors, warnings, stats)
        if errors:
            sys.exit(1)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[LỖI]: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

