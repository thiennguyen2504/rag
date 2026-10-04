"""Evaluation generation runner: executes pipeline on eval_set and logs results incrementally."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.config import BASE_DIR
from src.pipeline import ask

# Configure UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval")

LOGS_DIR = BASE_DIR / "logs"
RUNS_LOG_PATH = LOGS_DIR / "runs.jsonl"
_FILE_LOCK = threading.Lock()


def load_eval_set(eval_set_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tập đánh giá: {eval_set_path}")

    items: List[Dict[str, Any]] = []
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                items.append(item)
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ qua dòng %d do lỗi JSON: %s", line_num, exc)

    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def load_logged_question_ids(run_id: str) -> Set[str]:
    """Retrieve set of question IDs already logged for this specific run_id."""
    if not RUNS_LOG_PATH.exists():
        return set()

    logged_ids: Set[str] = set()
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entry = json.loads(line_str)
                if entry.get("run_id") == run_id:
                    qid = entry.get("question_id")
                    if qid:
                        logged_ids.add(qid)
            except Exception:
                continue
    return logged_ids


def log_run_entry(entry: Dict[str, Any]) -> None:
    """Safely append a single JSON run entry to logs/runs.jsonl with a thread lock."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _FILE_LOCK:
        with open(RUNS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()


def process_question(
    item: Dict[str, Any],
    config_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """Execute pipeline for a single question and return the formatted log entry."""
    question_id = str(item.get("id", "unknown"))
    question_text = str(item.get("question", "")).strip()

    try:
        result = ask(question=question_text, config_id=config_id)
    except Exception as exc:
        logger.error("Lỗi ngoại lệ khi chạy câu hỏi [%s]: %s", question_id, exc)
        result = {
            "question": question_text,
            "config_id": config_id,
            "answer": "",
            "sources": [],
            "final_prompt": "",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
            "latency_ms": {"retrieval": 0.0, "generation": 0.0, "total": 0.0},
            "error": str(exc),
        }

    # Transform sources to compact retrieval log contract
    retrieved = [
        {
            "chunk_id": s.get("chunk_id", ""),
            "doc_id": s.get("doc_id", ""),
            "score": round(float(s.get("score", 0.0)), 4),
        }
        for s in result.get("sources", [])
    ]

    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "question_id": question_id,
        "question": question_text,
        "retrieved": retrieved,
        "final_prompt": result.get("final_prompt", ""),
        "answer": result.get("answer", ""),
        "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        "cost_usd": round(float(result.get("cost_usd", 0.0)), 6),
        "latency_ms": result.get("latency_ms", {"retrieval": 0.0, "generation": 0.0, "total": 0.0}),
        "error": result.get("error"),
    }

    # Immediately write to log file
    log_run_entry(log_entry)
    return log_entry


def run_evaluation(
    config_id: str,
    eval_set_path: Path,
    run_id: Optional[str] = None,
    concurrency: int = 3,
    limit: Optional[int] = None,
    resume: bool = False,
) -> str:
    """Orchestrate evaluation run over eval set."""
    if not run_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{config_id}_{timestamp_str}"

    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU CHẠY ĐÁNH GIÁ (RUN EVALUATION)")
    print(f"Run ID      : {run_id}")
    print(f"Cấu hình    : {config_id}")
    print(f"Tập đánh giá: {eval_set_path}")
    print(f"Concurrency : {concurrency}")
    print("=" * 80 + "\n")

    items = load_eval_set(eval_set_path, limit=limit)
    if not items:
        print("Không có câu hỏi nào cần đánh giá trong file.")
        return run_id

    already_done_ids: Set[str] = set()
    if resume:
        already_done_ids = load_logged_question_ids(run_id)
        if already_done_ids:
            print(f"[RESUME] Đã tìm thấy {len(already_done_ids)} câu đã hoàn thành trước đó với run_id='{run_id}'.")

    pending_items = [it for it in items if str(it.get("id")) not in already_done_ids]
    total_pending = len(pending_items)

    print(f"Tổng số câu: {len(items)} | Đã chạy: {len(already_done_ids)} | Cần xử lý: {total_pending}")
    if total_pending == 0:
        print("[XONG] Toàn bộ câu hỏi đã được xử lý trước đó.")
        print(f"\nRUN ID: {run_id}\n")
        return run_id

    start_time = time.perf_counter()
    error_count = 0
    total_cost_usd = 0.0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_item = {
            executor.submit(process_question, it, config_id, run_id): it
            for it in pending_items
        }

        with tqdm(total=total_pending, desc=f"Evaluating ({config_id})") as pbar:
            for future in as_completed(future_to_item):
                pbar.update(1)
                try:
                    entry = future.result()
                    if entry.get("error"):
                        error_count += 1
                    total_cost_usd += entry.get("cost_usd", 0.0)
                except Exception as exc:
                    error_count += 1
                    logger.error("Lỗi không mong muốn trong luồng xử lý: %s", exc)

    elapsed_time = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("TỔNG KẾT ĐỢT CHẠY:")
    print(f"  - Run ID             : {run_id}")
    print(f"  - Số câu đã xử lý    : {total_pending}")
    print(f"  - Số câu lỗi (error) : {error_count}")
    print(f"  - Tổng chi phí (USD) : ${total_cost_usd:.6f}")
    print(f"  - Thời gian thực thi : {elapsed_time:.2f} giây")
    print(f"  - File log ghi nhận  : {RUNS_LOG_PATH}")
    print("=" * 80 + "\n")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline on eval_set and log results.")
    parser.add_argument("--config", type=str, required=True, help="ID cấu hình ('A' hoặc 'B')")
    parser.add_argument(
        "--eval-set",
        dest="eval_set",
        type=str,
        default="data/eval_set.jsonl",
        help="Đường dẫn file eval_set (mặc định: data/eval_set.jsonl)",
    )
    parser.add_argument("--run-id", dest="run_id", type=str, default=None, help="Tùy chọn gán ID cho run")
    parser.add_argument("--concurrency", type=int, default=3, help="Số luồng đồng thời (mặc định: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu hỏi chạy")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua các question_id đã chạy trong cùng run_id")

    args = parser.parse_args()
    eval_set_path = Path(args.eval_set)
    if not eval_set_path.is_absolute():
        eval_set_path = BASE_DIR / eval_set_path

    run_evaluation(
        config_id=args.config,
        eval_set_path=eval_set_path,
        run_id=args.run_id,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

