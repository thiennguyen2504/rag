"""CLI tool to draft candidate questions and verbatim evidence from candidate chunks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

from src.config import BASE_DIR, get_api_key, load_models_cfg
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
logger = logging.getLogger("draft_questions")

DATA_DIR = BASE_DIR / "data"

DRAFT_PROMPT_TEMPLATE = """Dưới đây là một đoạn trích từ tài liệu về ví điện tử MoMo:
---
{chunk_text}
---

Nhiệm vụ của bạn:
1. Đặt 01 câu hỏi tự nhiên bằng tiếng Việt (loại 'direct') mà một người dùng thực tế có thể thắc mắc, và câu hỏi này PHẢI được trả lời rõ ràng trong đoạn trích trên.
2. Cung cấp câu trả lời ngắn gọn, chính xác (gold_answer) dựa hoàn toàn trên đoạn trích.
3. Trích dẫn ĐÚNG NGUYÊN VĂN 01 câu hoặc vế câu ngắn (<= 200 ký tự) từ đoạn trích trên làm bằng chứng (gold_evidence).

Trả về ĐÚNG định dạng JSON duy nhất, không thêm giải thích hay markdown râu ria ngoài JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "..."
}}
"""


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code fences and whitespace from model response."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_draft_response(raw_text: str) -> Optional[Dict[str, str]]:
    """Parse JSON string safely from LLM output."""
    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "question" in data and "gold_answer" in data:
            return data
    except Exception:
        # Fallback regex extraction of json object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "question" in data:
                    return data
            except Exception:
                pass
    return None


@retry_with_backoff(max_retries=5, initial_delay=2.0)
def generate_single_draft(
    llm: ChatGoogleGenerativeAI,
    chunk: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Invoke LLM on a single chunk to draft question and evidence."""
    chunk_text = chunk.get("text", "").strip()
    prompt = DRAFT_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    response = llm.invoke(prompt)
    content = str(response.content)
    parsed = parse_draft_response(content)
    if not parsed:
        logger.warning("Không thể parse JSON từ phản hồi của model cho chunk %s", chunk.get("chunk_id"))
        return None
    return parsed


def load_candidate_chunks(min_chars: int = 150) -> List[Dict[str, Any]]:
    """Load chunks from data/chunks_B.jsonl and filter by minimum length."""
    chunks_b_path = DATA_DIR / "chunks_B.jsonl"
    if not chunks_b_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {chunks_b_path}. Vui lòng chạy 'python -m src.chunking' trước.")

    candidates = []
    with open(chunks_b_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                c = json.loads(line_str)
                if len(c.get("text", "").strip()) >= min_chars:
                    candidates.append(c)

    return candidates


def draft_questions_cli(
    num_questions: int = 10,
    seed: int = 42,
    concurrency: int = 3,
    out_file: str = "data/eval_draft.jsonl",
) -> None:
    """Sample candidate chunks, call LLM to draft questions, and output JSONL."""
    models_cfg = load_models_cfg()
    judge_model = models_cfg.judge_model
    api_key = get_api_key()

    candidates = load_candidate_chunks(min_chars=150)
    logger.info("Tìm thấy %d chunks hợp lệ (>= 150 ký tự) trong chunks_B.jsonl", len(candidates))

    if not candidates:
        logger.error("Không có chunk nào đạt yêu cầu độ dài.")
        return

    # Random sampling
    rng = random.Random(seed)
    sampled_count = min(num_questions, len(candidates))
    sampled_chunks = rng.sample(candidates, sampled_count)
    logger.info("Lấy mẫu ngẫu nhiên %d chunks với seed=%d", sampled_count, seed)

    llm = ChatGoogleGenerativeAI(
        model=judge_model,
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=4096,
    )

    drafted_items: List[Dict[str, Any]] = []

    def task_worker(item_idx: int, chk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        parsed = generate_single_draft(llm, chk)
        if not parsed:
            return None

        evidence_str = str(parsed.get("gold_evidence", "")).strip()
        # Verify evidence length constraint (<= 200 chars), truncate if slight overflow
        if len(evidence_str) > 200:
            evidence_str = evidence_str[:200]

        return {
            "id": f"draft_{item_idx:03d}",
            "question": str(parsed.get("question", "")).strip(),
            "type": "direct",
            "answerable": True,
            "gold_answer": str(parsed.get("gold_answer", "")).strip(),
            "gold_doc_ids": [chk["doc_id"]],
            "gold_evidence": [evidence_str] if evidence_str else [],
            "notes": "DRAFT - cần duyệt tay",
        }

    print(f"Đang sinh {sampled_count} câu hỏi nháp bằng judge_model='{judge_model}' (concurrency={concurrency})...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(task_worker, idx, chunk): idx
            for idx, chunk in enumerate(sampled_chunks, 1)
        }

        with tqdm(total=sampled_count, desc="Drafting questions") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                pbar.update(1)
                try:
                    res = future.result()
                    if res:
                        drafted_items.append(res)
                except Exception as exc:
                    logger.error("Lỗi khi sinh câu hỏi nháp cho chunk index %d: %s", idx, exc)

    # Sort by item ID
    drafted_items.sort(key=lambda x: x["id"])

    out_path = Path(out_file)
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in drafted_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[OK] Đã tạo thành công {len(drafted_items)}/{sampled_count} câu hỏi nháp vào {out_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft candidate eval questions using LLM.")
    parser.add_argument("--n", type=int, default=10, help="Số lượng câu hỏi muốn sinh nháp (mặc định: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--concurrency", type=int, default=3, help="Số lượng worker đồng thời (mặc định: 3)")
    parser.add_argument(
        "--out",
        type=str,
        default="data/eval_draft.jsonl",
        help="Đường dẫn file JSONL đầu ra (mặc định: data/eval_draft.jsonl)",
    )
    args = parser.parse_args()

    draft_questions_cli(
        num_questions=args.n,
        seed=args.seed,
        concurrency=args.concurrency,
        out_file=args.out,
    )


if __name__ == "__main__":
    main()

