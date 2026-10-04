"""LLM-as-a-Judge module for evaluating RAG answers against gold data and retrieved context."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

from src.config import BASE_DIR, get_api_key, load_models_cfg, load_pipeline_cfg
from src.llm_utils import retry_with_backoff

# Configure UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("judge")

LOGS_DIR = BASE_DIR / "logs"
RUNS_LOG_PATH = LOGS_DIR / "runs.jsonl"
RESULTS_DIR = BASE_DIR / "results"
PROMPTS_DIR = BASE_DIR / "prompts"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge.txt"

_WRITE_LOCK = threading.Lock()


def load_chunks_lookup() -> Dict[str, Dict[str, Any]]:
    """Load all chunks from chunks_A.jsonl and chunks_B.jsonl into a dictionary keyed by chunk_id."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for filename in ["chunks_A.jsonl", "chunks_B.jsonl"]:
        path = BASE_DIR / "data" / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    c = json.loads(line_str)
                    cid = c.get("chunk_id")
                    if cid:
                        lookup[cid] = c
                except Exception:
                    continue
    return lookup


def clean_json_text(raw_text: str) -> str:
    """Extract and clean JSON string from LLM output."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        text = obj_match.group(0)

    return text.strip()


def parse_judge_output(raw_text: str) -> Dict[str, Any]:
    """Parse and validate judge JSON output according to contract."""
    cleaned = clean_json_text(raw_text)
    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError("Kết quả từ judge không phải dictionary JSON.")

    correctness = str(data.get("correctness", "")).lower().strip()
    if correctness not in {"correct", "partial", "incorrect"}:
        raise ValueError(f"Giá trị correctness không hợp lệ: '{correctness}'")

    faithful_raw = data.get("faithful")
    if not isinstance(faithful_raw, bool):
        if str(faithful_raw).lower() in {"true", "1"}:
            faithful = True
        elif str(faithful_raw).lower() in {"false", "0"}:
            faithful = False
        else:
            raise ValueError(f"Giá trị faithful không hợp lệ: '{faithful_raw}'")
    else:
        faithful = faithful_raw

    refused_raw = data.get("refused")
    if not isinstance(refused_raw, bool):
        if str(refused_raw).lower() in {"true", "1"}:
            refused = True
        elif str(refused_raw).lower() in {"false", "0"}:
            refused = False
        else:
            raise ValueError(f"Giá trị refused không hợp lệ: '{refused_raw}'")
    else:
        refused = refused_raw

    reason = str(data.get("reason", "")).strip()

    return {
        "correctness": correctness,
        "faithful": faithful,
        "refused": refused,
        "reason": reason,
    }


def load_runs_for_run_id(run_id: str) -> List[Dict[str, Any]]:
    """Read logs/runs.jsonl and filter entries for the given run_id."""
    if not RUNS_LOG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhật ký chạy: {RUNS_LOG_PATH}")

    entries = []
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entry = json.loads(line_str)
                if entry.get("run_id") == run_id:
                    entries.append(entry)
            except Exception:
                continue
    return entries


def load_eval_set_map(eval_set_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load eval set file into a mapping: question_id -> eval_item."""
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tập đánh giá: {eval_set_path}")

    mapping = {}
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                qid = str(item.get("id"))
                mapping[qid] = item
            except Exception:
                continue
    return mapping


def load_existing_judgements(judgements_path: Path) -> Set[str]:
    """Load set of already judged question IDs from judgements.jsonl."""
    if not judgements_path.exists():
        return set()

    judged_ids: Set[str] = set()
    with open(judgements_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                qid = data.get("question_id")
                if qid:
                    judged_ids.add(str(qid))
            except Exception:
                continue
    return judged_ids


def append_judgement_entry(judgements_path: Path, entry: Dict[str, Any]) -> None:
    """Safely append a single judgement line to judgements.jsonl."""
    judgements_path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with open(judgements_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()


def format_retrieved_context(
    retrieved_list: List[Dict[str, Any]],
    chunks_lookup: Dict[str, Dict[str, Any]],
) -> str:
    """Format retrieved chunks into numbered context blocks for judge prompt."""
    if not retrieved_list:
        return "(Không có đoạn trích nào được truy xuất)"

    blocks = []
    for idx, r in enumerate(retrieved_list, 1):
        cid = r.get("chunk_id", "")
        chunk_data = chunks_lookup.get(cid, {})
        title = chunk_data.get("title", r.get("doc_id", ""))
        section = chunk_data.get("section", "")
        text = chunk_data.get("text", "").strip()

        header = f"{title} - {section}" if section else title
        if not text:
            text = f"Chunk ID: {cid}"

        blocks.append(f"[{idx}] ({header})\n{text}")

    return "\n\n".join(blocks)


@retry_with_backoff(max_retries=5, initial_delay=2.0)
def call_judge_llm(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    """Invoke judge LLM with exponential backoff and retry."""
    response = llm.invoke(prompt)
    return str(response.content)


def evaluate_single_run_item(
    run_entry: Dict[str, Any],
    eval_item: Optional[Dict[str, Any]],
    prompt_template: str,
    llm: ChatGoogleGenerativeAI,
    chunks_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Judge a single question run item."""
    run_id = run_entry["run_id"]
    qid = run_entry["question_id"]
    question = run_entry.get("question", "")
    system_answer = run_entry.get("answer", "")
    error = run_entry.get("error")

    if error:
        return {
            "run_id": run_id,
            "question_id": qid,
            "correctness": "incorrect",
            "faithful": False,
            "refused": False,
            "reason": f"Hệ thống bị lỗi thực thi: {error}",
        }

    is_answerable = True
    gold_answer = "KHÔNG CÓ ĐÁP ÁN TRONG TÀI LIỆU"
    if eval_item:
        is_answerable = eval_item.get("answerable", True)
        if is_answerable:
            gold_answer = eval_item.get("gold_answer", "")
        else:
            gold_answer = "KHÔNG CÓ ĐÁP ÁN TRONG TÀI LIỆU (Câu hỏi unanswerable, hệ thống bắt buộc phải từ chối lịch sự)."

    retrieved_context_str = format_retrieved_context(run_entry.get("retrieved", []), chunks_lookup)

    prompt = prompt_template.format(
        question=question,
        gold_answer=gold_answer,
        retrieved_chunks=retrieved_context_str,
        answer=system_answer,
    )

    for attempt in range(3):
        raw_response = call_judge_llm(llm, prompt)
        try:
            parsed = parse_judge_output(raw_response)
            break
        except Exception as exc:
            logger.warning("Thử lại parse judge cho câu [%s] lần %d: %s", qid, attempt + 1, exc)
            if attempt == 2:
                parsed = {
                    "correctness": "incorrect",
                    "faithful": False,
                    "refused": False,
                    "reason": f"Lỗi parse JSON kết quả từ Judge: {exc}",
                }

    if not is_answerable:
        if not parsed["refused"] and parsed["correctness"] == "correct":
            parsed["correctness"] = "incorrect"
            parsed["reason"] = (
                parsed["reason"]
                + " [Hợp đồng: Câu unanswerable đưa ra câu trả lời thay vì từ chối nên bị tính incorrect]"
            ).strip()

    return {
        "run_id": run_id,
        "question_id": qid,
        "correctness": parsed["correctness"],
        "faithful": parsed["faithful"],
        "refused": parsed["refused"],
        "reason": parsed["reason"],
    }


def judge_run(
    run_id: str,
    concurrency: int = 3,
    eval_set_path: Optional[Path] = None,
) -> Path:
    """Execute LLM-as-a-Judge for all entries belonging to run_id."""
    if eval_set_path is None:
        eval_set_path = BASE_DIR / "data" / "eval_set.jsonl"

    run_entries = load_runs_for_run_id(run_id)
    if not run_entries:
        raise ValueError(f"Không tìm thấy lượt chạy nào cho run_id='{run_id}' trong {RUNS_LOG_PATH}")

    config_id = run_entries[0].get("config_id", "A")

    models_cfg = load_models_cfg()
    judge_model_name = models_cfg.judge_model

    pipeline_cfg = load_pipeline_cfg(config_id)
    generator_model_name = pipeline_cfg.llm.model or models_cfg.generator_model

    if judge_model_name == generator_model_name:
        print("\n" + "!" * 80)
        print("  CẢNH BÁO QUAN TRỌNG (LLM-as-a-Judge):")
        print(f"  judge_model ('{judge_model_name}') TRÙNG VỚI generator_model ('{generator_model_name}').")
        print("  Đánh giá có thể có độ thiên vị (bias) cao đối với câu trả lời của chính mô hình.")
        print("!" * 80 + "\n")

    if not JUDGE_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file prompt judge: {JUDGE_PROMPT_PATH}")
    with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    chunks_lookup = load_chunks_lookup()
    eval_set_map = load_eval_set_map(eval_set_path)

    run_results_dir = RESULTS_DIR / run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)
    judgements_path = run_results_dir / "judgements.jsonl"

    already_judged_ids = load_existing_judgements(judgements_path)
    pending_entries = [r for r in run_entries if str(r.get("question_id")) not in already_judged_ids]

    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU CHẤM ĐIỂM (LLM-as-a-Judge)")
    print(f"Run ID          : {run_id}")
    print(f"Cấu hình        : {config_id}")
    print(f"Judge Model     : {judge_model_name} (temperature=0.0)")
    print(f"Tổng số câu     : {len(run_entries)} (Đã chấm: {len(already_judged_ids)} | Cần chấm: {len(pending_entries)})")
    print(f"File kết quả    : {judgements_path}")
    print("=" * 80 + "\n")

    if not pending_entries:
        print("[XONG] Tất cả câu hỏi của run này đã được chấm trước đó.")
        return judgements_path

    api_key = get_api_key()
    llm = ChatGoogleGenerativeAI(
        model=judge_model_name,
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=2048,
    )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_entry = {
            executor.submit(
                evaluate_single_run_item,
                entry,
                eval_set_map.get(str(entry.get("question_id"))),
                prompt_template,
                llm,
                chunks_lookup,
            ): entry
            for entry in pending_entries
        }

        with tqdm(total=len(pending_entries), desc="Judging answers") as pbar:
            for future in as_completed(future_to_entry):
                pbar.update(1)
                try:
                    result = future.result()
                    if result:
                        append_judgement_entry(judgements_path, result)
                except Exception as exc:
                    logger.error("Lỗi khi chấm câu hỏi: %s", exc)

    print(f"\n[HOÀN TẤT] Đã lưu kết quả chấm vào {judgements_path}\n")
    return judgements_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for evaluation runs.")
    parser.add_argument("--run-id", dest="run_id", type=str, required=True, help="Run ID cần chấm điểm")
    parser.add_argument("--concurrency", type=int, default=3, help="Số luồng song song (mặc định: 3)")
    parser.add_argument(
        "--eval-set",
        dest="eval_set",
        type=str,
        default="data/eval_set.jsonl",
        help="Đường dẫn file eval_set (mặc định: data/eval_set.jsonl)",
    )

    args = parser.parse_args()
    eval_set_path = Path(args.eval_set)
    if not eval_set_path.is_absolute():
        eval_set_path = BASE_DIR / eval_set_path

    judge_run(run_id=args.run_id, concurrency=args.concurrency, eval_set_path=eval_set_path)


if __name__ == "__main__":
    main()

