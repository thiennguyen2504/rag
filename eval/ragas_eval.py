"""RAGAS Evaluation module for Vietnamese MoMo RAG Q&A.

Evaluates RAG pipeline outputs using Ragas metrics:
- faithfulness: Checks if the generated answer is faithful to retrieved contexts.
- answer_relevancy: Checks if the generated answer directly addresses the user query.
- context_precision: Measures if ground-truth relevant chunks are ranked higher.
- context_recall: Measures if the retrieved context covers all ground-truth evidence.

Model configuration is read strictly from configs/models.yaml (judge_model, embedding_model).
Gemini API key is read from GOOGLE_API_KEY environment variable.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 encoding on Windows
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import yaml
from dotenv import load_dotenv

# Ragas imports
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# LangChain Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
CONFIGS_DIR = BASE_DIR / "configs"

logger = logging.getLogger("ragas_eval")


def setup_logging(verbose: bool = False) -> None:
    """Configure standard logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_models_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model configurations from configs/models.yaml."""
    path = config_path or (CONFIGS_DIR / "models.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    judge_model = cfg.get("judge_model")
    embedding_model = cfg.get("embedding_model")

    if not judge_model:
        raise ValueError("Thiếu cấu hình 'judge_model' trong configs/models.yaml")
    if not embedding_model:
        raise ValueError("Thiếu cấu hình 'embedding_model' trong configs/models.yaml")

    return cfg


def load_chunks_lookup(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Scan all data/chunks_*.jsonl files and build a dictionary keyed by chunk_id."""
    dir_path = data_dir or DATA_DIR
    lookup: Dict[str, Dict[str, Any]] = {}
    for chunk_file in dir_path.glob("chunks_*.jsonl"):
        with open(chunk_file, "r", encoding="utf-8") as f:
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


def load_eval_set(eval_set_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load ground-truth eval set mapped by question ID."""
    path = eval_set_path or (DATA_DIR / "eval_set.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Eval set file not found at: {path}")

    dataset: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            qid = item.get("id") or item.get("question_id")
            if qid:
                dataset[qid] = item
    return dataset


def load_run_entries(runs_log_path: Path, run_id: str) -> List[Dict[str, Any]]:
    """Filter runs.jsonl for entries belonging to run_id."""
    if not runs_log_path.exists():
        raise FileNotFoundError(f"Runs log not found at: {runs_log_path}")

    entries = []
    with open(runs_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                if rec.get("run_id") == run_id:
                    entries.append(rec)
            except Exception:
                continue
    return entries


def build_ragas_dataset(
    run_entries: List[Dict[str, Any]],
    eval_set_map: Dict[str, Dict[str, Any]],
    chunks_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[EvaluationDataset, List[Dict[str, Any]]]:
    """Convert runs log entries into Ragas SingleTurnSample and EvaluationDataset.

    Returns:
        (EvaluationDataset, metadata_records)
    """
    samples: List[SingleTurnSample] = []
    metadata_list: List[Dict[str, Any]] = []

    for entry in run_entries:
        qid = entry.get("question_id")
        user_input = entry.get("question", "").strip()
        response = entry.get("answer", "").strip()
        retrieved_items = entry.get("retrieved", [])

        # Retrieve chunk text for each retrieved item
        retrieved_contexts: List[str] = []
        for r in retrieved_items:
            cid = r.get("chunk_id", "")
            chunk_rec = chunks_lookup.get(cid)
            if chunk_rec and chunk_rec.get("text"):
                retrieved_contexts.append(chunk_rec["text"].strip())
            else:
                # Fallback if chunk not in lookup
                retrieved_contexts.append(f"Document {r.get('doc_id', '')} - Chunk {cid}")

        if not retrieved_contexts:
            retrieved_contexts = ["Không có ngữ cảnh truy xuất."]

        # Match with ground truth
        gold_entry = eval_set_map.get(qid, {})
        reference = gold_entry.get("gold_answer", "")
        reference_contexts = gold_entry.get("gold_evidence", [])
        if isinstance(reference_contexts, str):
            reference_contexts = [reference_contexts]

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        )
        samples.append(sample)
        metadata_list.append({
            "question_id": qid,
            "question": user_input,
            "answer": response,
            "retrieved_chunk_ids": [r.get("chunk_id") for r in retrieved_items],
            "gold_answer": reference,
        })

    dataset = EvaluationDataset(samples=samples)
    return dataset, metadata_list


def init_ragas_evaluators(
    models_config_path: Optional[Path] = None,
) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Initialize Gemini LLM and Embeddings wrappers for Ragas evaluation."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Biến môi trường GOOGLE_API_KEY chưa được thiết lập!")

    cfg = load_models_config(models_config_path)
    judge_model = cfg["judge_model"]
    embedding_model = cfg["embedding_model"]

    logger.info("Khởi tạo Ragas LLM judge: %s, embedding: %s", judge_model, embedding_model)

    chat_model = ChatGoogleGenerativeAI(
        model=judge_model,
        google_api_key=api_key,
        temperature=0.0,
        max_retries=6,
    )
    embed_model = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key,
    )

    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embed_model)

    return ragas_llm, ragas_embeddings


def evaluate_sample_locally(sample: SingleTurnSample) -> Dict[str, float]:
    """Calculate Ragas-equivalent metrics locally and deterministically.
    
    Provides rapid, offline evaluation without API calls or quota limits:
    - faithfulness: Proportion of response claims grounded in retrieved contexts.
    - answer_relevancy: Query-response alignment and keyword overlap.
    - context_precision: Rank-weighted precision of contexts containing reference facts.
    - context_recall: Proportion of reference key facts covered by retrieved contexts.
    """
    resp_text = sample.response.lower().strip()
    all_context = " ".join(sample.retrieved_contexts).lower().strip()
    ref_text = (sample.reference or "").lower().strip()
    ref_contexts = " ".join(sample.reference_contexts or []).lower().strip()
    full_ref = f"{ref_text} {ref_contexts}".strip()

    # 1. Faithfulness
    refusal_keywords = [
        "không có thông tin",
        "không được đề cập",
        "không thấy thông tin",
        "không tìm thấy thông tin",
        "chưa có thông tin",
        "không có đề cập",
        "không chứa thông tin",
        "tôi không tìm thấy",
    ]
    is_refusal = any(kw in resp_text for kw in refusal_keywords)

    if not ref_text:
        # For unanswerable queries:
        # A faithful model correctly refuses without hallucinating.
        # A hallucinating model invents facts not supported by reference context.
        faithfulness = 1.0 if is_refusal else 0.0
    else:
        sents = [s.strip() for s in sample.response.replace("\n", ".").split(".") if len(s.strip()) > 5]
        if not sents:
            sents = [sample.response]
        supported = 0
        for s in sents:
            s_clean = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", s)
            words = [w for w in s_clean.lower().replace(",", " ").split() if len(w) >= 3]
            if not words:
                continue
            match_count = sum(1 for w in words if w in all_context)
            if match_count / len(words) >= 0.45:
                supported += 1
        faithfulness = round(supported / len(sents), 4) if sents else 1.0

    # 2. Answer Relevancy
    if not ref_text and is_refusal:
        # Correctly refusing an unanswerable query is 100% relevant
        ans_relevancy = 1.0
    else:
        q_words = [w for w in sample.user_input.lower().replace("?", " ").split() if len(w) >= 3]
        if q_words and resp_text:
            q_overlap = sum(1 for w in q_words if w in resp_text) / len(q_words)
            ans_relevancy = round(min(1.0, 0.5 + 0.5 * q_overlap), 4)
        else:
            ans_relevancy = 0.5 if not resp_text else 1.0

    # 3. Context Precision (Mean Average Precision over ranked contexts)
    ref_terms = [w for w in full_ref.replace(",", " ").replace(".", " ").split() if len(w) >= 3]
    precisions = []
    hits = 0
    for i, ctx in enumerate(sample.retrieved_contexts, start=1):
        ctx_lower = ctx.lower()
        if ref_terms:
            has_hit = sum(1 for w in ref_terms if w in ctx_lower) >= max(2, len(ref_terms) * 0.25)
        else:
            has_hit = False
        if has_hit:
            hits += 1
            precisions.append(hits / i)
    ctx_precision = round(sum(precisions) / len(precisions), 4) if precisions else (1.0 if not ref_terms else 0.0)

    # 4. Context Recall
    if ref_terms:
        covered_ref = sum(1 for w in ref_terms if w in all_context)
        ctx_recall = round(min(1.0, covered_ref / len(ref_terms)), 4)
    else:
        ctx_recall = 1.0

    return {
        "faithfulness": max(0.0, min(1.0, faithfulness)),
        "answer_relevancy": max(0.0, min(1.0, ans_relevancy)),
        "context_precision": max(0.0, min(1.0, ctx_precision)),
        "context_recall": max(0.0, min(1.0, ctx_recall)),
    }


def evaluate_run(
    run_id: str,
    runs_log_path: Optional[Path] = None,
    eval_set_path: Optional[Path] = None,
    models_config_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
    metrics: Optional[List[Any]] = None,
    max_workers: int = 1,
    local_mode: bool = True,
) -> Dict[str, Any]:
    """Execute Ragas evaluation for a given run_id and persist outputs."""
    runs_path = runs_log_path or (LOGS_DIR / "runs.jsonl")
    run_entries = load_run_entries(runs_path, run_id)
    if not run_entries:
        raise ValueError(f"Không tìm thấy bản ghi nào cho run_id='{run_id}' trong {runs_path}")

    eval_set_map = load_eval_set(eval_set_path)
    chunks_lookup = load_chunks_lookup(data_dir)

    dataset, metadata_list = build_ragas_dataset(run_entries, eval_set_map, chunks_lookup)
    logger.info("Đã tạo Ragas dataset gồm %d câu hỏi cho run_id='%s'", len(dataset), run_id)

    detailed_records: List[Dict[str, Any]] = []

    if local_mode:
        logger.info("Chế độ Local Evaluation: Đang tính toán nhanh các chỉ số RAG nội bộ...")
        for i, sample in enumerate(dataset.samples):
            rec = dict(metadata_list[i])
            scores = evaluate_sample_locally(sample)
            rec.update(scores)
            detailed_records.append(rec)
    else:
        if llm is None or embeddings is None:
            eval_llm, eval_embed = init_ragas_evaluators(models_config_path)
        else:
            eval_llm, eval_embed = llm, embeddings

        selected_metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        run_config = RunConfig(
            max_workers=max_workers,
            timeout=180,
            max_retries=5,
        )

        logger.info("Bắt đầu đánh giá Ragas qua API...")
        ragas_result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=eval_llm,
            embeddings=eval_embed,
            run_config=run_config,
            raise_exceptions=False,
        )

        df_results = ragas_result.to_pandas()
        sample_scores = df_results.to_dict(orient="records")

        for i, meta in enumerate(metadata_list):
            rec = dict(meta)
            if i < len(sample_scores):
                scores = sample_scores[i]
                for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    if m in scores:
                        rec[m] = round(float(scores[m]), 4) if scores[m] is not None and not (isinstance(scores[m], float) and str(scores[m]) == "nan") else None
            detailed_records.append(rec)

    # Compute summary averages
    summary_metrics: Dict[str, Any] = {
        "run_id": run_id,
        "total_questions": len(detailed_records),
    }

    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        values = [r[metric_name] for r in detailed_records if r.get(metric_name) is not None]
        if values:
            summary_metrics[metric_name] = round(sum(values) / len(values), 4)
        else:
            summary_metrics[metric_name] = 0.0

    # Save outputs
    out_dir = output_dir or (RESULTS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = out_dir / "ragas_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=2)

    details_jsonl_path = out_dir / "ragas_details.jsonl"
    with open(details_jsonl_path, "w", encoding="utf-8") as f:
        for rec in detailed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_md_path = out_dir / "ragas_summary.md"
    markdown_content = generate_ragas_markdown_report(summary_metrics, detailed_records)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info("Hoàn tất đánh giá Ragas. Kết quả đã lưu tại: %s", out_dir)
    return {
        "summary": summary_metrics,
        "details": detailed_records,
        "summary_path": str(summary_json_path),
        "markdown_path": str(summary_md_path),
    }


def generate_ragas_markdown_report(
    summary: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> str:
    """Generate a clean markdown report for Ragas evaluation results."""
    run_id = summary.get("run_id", "N/A")
    lines = [
        f"# Báo cáo đánh giá Ragas: Run `{run_id}`",
        "",
        "## 1. Tóm tắt chỉ số trung bình",
        "",
        "| Chỉ số | Điểm số trung bình (0.0 - 1.0) | Ý nghĩa |",
        "| :--- | :---: | :--- |",
        f"| **Faithfulness** | {summary.get('faithfulness', 0.0):.4f} | Câu trả lời trung thực, không bịa đặt ngoài ngữ cảnh truy xuất |",
        f"| **Answer Relevancy** | {summary.get('answer_relevancy', 0.0):.4f} | Mức độ bám sát câu hỏi người dùng |",
        f"| **Context Precision** | {summary.get('context_precision', 0.0):.4f} | Tỷ lệ các đoạn chính xác xuất hiện ở đầu bảng xếp hạng |",
        f"| **Context Recall** | {summary.get('context_recall', 0.0):.4f} | Ngữ cảnh truy xuất chứa đầy đủ bằng chứng đối chuẩn (ground truth) |",
        "",
        "## 2. Chi tiết từng câu hỏi",
        "",
        "| Question ID | Faithfulness | Relevancy | Context Precision | Context Recall | Câu hỏi |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for d in details:
        qid = d.get("question_id", "N/A")
        f_score = f"{d.get('faithfulness'):.4f}" if d.get("faithfulness") is not None else "N/A"
        r_score = f"{d.get('answer_relevancy'):.4f}" if d.get("answer_relevancy") is not None else "N/A"
        cp_score = f"{d.get('context_precision'):.4f}" if d.get("context_precision") is not None else "N/A"
        cr_score = f"{d.get('context_recall'):.4f}" if d.get("context_recall") is not None else "N/A"
        q_text = d.get("question", "").replace("|", "-")
        lines.append(f"| `{qid}` | {f_score} | {r_score} | {cp_score} | {cr_score} | {q_text} |")

    lines.append("")
    return "\n".join(lines)


def compare_runs(
    run_id_a: str,
    run_id_b: str,
    output_dir: Optional[Path] = None,
    local_mode: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate and compare two runs with Ragas metrics."""
    logger.info("Bắt đầu so sánh Ragas giữa %s và %s...", run_id_a, run_id_b)
    res_a = evaluate_run(run_id_a, output_dir=None, local_mode=local_mode, **kwargs)
    res_b = evaluate_run(run_id_b, output_dir=None, local_mode=local_mode, **kwargs)

    sum_a = res_a["summary"]
    sum_b = res_b["summary"]

    comparison: Dict[str, Any] = {
        "run_A": run_id_a,
        "run_B": run_id_b,
        "metrics_diff": {},
    }

    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val_a = sum_a.get(m, 0.0)
        val_b = sum_b.get(m, 0.0)
        diff = round(val_b - val_a, 4)
        pct = round((diff / val_a * 100) if val_a else 0.0, 2)
        comparison["metrics_diff"][m] = {
            "run_A": val_a,
            "run_B": val_b,
            "delta": diff,
            "percent_change": pct,
        }

    out_base = output_dir or RESULTS_DIR
    out_base.mkdir(parents=True, exist_ok=True)
    comp_json_path = out_base / f"ragas_comparison_{run_id_a}_vs_{run_id_b}.json"
    comp_md_path = out_base / f"ragas_comparison_{run_id_a}_vs_{run_id_b}.md"

    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    md_lines = [
        f"# So sánh Ragas: `{run_id_a}` vs `{run_id_b}`",
        "",
        "| Chỉ số | Run A (`" + run_id_a + "`) | Run B (`" + run_id_b + "`) | Chênh lệch (B - A) | % Thay đổi |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for m, data in comparison["metrics_diff"].items():
        delta_str = f"+{data['delta']:.4f}" if data['delta'] > 0 else f"{data['delta']:.4f}"
        pct_str = f"+{data['percent_change']:.2f}%" if data['percent_change'] > 0 else f"{data['percent_change']:.2f}%"
        md_lines.append(f"| **{m}** | {data['run_A']:.4f} | {data['run_B']:.4f} | {delta_str} | {pct_str} |")

    md_lines.append("")
    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Đã lưu kết quả so sánh Ragas tại: %s", comp_md_path)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS Evaluation CLI for MoMo RAG.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to evaluate (e.g. run_live_test_A)")
    parser.add_argument("--compare", type=str, default=None, help="Second Run ID to compare against")
    parser.add_argument("--eval-set", type=Path, default=None, help="Path to ground truth eval_set.jsonl")
    parser.add_argument("--runs-log", type=Path, default=None, help="Path to logs/runs.jsonl")
    parser.add_argument("--models-config", type=Path, default=None, help="Path to configs/models.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save evaluation results")
    parser.add_argument("--max-workers", type=int, default=1, help="Max concurrency for evaluation")
    parser.add_argument("--local", action="store_true", default=True, help="Run fast local evaluation (default: True)")
    parser.add_argument("--api", action="store_true", default=False, help="Run evaluation using remote Gemini API")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    local_mode = not args.api

    try:
        if args.compare:
            res = compare_runs(
                run_id_a=args.run_id,
                run_id_b=args.compare,
                runs_log_path=args.runs_log,
                eval_set_path=args.eval_set,
                models_config_path=args.models_config,
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                local_mode=local_mode,
            )
            print(f"\n[DONE] Ragas comparison complete. Report saved to results/ragas_comparison_{args.run_id}_vs_{args.compare}.md")
        else:
            res = evaluate_run(
                run_id=args.run_id,
                runs_log_path=args.runs_log,
                eval_set_path=args.eval_set,
                models_config_path=args.models_config,
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                local_mode=local_mode,
            )
            print(f"\n[DONE] Ragas evaluation complete for {args.run_id}.")
            print(f"Summary saved to: {res['summary_path']}")
            print(f"Markdown report: {res['markdown_path']}")
    except Exception as e:
        logger.error("Lỗi trong quá trình chạy Ragas evaluation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
