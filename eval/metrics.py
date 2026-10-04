"""Evaluation metrics calculation module for RAG system benchmarking."""

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rapidfuzz

from src.config import BASE_DIR
from eval.judge import load_chunks_lookup, load_eval_set_map

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("metrics")

LOGS_DIR = BASE_DIR / "logs"
RUNS_LOG_PATH = LOGS_DIR / "runs.jsonl"
RESULTS_DIR = BASE_DIR / "results"


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and lowercasing."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def check_chunk_contains_evidence(chunk_norm: str, evidence_norm: str) -> bool:
    """Check if a normalized chunk contains normalized evidence (exact substring or fuzzy >= 90)."""
    if not chunk_norm or not evidence_norm:
        return False
    if evidence_norm in chunk_norm:
        return True
    return rapidfuzz.fuzz.partial_ratio(evidence_norm, chunk_norm) >= 90.0


def compute_retrieval_stats(
    retrieved_list: List[Dict[str, Any]],
    gold_doc_ids: List[str],
    gold_evidences: List[str],
    chunks_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute retrieval metrics for an individual answerable question."""
    if not gold_doc_ids and not gold_evidences:
        return {
            "doc_hit": None,
            "evidence_hit": None,
            "evidence_recall": None,
            "mrr": None,
        }

    retrieved_docs = [r.get("doc_id", "") for r in retrieved_list]
    retrieved_chunk_norms = []
    for r in retrieved_list:
        cid = r.get("chunk_id", "")
        chk_data = chunks_lookup.get(cid, {})
        txt = chk_data.get("text", "")
        retrieved_chunk_norms.append(normalize_text(txt))

    doc_hit = any(d in gold_doc_ids for d in retrieved_docs)
    norm_evidences = [normalize_text(ev) for ev in gold_evidences if ev.strip()]

    if not norm_evidences:
        evidence_hit = doc_hit
        evidence_recall = 1.0 if doc_hit else 0.0
        mrr = 1.0 if doc_hit else 0.0
    else:
        matched_evidences_count = 0
        first_match_rank: Optional[int] = None

        for ev in norm_evidences:
            ev_found = False
            for rank_idx, chk_norm in enumerate(retrieved_chunk_norms):
                if check_chunk_contains_evidence(chk_norm, ev):
                    ev_found = True
                    if first_match_rank is None or rank_idx < first_match_rank:
                        first_match_rank = rank_idx
                    break
            if ev_found:
                matched_evidences_count += 1

        evidence_hit = matched_evidences_count > 0
        evidence_recall = matched_evidences_count / len(norm_evidences)
        mrr = 1.0 / (first_match_rank + 1) if first_match_rank is not None else 0.0

    return {
        "doc_hit": doc_hit,
        "evidence_hit": evidence_hit,
        "evidence_recall": round(evidence_recall, 4),
        "mrr": round(mrr, 4),
    }


def analyze_citations(answer: str, num_sources: int) -> Dict[str, Any]:
    """Check citation markers [n] in answer and test validity against sources count."""
    if not answer:
        return {"has_citation": False, "valid_citations": True, "total_citations": 0}

    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return {"has_citation": False, "valid_citations": True, "total_citations": 0}

    all_valid = True
    for c in citations:
        val = int(c)
        if val < 1 or val > num_sources:
            all_valid = False
            break

    return {
        "has_citation": True,
        "valid_citations": all_valid,
        "total_citations": len(citations),
    }


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero division."""
    if den == 0:
        return default
    return num / den


def compute_metrics_for_items(
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics for a collection of processed question records."""
    total_count = len(items)
    if total_count == 0:
        return {"count": 0}

    answerable_items = [it for it in items if it.get("answerable") is True]
    unanswerable_items = [it for it in items if it.get("answerable") is False]
    judged_items = [it for it in items if it.get("judgement") is not None]

    ans_count = len(answerable_items)
    if ans_count > 0:
        doc_hit_rate = safe_div(sum(1 for it in answerable_items if it.get("doc_hit")), ans_count)
        evidence_hit_rate = safe_div(sum(1 for it in answerable_items if it.get("evidence_hit")), ans_count)
        avg_evidence_recall = safe_div(sum(it.get("evidence_recall", 0.0) for it in answerable_items), ans_count)
        mrr = safe_div(sum(it.get("mrr", 0.0) for it in answerable_items), ans_count)
    else:
        doc_hit_rate = evidence_hit_rate = avg_evidence_recall = mrr = 0.0

    judge_count = len(judged_items)
    if judge_count > 0:
        correct_count = sum(1 for it in judged_items if it["judgement"].get("correctness") == "correct")
        partial_count = sum(1 for it in judged_items if it["judgement"].get("correctness") == "partial")
        incorrect_count = sum(1 for it in judged_items if it["judgement"].get("correctness") == "incorrect")
        faithful_count = sum(1 for it in judged_items if it["judgement"].get("faithful") is True)
        unfaithful_count = sum(1 for it in judged_items if it["judgement"].get("faithful") is False)

        correct_rate = safe_div(correct_count, judge_count)
        partial_rate = safe_div(partial_count, judge_count)
        incorrect_rate = safe_div(incorrect_count, judge_count)
        faithful_rate = safe_div(faithful_count, judge_count)
        unfaithful_rate = safe_div(unfaithful_count, judge_count)
    else:
        correct_rate = partial_rate = incorrect_rate = faithful_rate = unfaithful_rate = 0.0

    unans_count = len(unanswerable_items)
    if unans_count > 0:
        unans_judged = [it for it in unanswerable_items if it.get("judgement") is not None]
        unans_hallucination_count = sum(1 for it in unans_judged if it["judgement"].get("refused") is False)
        unanswerable_hallucination_rate = safe_div(unans_hallucination_count, len(unans_judged))
    else:
        unanswerable_hallucination_rate = 0.0

    if ans_count > 0:
        ans_judged = [it for it in answerable_items if it.get("judgement") is not None]
        ans_refusal_count = sum(1 for it in ans_judged if it["judgement"].get("refused") is True)
        answerable_refusal_rate = safe_div(ans_refusal_count, len(ans_judged))
    else:
        answerable_refusal_rate = 0.0

    retrieval_latencies = [it["latency_ms"]["retrieval"] for it in items if "latency_ms" in it]
    generation_latencies = [it["latency_ms"]["generation"] for it in items if "latency_ms" in it]
    total_latencies = [it["latency_ms"]["total"] for it in items if "latency_ms" in it]

    p50_retrieval = float(np.percentile(retrieval_latencies, 50)) if retrieval_latencies else 0.0
    p95_retrieval = float(np.percentile(retrieval_latencies, 95)) if retrieval_latencies else 0.0
    p50_generation = float(np.percentile(generation_latencies, 50)) if generation_latencies else 0.0
    p95_generation = float(np.percentile(generation_latencies, 95)) if generation_latencies else 0.0
    p50_total = float(np.percentile(total_latencies, 50)) if total_latencies else 0.0
    p95_total = float(np.percentile(total_latencies, 95)) if total_latencies else 0.0

    input_tokens = [it["usage"]["input_tokens"] for it in items if "usage" in it]
    output_tokens = [it["usage"]["output_tokens"] for it in items if "usage" in it]
    total_tokens = [it["usage"]["total_tokens"] for it in items if "usage" in it]
    costs = [it.get("cost_usd", 0.0) for it in items]

    avg_input_tokens = safe_div(sum(input_tokens), total_count)
    avg_output_tokens = safe_div(sum(output_tokens), total_count)
    avg_total_tokens = safe_div(sum(total_tokens), total_count)
    avg_cost_usd = safe_div(sum(costs), total_count)
    total_cost_usd = sum(costs)

    error_count = sum(1 for it in items if it.get("error"))
    error_rate = safe_div(error_count, total_count)

    citation_items = [it for it in items if it.get("citation_info")]
    if citation_items:
        with_citations = sum(1 for it in citation_items if it["citation_info"]["has_citation"])
        valid_citations = sum(1 for it in citation_items if it["citation_info"]["has_citation"] and it["citation_info"]["valid_citations"])
        citation_presence_rate = safe_div(with_citations, len(citation_items))
        citation_validity_rate = safe_div(valid_citations, with_citations) if with_citations > 0 else 1.0
    else:
        citation_presence_rate = citation_validity_rate = 0.0

    return {
        "count": total_count,
        "answerable_count": ans_count,
        "unanswerable_count": unans_count,
        "judged_count": judge_count,
        "retrieval": {
            "doc_hit_rate": round(doc_hit_rate, 4),
            "evidence_hit_rate": round(evidence_hit_rate, 4),
            "evidence_recall": round(avg_evidence_recall, 4),
            "mrr": round(mrr, 4),
        },
        "generation": {
            "correct_rate": round(correct_rate, 4),
            "partial_rate": round(partial_rate, 4),
            "incorrect_rate": round(incorrect_rate, 4),
            "faithful_rate": round(faithful_rate, 4),
        },
        "hallucination": {
            "unanswerable_hallucination_rate": round(unanswerable_hallucination_rate, 4),
            "answerable_refusal_rate": round(answerable_refusal_rate, 4),
            "unfaithful_rate": round(unfaithful_rate, 4),
        },
        "operational": {
            "latency_p50_ms": {
                "retrieval": round(p50_retrieval, 1),
                "generation": round(p50_generation, 1),
                "total": round(p50_total, 1),
            },
            "latency_p95_ms": {
                "retrieval": round(p95_retrieval, 1),
                "generation": round(p95_generation, 1),
                "total": round(p95_total, 1),
            },
            "avg_tokens": {
                "input": round(avg_input_tokens, 1),
                "output": round(avg_output_tokens, 1),
                "total": round(avg_total_tokens, 1),
            },
            "cost_usd": {
                "avg_per_query": round(avg_cost_usd, 6),
                "total": round(total_cost_usd, 6),
            },
            "errors": {
                "count": error_count,
                "rate": round(error_rate, 4),
            },
        },
        "citations": {
            "presence_rate": round(citation_presence_rate, 4),
            "validity_rate": round(citation_validity_rate, 4),
        },
    }


def generate_markdown_report(summary_data: Dict[str, Any]) -> str:
    """Generate clean, publication-ready markdown summary table report."""
    run_id = summary_data.get("run_id", "N/A")
    config_id = summary_data.get("config_id", "N/A")
    total_q = summary_data["overall"].get("count", 0)
    overall = summary_data["overall"]

    ret = overall["retrieval"]
    gen = overall["generation"]
    hal = overall["hallucination"]
    ops = overall["operational"]
    cit = overall["citations"]

    md = []
    md.append(f"# Báo Cáo Đánh Giá Thực Nghiệm RAG: `{run_id}`")
    md.append(f"\n- **Cấu hình (Config ID)**: `{config_id}`")
    md.append(f"- **Tổng số câu hỏi đánh giá**: `{total_q}`")
    md.append(f"- **Số câu hoàn tất chấm (Judged)**: `{overall.get('judged_count', 0)}`")
    md.append(f"- **Số lỗi hệ thống**: `{ops['errors']['count']}` (`{ops['errors']['rate']*100:.1f}%`)\n")

    md.append("## 1. Bảng Chỉ Số Tổng Thể (Overall Performance)\n")
    md.append("| Nhóm chỉ số | Chỉ số (Metric) | Kết quả | Ghi chú |")
    md.append("|---|---|---|---|")
    md.append(f"| **Retrieval** | Doc Hit@k | `{ret['doc_hit_rate']*100:.1f}%` | Tỉ lệ tìm thấy đúng tài liệu nguồn |")
    md.append(f"| | Evidence Hit@k | `{ret['evidence_hit_rate']*100:.1f}%` | Tỉ lệ tìm thấy ít nhất 1 đoạn chứng cứ |")
    md.append(f"| | Evidence Recall | `{ret['evidence_recall']*100:.1f}%` | Độ phủ chứng cứ (đặc biệt cho multi-hop) |")
    md.append(f"| | MRR | `{ret['mrr']:.4f}` | Thứ hạng trung bình của chunk chứa evidence |")
    md.append(f"| **Generation** | Correct Rate | `{gen['correct_rate']*100:.1f}%` | Tỉ lệ trả lời đúng hoàn toàn |")
    md.append(f"| | Partial Rate | `{gen['partial_rate']*100:.1f}%` | Tỉ lệ trả lời đúng một phần / thiếu ý |")
    md.append(f"| | Incorrect Rate | `{gen['incorrect_rate']*100:.1f}%` | Tỉ lệ trả lời sai hoặc bịa đặt |")
    md.append(f"| | Faithful Rate | `{gen['faithful_rate']*100:.1f}%` | Mọi khẳng định đều có cơ sở trong context |")
    md.append(f"| **Hallucination** | Unanswerable Hallucination | `{hal['unanswerable_hallucination_rate']*100:.1f}%` | Trả lời bừa khi câu hỏi unanswerable |")
    md.append(f"| | Answerable Refusal | `{hal['answerable_refusal_rate']*100:.1f}%` | Từ chối nhầm câu hỏi có đáp án |")
    md.append(f"| | Unfaithful Rate | `{hal['unfaithful_rate']*100:.1f}%` | Tỉ lệ câu có thông tin ngoài lề |")
    md.append(f"| **Citations** | Trích dẫn nguồn `[n]` | `{cit['presence_rate']*100:.1f}%` | Tỉ lệ câu có dấu trích dẫn |")
    md.append(f"| | Độ hợp lệ trích dẫn | `{cit['validity_rate']*100:.1f}%` | Tỉ lệ trích dẫn đúng số hiệu nguồn |")

    md.append("\n## 2. Phân Tích Chi Tiết Theo Loại Câu Hỏi (Breakdown by Type)\n")
    md.append("| Loại câu hỏi (Type) | Số câu | Doc Hit@k | Evidence Hit@k | Correct | Partial | Incorrect | Faithful | Refused |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    types_order = ["direct", "multi", "unanswerable", "trap", "casual"]
    by_type = summary_data.get("by_type", {})
    for t in types_order:
        t_data = by_type.get(t)
        if not t_data:
            continue
        c = t_data["count"]
        t_ret = t_data["retrieval"]
        t_gen = t_data["generation"]
        t_hal = t_data["hallucination"]

        doc_hit_str = f"{t_ret['doc_hit_rate']*100:.1f}%" if t_data["answerable_count"] > 0 else "N/A"
        ev_hit_str = f"{t_ret['evidence_hit_rate']*100:.1f}%" if t_data["answerable_count"] > 0 else "N/A"
        corr_str = f"{t_gen['correct_rate']*100:.1f}%"
        part_str = f"{t_gen['partial_rate']*100:.1f}%"
        incorr_str = f"{t_gen['incorrect_rate']*100:.1f}%"
        faith_str = f"{t_gen['faithful_rate']*100:.1f}%"

        if t == "unanswerable":
            refused_rate = 1.0 - t_hal["unanswerable_hallucination_rate"]
        else:
            refused_rate = t_hal["answerable_refusal_rate"]
        ref_str = f"{refused_rate*100:.1f}%"

        md.append(f"| `{t}` | {c} | {doc_hit_str} | {ev_hit_str} | {corr_str} | {part_str} | {incorr_str} | {faith_str} | {ref_str} |")

    md.append("\n## 3. Chỉ Số Vận Hành & Chi Phí (Operational & Cost)\n")
    lat_p50 = ops["latency_p50_ms"]
    lat_p95 = ops["latency_p95_ms"]
    tok = ops["avg_tokens"]
    cost = ops["cost_usd"]

    md.append("| Chỉ số | Retrieval | Generation | Total / Chi tiết |")
    md.append("|---|---|---|---|")
    md.append(f"| **Độ trễ p50 (ms)** | `{lat_p50['retrieval']}` ms | `{lat_p50['generation']}` ms | `{lat_p50['total']}` ms |")
    md.append(f"| **Độ trễ p95 (ms)** | `{lat_p95['retrieval']}` ms | `{lat_p95['generation']}` ms | `{lat_p95['total']}` ms |")
    md.append(f"| **Token trung bình** | Input: `{tok['input']}` | Output: `{tok['output']}` | Total: `{tok['total']}` tokens |")
    md.append(f"| **Chi phí API (USD)** | - | - | TB: `${cost['avg_per_query']:.6f}` / câu - Tổng: `${cost['total']:.6f}` |")

    md.append("\n---\n*Báo cáo được khởi tạo tự động bởi `eval.metrics`.*")
    return "\n".join(md) + "\n"


def compute_run_metrics(
    run_id: str,
    eval_set_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Load run logs and judgements, compute full metrics, and write summary.json & summary.md."""
    if eval_set_path is None:
        eval_set_path = BASE_DIR / "data" / "eval_set.jsonl"

    eval_set_map = load_eval_set_map(eval_set_path)
    chunks_lookup = load_chunks_lookup()

    if not RUNS_LOG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhật ký {RUNS_LOG_PATH}")

    run_entries = []
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entry = json.loads(line_str)
                if entry.get("run_id") == run_id:
                    run_entries.append(entry)
            except Exception:
                continue

    if not run_entries:
        raise ValueError(f"Không tìm thấy kết quả nào cho run_id='{run_id}' trong {RUNS_LOG_PATH}")

    config_id = run_entries[0].get("config_id", "A")

    run_results_dir = RESULTS_DIR / run_id
    judgements_path = run_results_dir / "judgements.jsonl"
    judgements_map: Dict[str, Dict[str, Any]] = {}
    if judgements_path.exists():
        with open(judgements_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    j = json.loads(line_str)
                    qid = str(j.get("question_id"))
                    judgements_map[qid] = j
                except Exception:
                    continue

    processed_items = []
    for entry in run_entries:
        qid = str(entry.get("question_id"))
        eval_meta = eval_set_map.get(qid, {})
        q_type = eval_meta.get("type", "direct")
        is_answerable = eval_meta.get("answerable", True)
        gold_doc_ids = eval_meta.get("gold_doc_ids", [])
        gold_evidences = eval_meta.get("gold_evidence", [])

        if is_answerable:
            ret_stats = compute_retrieval_stats(
                retrieved_list=entry.get("retrieved", []),
                gold_doc_ids=gold_doc_ids,
                gold_evidences=gold_evidences,
                chunks_lookup=chunks_lookup,
            )
        else:
            ret_stats = {"doc_hit": None, "evidence_hit": None, "evidence_recall": None, "mrr": None}

        num_sources = len(entry.get("retrieved", []))
        citation_info = analyze_citations(entry.get("answer", ""), num_sources)

        item_record = {
            "question_id": qid,
            "type": q_type,
            "answerable": is_answerable,
            "doc_hit": ret_stats["doc_hit"],
            "evidence_hit": ret_stats["evidence_hit"],
            "evidence_recall": ret_stats["evidence_recall"],
            "mrr": ret_stats["mrr"],
            "citation_info": citation_info,
            "judgement": judgements_map.get(qid),
            "latency_ms": entry.get("latency_ms", {}),
            "usage": entry.get("usage", {}),
            "cost_usd": entry.get("cost_usd", 0.0),
            "error": entry.get("error"),
        }
        processed_items.append(item_record)

    overall_metrics = compute_metrics_for_items(processed_items)

    by_type_metrics: Dict[str, Any] = {}
    known_types = ["direct", "multi", "unanswerable", "trap", "casual"]
    for t in known_types:
        subset = [it for it in processed_items if it.get("type") == t]
        if subset:
            by_type_metrics[t] = compute_metrics_for_items(subset)

    summary_data = {
        "run_id": run_id,
        "config_id": config_id,
        "overall": overall_metrics,
        "by_type": by_type_metrics,
    }

    run_results_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_results_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    md_content = generate_markdown_report(summary_data)
    md_path = run_results_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return summary_data, json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate metrics and generate summary report for an evaluation run.")
    parser.add_argument("--run-id", dest="run_id", type=str, required=True, help="Run ID cần tổng hợp chỉ số")
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

    summary_data, json_path, md_path = compute_run_metrics(run_id=args.run_id, eval_set_path=eval_set_path)

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG HỢP CHỈ SỐ (Run ID: {args.run_id})")
    print("=" * 80)
    overall = summary_data["overall"]
    print(f"Tổng số câu hỏi   : {overall['count']}")
    print(f"Đã chấm (Judged)  : {overall['judged_count']}")
    print(f"Retrieval Doc Hit : {overall['retrieval']['doc_hit_rate']*100:.1f}%")
    print(f"Evidence Hit      : {overall['retrieval']['evidence_hit_rate']*100:.1f}%")
    print(f"Evidence Recall   : {overall['retrieval']['evidence_recall']*100:.1f}%")
    print(f"MRR               : {overall['retrieval']['mrr']:.4f}")
    print(f"Correctness       : {overall['generation']['correct_rate']*100:.1f}%")
    print(f"Faithfulness      : {overall['generation']['faithful_rate']*100:.1f}%")
    print(f"Độ trễ Total p50  : {overall['operational']['latency_p50_ms']['total']} ms")
    print(f"Tổng chi phí      : ${overall['operational']['cost_usd']['total']:.6f} USD")
    print("-" * 80)
    print(f"Đã xuất báo cáo:")
    print(f"  - JSON : {json_path}")
    print(f"  - MD   : {md_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

