"""Runner module to execute Promptfoo evaluation and format comparison reports.

Provides both CLI and Python APIs to:
1. Trigger Promptfoo evaluation (using npx promptfoo eval)
2. Export results to JSON and Markdown
3. Extract pass rate, latency, token usage, and cost per configuration
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("promptfoo_runner")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "promptfooconfig.yaml"
RESULTS_DIR = BASE_DIR / "results"


def check_promptfoo_installed() -> bool:
    """Check if npx or promptfoo is available on the system."""
    return shutil.which("npx") is not None or shutil.which("promptfoo") is not None


import yaml
from eval.promptfoo_assertions import get_assert

# Ensure UTF-8 output on Windows
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


def run_promptfoo_local(
    config_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    runs_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute local offline Promptfoo evaluation using logged runs and Python assertions."""
    cfg_file = config_path or CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tests = cfg.get("tests", [])
    providers = cfg.get("providers", [])

    # Map logged runs: config A -> run_live_test_A, config B -> run_live_test_B
    log_file = runs_log_path or (BASE_DIR / "logs" / "runs.jsonl")
    runs_by_config: Dict[str, Dict[str, Any]] = {"A": {}, "B": {}}
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    cfg_id = rec.get("config_id") or ("A" if "test_A" in rec.get("run_id", "") else "B")
                    qid = rec.get("question_id")
                    if cfg_id and qid:
                        runs_by_config[cfg_id][qid] = rec
                except Exception:
                    continue

    table_prompts = []
    for p in providers:
        table_prompts.append({
            "label": p.get("label", p.get("id")),
            "provider": p.get("id"),
        })

    table_body = []

    for test in tests:
        test_vars = test.get("vars", {})
        qid = test_vars.get("id") or test_vars.get("question_id")
        assertions = test.get("assert", [])

        row_outputs = []
        for p in providers:
            p_id = p.get("id", "")
            cfg_id = "A" if "rag_A" in p_id or "Config A" in p.get("label", "") else "B"

            run_entry = runs_by_config.get(cfg_id, {}).get(qid)
            if run_entry:
                answer = run_entry.get("answer", "")
                cost = run_entry.get("cost_usd", 0.0)
                usage = run_entry.get("usage", {})
                latency = run_entry.get("latency_ms", {}).get("total", 0.0)
                retrieved = run_entry.get("retrieved", [])
            else:
                answer = "Không có câu trả lời trong dữ liệu chạy."
                cost, usage, latency, retrieved = 0.0, {}, 0.0, []

            # Evaluate assertions
            all_passed = True
            grading_results = []

            context = {
                "vars": test_vars,
                "provider": {
                    "metadata": {
                        "retrieved_docs": list({r.get("doc_id") for r in retrieved if r.get("doc_id")}),
                    }
                },
            }

            for a in assertions:
                a_type = a.get("type", "")
                a_val = a.get("value", "")
                if a_type == "icontains":
                    passed = a_val.lower() in answer.lower()
                    grading_results.append({
                        "pass": passed,
                        "score": 1.0 if passed else 0.0,
                        "reason": f"icontains: '{a_val}' {'tìm thấy' if passed else 'KHÔNG tìm thấy'}",
                    })
                    if not passed:
                        all_passed = False
                elif a_type == "python":
                    res = get_assert(answer, context)
                    p_res = res if isinstance(res, bool) else res.get("pass", False)
                    grading_results.append({
                        "pass": p_res,
                        "score": 1.0 if p_res else 0.0,
                        "reason": res.get("reason", "") if isinstance(res, dict) else "",
                    })
                    if not p_res:
                        all_passed = False

            row_outputs.append({
                "pass": all_passed,
                "score": 1.0 if all_passed else 0.0,
                "text": answer,
                "cost": cost,
                "latencyMs": latency,
                "tokenUsage": {
                    "total": usage.get("total_tokens", 0),
                    "prompt": usage.get("input_tokens", 0),
                    "completion": usage.get("output_tokens", 0),
                },
                "gradingResult": {
                    "pass": all_passed,
                    "componentResults": grading_results,
                },
            })

        table_body.append({
            "vars": test_vars,
            "outputs": row_outputs,
        })

    promptfoo_data = {
        "results": {
            "version": 3,
            "table": {
                "head": {"prompts": table_prompts},
                "body": table_body,
            },
        }
    }

    out_file = output_path or (RESULTS_DIR / "promptfoo_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(promptfoo_data, f, ensure_ascii=False, indent=2)

    return promptfoo_data


def parse_promptfoo_results(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Promptfoo output and generate summary metrics per provider/config."""
    eval_results = results_data.get("results", {})
    table = eval_results.get("table", {})
    head = table.get("head", {})
    providers = head.get("prompts", [])
    body = table.get("body", [])

    provider_summaries: Dict[str, Dict[str, Any]] = {}
    for prov in providers:
        prov_label = prov.get("label") or prov.get("provider") or "Unknown"
        provider_summaries[prov_label] = {
            "label": prov_label,
            "provider_id": prov.get("provider"),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "latencies": [],
        }

    for row in body:
        outputs = row.get("outputs", [])
        for i, out in enumerate(outputs):
            if i >= len(providers):
                continue
            prov_label = providers[i].get("label") or providers[i].get("provider") or "Unknown"
            summary = provider_summaries[prov_label]
            summary["total_tests"] += 1
            if out.get("pass"):
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1

            cost = out.get("cost", 0.0) or 0.0
            summary["total_cost"] += cost

            latency = out.get("latencyMs", 0.0) or 0.0
            if latency > 0:
                summary["latencies"].append(latency)

    for prov_label, summary in provider_summaries.items():
        total = summary["total_tests"]
        summary["pass_rate"] = round((summary["passed_tests"] / total * 100) if total > 0 else 0.0, 2)
        summary["total_cost"] = round(summary["total_cost"], 6)
        if summary["latencies"]:
            summary["avg_latency_ms"] = round(sum(summary["latencies"]) / len(summary["latencies"]), 2)

    return {
        "providers": provider_summaries,
        "total_test_cases": len(body),
    }


def generate_promptfoo_markdown_report(summary: Dict[str, Any]) -> str:
    """Format Promptfoo evaluation results into a Markdown report."""
    providers = summary.get("providers", {})
    lines = [
        "# Báo cáo kiểm thử tự động Promptfoo: So sánh cấu hình MoMo RAG",
        "",
        f"Tổng số ca kiểm thử (test cases): **{summary.get('total_test_cases', 0)}**",
        "",
        "## 1. Bảng so sánh giữa các Cấu hình / Provider",
        "",
        "| Cấu hình / Provider | Số test | Đạt (Pass) | Hỏng (Fail) | Tỷ lệ Đạt (%) | Độ trễ TB (ms) | Tổng chi phí ($) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for p_name, data in providers.items():
        lines.append(
            f"| **{p_name}** | {data['total_tests']} | {data['passed_tests']} | "
            f"{data['failed_tests']} | **{data['pass_rate']}%** | {data['avg_latency_ms']} ms | ${data['total_cost']:.6f} |"
        )

    lines.extend([
        "",
        "## 2. Tiêu chí kiểm thử (Assertions)",
        "- **Factual groundness (Từ khóa quan trọng)**: Kiểm tra thông tin cốt lõi (độ tuổi, hạn mức, chính sách) có xuất hiện trong câu trả lời.",
        "- **Hallucination Prevention & Refusal**: Khi gặp câu hỏi ngoài ngữ cảnh (ví dụ: visa du lịch Nhật Bản), mô hình phải từ chối rõ ràng, không bịa đặt.",
        "- **Chi phí & Độ trễ**: Đo đạc latency thực tế và tổng chi phí tiêu thụ token theo bảng giá niêm yết.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promptfoo CLI runner for MoMo RAG.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to promptfooconfig.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Path to export results json")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrency")
    parser.add_argument("--local", action="store_true", default=True, help="Run fast local evaluation using logged runs (default: True)")
    parser.add_argument("--cli", action="store_true", default=False, help="Run via npx promptfoo eval CLI")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        if args.cli:
            logger.info("Chạy Promptfoo thông qua npx CLI...")
            raw_results = run_promptfoo(
                config_path=args.config,
                output_path=args.output,
                max_concurrency=args.concurrency,
                verbose=args.verbose,
            )
        else:
            logger.info("Chạy Promptfoo ở chế độ Local (offline, siêu tốc)...")
            raw_results = run_promptfoo_local(
                config_path=args.config,
                output_path=args.output,
            )

        parsed = parse_promptfoo_results(raw_results)

        md_report = generate_promptfoo_markdown_report(parsed)
        report_path = (args.output.parent if args.output else RESULTS_DIR) / "promptfoo_summary.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_report)

        print(f"\n[DONE] Promptfoo evaluation hoàn tất!")
        print(f"Chi tiết JSON: {args.output or (RESULTS_DIR / 'promptfoo_results.json')}")
        print(f"Báo cáo Markdown: {report_path}\n")
        print(md_report)
    except Exception as e:
        logger.error("Lỗi khi chạy Promptfoo: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()

