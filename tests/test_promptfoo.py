"""Offline unit tests for Promptfoo integration (provider, assertions, runner)."""

from unittest.mock import MagicMock, patch

from eval.promptfoo_assertions import (
    assert_gold_evidence_present,
    assert_retrieval_doc_match,
    assert_vietnamese_refusal,
    get_assert,
)
from eval.promptfoo_provider import (
    call_api,
    call_rag_A,
    call_rag_B,
    call_rag_pipeline,
)
from eval.promptfoo_runner import (
    generate_promptfoo_markdown_report,
    parse_promptfoo_results,
)


# ===================== Provider Tests =====================

@patch("eval.promptfoo_provider.ask")
def test_call_rag_pipeline_success(mock_ask):
    mock_ask.return_value = {
        "answer": "Người từ 15 tuổi được mở ví MoMo.",
        "cost_usd": 0.00015,
        "usage": {"total_tokens": 500, "input_tokens": 400, "output_tokens": 100},
        "latency_ms": {"retrieval": 200.0, "generation": 800.0, "total": 1000.0},
        "retrieved": [{"chunk_id": "c1", "doc_id": "momo_terms", "score": 0.85}],
    }

    res = call_rag_pipeline(
        prompt="15 tuổi có mở ví MoMo được không?",
        config_id="A",
        context={"vars": {"question": "15 tuổi có mở ví MoMo được không?"}},
    )

    assert res["output"] == "Người từ 15 tuổi được mở ví MoMo."
    assert res["cost"] == 0.00015
    assert res["tokenUsage"]["total"] == 500
    assert res["metadata"]["config_id"] == "A"
    assert "momo_terms" in res["metadata"]["retrieved_docs"]


@patch("eval.promptfoo_provider.ask")
def test_call_rag_helpers(mock_ask):
    mock_ask.return_value = {
        "answer": "Hạn mức tối đa 50.000đ.",
        "cost_usd": 0.0001,
        "usage": {"total_tokens": 300, "input_tokens": 200, "output_tokens": 100},
        "latency_ms": {"total": 500.0},
        "retrieved": [],
    }

    # Call A
    res_a = call_rag_A("Hạn mức bao nhiêu?")
    assert res_a["output"] == "Hạn mức tối đa 50.000đ."
    assert res_a["metadata"]["config_id"] == "A"

    # Call B
    res_b = call_rag_B("Hạn mức bao nhiêu?")
    assert res_b["metadata"]["config_id"] == "B"

    # Call default router
    res_default = call_api("Hạn mức bao nhiêu?", context={"vars": {"config_id": "A"}})
    assert res_default["metadata"]["config_id"] == "A"


@patch("eval.promptfoo_provider.ask")
def test_call_rag_pipeline_error_handling(mock_ask):
    mock_ask.side_effect = RuntimeError("Chroma connection error")

    res = call_rag_pipeline("Câu hỏi bất kỳ", config_id="A")
    assert "error" in res
    assert "Chroma connection error" in res["error"]


# ===================== Custom Assertion Tests =====================

def test_assert_vietnamese_refusal():
    # Case 1: Unanswerable query correctly refused
    context_unanswerable = {"vars": {"answerable": False}}
    refusal_output = "Rất tiếc, tài liệu không có thông tin về quy trình xin visa."
    res1 = assert_vietnamese_refusal(refusal_output, context_unanswerable)
    assert res1["pass"] is True
    assert res1["score"] == 1.0

    # Case 2: Unanswerable query hallucinated / not refused
    hallucination_output = "Bước 1 bạn nộp hồ sơ qua app MoMo..."
    res2 = assert_vietnamese_refusal(hallucination_output, context_unanswerable)
    assert res2["pass"] is False
    assert res2["score"] == 0.0

    # Case 3: Answerable query passes by default
    context_answerable = {"vars": {"answerable": True}}
    res3 = assert_vietnamese_refusal("Từ 15 tuổi được mở ví.", context_answerable)
    assert res3["pass"] is True


def test_assert_gold_evidence_present():
    context = {
        "vars": {
            "gold_answer": "Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi được mở ví MoMo.",
        }
    }

    good_output = "Theo quy định, người từ đủ 15 tuổi đến chưa đủ 18 tuổi có thể mở ví MoMo phù hợp pháp luật."
    res_good = assert_gold_evidence_present(good_output, context)
    assert res_good["pass"] is True
    assert res_good["score"] > 0.35

    bad_output = "MoMo áp dụng cho doanh nghiệp lớn thanh toán quốc tế."
    res_bad = assert_gold_evidence_present(bad_output, context)
    assert res_bad["pass"] is False


def test_assert_retrieval_doc_match():
    context_match = {
        "vars": {"gold_doc_ids": ["momo_terms"]},
        "provider": {"metadata": {"retrieved_docs": ["momo_terms", "momo_privacy"]}},
    }
    assert assert_retrieval_doc_match("output", context_match)["pass"] is True

    context_miss = {
        "vars": {"gold_doc_ids": ["momo_terms"]},
        "provider": {"metadata": {"retrieved_docs": ["momo_linking"]}},
    }
    assert assert_retrieval_doc_match("output", context_miss)["pass"] is False


def test_get_assert_routing():
    # Route to refusal check
    ctx_refusal = {"vars": {"answerable": False}}
    assert get_assert("Không có thông tin.", ctx_refusal)["pass"] is True

    # Route to evidence check
    ctx_answerable = {
        "vars": {
            "answerable": True,
            "gold_answer": "Hạn mức nạp tiền 50 triệu",
        }
    }
    assert get_assert("Hạn mức nạp tiền tối đa là 50 triệu", ctx_answerable)["pass"] is True


# ===================== Runner & Parsing Tests =====================

def test_parse_promptfoo_results():
    sample_data = {
        "results": {
            "table": {
                "head": {
                    "prompts": [
                        {"label": "Config A", "provider": "call_rag_A"},
                        {"label": "Config B", "provider": "call_rag_B"},
                    ]
                },
                "body": [
                    {
                        "outputs": [
                            {"pass": True, "cost": 0.0001, "latencyMs": 1000},
                            {"pass": True, "cost": 0.0002, "latencyMs": 800},
                        ]
                    },
                    {
                        "outputs": [
                            {"pass": False, "cost": 0.0001, "latencyMs": 1200},
                            {"pass": True, "cost": 0.0002, "latencyMs": 700},
                        ]
                    },
                ],
            }
        }
    }

    parsed = parse_promptfoo_results(sample_data)
    assert parsed["total_test_cases"] == 2
    providers = parsed["providers"]

    assert providers["Config A"]["total_tests"] == 2
    assert providers["Config A"]["passed_tests"] == 1
    assert providers["Config A"]["failed_tests"] == 1
    assert providers["Config A"]["pass_rate"] == 50.0
    assert providers["Config A"]["avg_latency_ms"] == 1100.0

    assert providers["Config B"]["total_tests"] == 2
    assert providers["Config B"]["passed_tests"] == 2
    assert providers["Config B"]["pass_rate"] == 100.0
    assert providers["Config B"]["avg_latency_ms"] == 750.0

    # Markdown report generation
    report = generate_promptfoo_markdown_report(parsed)
    assert "Config A" in report
    assert "Config B" in report
    assert "50.0%" in report
    assert "100.0%" in report

