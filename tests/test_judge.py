"""Unit tests for eval/judge.py LLM-as-a-Judge module."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from eval.judge import (
    clean_json_text,
    evaluate_single_run_item,
    judge_run,
    parse_judge_output,
)


def test_clean_and_parse_judge_output():
    """Verify JSON parsing with and without markdown code fences."""
    # 1. With code fence
    fence_json = """```json
    {
        "correctness": "correct",
        "faithful": true,
        "refused": false,
        "reason": "Câu trả lời đúng và có cơ sở."
    }
    ```"""
    parsed1 = parse_judge_output(fence_json)
    assert parsed1["correctness"] == "correct"
    assert parsed1["faithful"] is True
    assert parsed1["refused"] is False
    assert "đúng" in parsed1["reason"]

    # 2. Plain JSON without fence
    plain_json = """
    {
        "correctness": "partial",
        "faithful": false,
        "refused": true,
        "reason": "Đúng một phần nhưng thiếu điều kiện."
    }
    """
    parsed2 = parse_judge_output(plain_json)
    assert parsed2["correctness"] == "partial"
    assert parsed2["faithful"] is False
    assert parsed2["refused"] is True

    # 3. Invalid correctness value raises ValueError
    invalid_json = '{"correctness": "SUPER_CORRECT", "faithful": true, "refused": false, "reason": ""}'
    with pytest.raises(ValueError, match="correctness không hợp lệ"):
        parse_judge_output(invalid_json)


def test_evaluate_single_unanswerable_contract_enforcement():
    """Verify that unanswerable question with refused=False cannot be marked correct."""
    run_entry = {
        "run_id": "test_run",
        "question_id": "q_unans",
        "question": "MoMo có cung cấp dịch vụ gửi vàng không?",
        "answer": "Có, MoMo cho phép gửi vàng.",
        "retrieved": [],
        "error": None,
    }
    eval_item = {
        "id": "q_unans",
        "type": "unanswerable",
        "answerable": False,
        "gold_answer": "",
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "correctness": "correct",
        "faithful": True,
        "refused": False,
        "reason": "Trả lời đầy đủ.",
    })
    mock_llm.invoke.return_value = mock_response

    result = evaluate_single_run_item(
        run_entry=run_entry,
        eval_item=eval_item,
        prompt_template="{question} {gold_answer} {retrieved_chunks} {answer}",
        llm=mock_llm,
        chunks_lookup={},
    )

    assert result["correctness"] == "incorrect"
    assert result["refused"] is False
    assert "Hợp đồng: Câu unanswerable" in result["reason"]


def test_evaluate_single_with_retry_on_broken_json():
    """Verify evaluate_single_run_item retries when LLM initially returns broken JSON."""
    run_entry = {
        "run_id": "test_run",
        "question_id": "q_valid",
        "question": "Hạn mức MoMo là bao nhiêu?",
        "answer": "50 triệu.",
        "retrieved": [],
        "error": None,
    }
    eval_item = {
        "id": "q_valid",
        "type": "direct",
        "answerable": True,
        "gold_answer": "50 triệu đồng.",
    }

    mock_llm = MagicMock()
    broken_resp = MagicMock()
    broken_resp.content = "Đây là kết quả: { 'broken_json' ... "

    valid_resp = MagicMock()
    valid_resp.content = json.dumps({
        "correctness": "correct",
        "faithful": True,
        "refused": False,
        "reason": "Chính xác.",
    })

    mock_llm.invoke.side_effect = [broken_resp, valid_resp]

    result = evaluate_single_run_item(
        run_entry=run_entry,
        eval_item=eval_item,
        prompt_template="{question} {gold_answer} {retrieved_chunks} {answer}",
        llm=mock_llm,
        chunks_lookup={},
    )

    assert result["correctness"] == "correct"
    assert mock_llm.invoke.call_count == 2


@patch("eval.judge.ChatGoogleGenerativeAI")
def test_judge_run_orchestration(mock_chat_class, tmp_path, monkeypatch):
    """Verify end-to-end judge_run execution writing to results/<run_id>/judgements.jsonl."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_file = logs_dir / "runs.jsonl"

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("eval.judge.LOGS_DIR", logs_dir)
    monkeypatch.setattr("eval.judge.RUNS_LOG_PATH", runs_file)
    monkeypatch.setattr("eval.judge.RESULTS_DIR", results_dir)
    monkeypatch.setattr("eval.judge.get_api_key", lambda: "mock_key")

    run_id = "run_test_judge"
    run_data = {
        "run_id": run_id,
        "timestamp": "2026-10-04T00:00:00Z",
        "config_id": "A",
        "question_id": "q_01",
        "question": "Hạn mức MoMo là bao nhiêu?",
        "retrieved": [{"chunk_id": "c1", "doc_id": "doc1", "score": 0.9}],
        "final_prompt": "",
        "answer": "50 triệu.",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "cost_usd": 0.00001,
        "latency_ms": {"retrieval": 10, "generation": 50, "total": 60},
        "error": None,
    }
    with open(runs_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(run_data) + "\n")

    eval_file = tmp_path / "eval_set.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "q_01",
            "type": "direct",
            "answerable": True,
            "gold_answer": "50 triệu.",
        }) + "\n")

    mock_instance = MagicMock()
    mock_chat_class.return_value = mock_instance
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "correctness": "correct",
        "faithful": True,
        "refused": False,
        "reason": "Chuẩn xác.",
    })
    mock_instance.invoke.return_value = mock_resp

    judgements_path = judge_run(run_id=run_id, concurrency=1, eval_set_path=eval_file)

    assert judgements_path.exists()
    with open(judgements_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 1
    assert lines[0]["run_id"] == run_id
    assert lines[0]["question_id"] == "q_01"
    assert lines[0]["correctness"] == "correct"
    assert lines[0]["faithful"] is True

