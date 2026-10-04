"""Unit tests for eval/run_eval.py evaluation runner."""

import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from eval.run_eval import run_evaluation, load_logged_question_ids, process_question


@pytest.fixture
def temp_eval_dir(tmp_path, monkeypatch):
    """Setup temporary directories for logs and eval files."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_file = logs_dir / "runs.jsonl"

    monkeypatch.setattr("eval.run_eval.LOGS_DIR", logs_dir)
    monkeypatch.setattr("eval.run_eval.RUNS_LOG_PATH", runs_file)

    eval_file = tmp_path / "eval_set.jsonl"
    items = [
        {"id": "q001", "question": "Hạn mức là bao nhiêu?", "type": "direct", "answerable": True},
        {"id": "q002", "question": "Cách thức thanh toán quốc tế?", "type": "direct", "answerable": True},
    ]
    with open(eval_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return tmp_path, eval_file, runs_file


@patch("eval.run_eval.ask")
def test_run_eval_schema_and_execution(mock_ask, temp_eval_dir):
    """Verify evaluation generates logs adhering to project log contract."""
    tmp_path, eval_file, runs_file = temp_eval_dir

    mock_ask.return_value = {
        "question": "Hạn mức là bao nhiêu?",
        "config_id": "A",
        "answer": "Hạn mức là 50 triệu.",
        "sources": [
            {
                "chunk_id": "terms-01",
                "doc_id": "terms",
                "score": 0.88,
                "title": "Điều khoản",
                "source_url": "url",
                "text": "text",
            }
        ],
        "final_prompt": "Prompt context...",
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "cost_usd": 0.00015,
        "latency_ms": {"retrieval": 10.0, "generation": 200.0, "total": 210.0},
        "error": None,
    }

    run_id = "test_run_01"
    run_evaluation(
        config_id="A",
        eval_set_path=eval_file,
        run_id=run_id,
        concurrency=1,
        resume=False,
    )

    assert runs_file.exists()
    logged_lines = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logged_lines.append(json.loads(line))

    assert len(logged_lines) == 2
    entry = logged_lines[0]
    assert entry["run_id"] == run_id
    assert "timestamp" in entry
    assert entry["config_id"] == "A"
    assert entry["question_id"] in {"q001", "q002"}
    assert "question" in entry
    assert isinstance(entry["retrieved"], list)
    assert entry["retrieved"][0]["chunk_id"] == "terms-01"
    assert entry["retrieved"][0]["doc_id"] == "terms"
    assert entry["retrieved"][0]["score"] == 0.88
    assert "final_prompt" in entry
    assert entry["answer"] == "Hạn mức là 50 triệu."
    assert entry["usage"]["total_tokens"] == 120
    assert entry["cost_usd"] == 0.00015
    assert entry["latency_ms"]["total"] == 210.0
    assert entry["error"] is None


@patch("eval.run_eval.ask")
def test_run_eval_resume(mock_ask, temp_eval_dir):
    """Verify --resume skips already processed question_ids."""
    tmp_path, eval_file, runs_file = temp_eval_dir

    mock_ask.return_value = {
        "question": "Sample question",
        "config_id": "A",
        "answer": "Sample answer",
        "sources": [],
        "final_prompt": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "latency_ms": {"retrieval": 0, "generation": 0, "total": 0},
        "error": None,
    }

    run_id = "test_run_resume"
    pre_logged = {
        "run_id": run_id,
        "timestamp": "2026-10-04T00:00:00Z",
        "config_id": "A",
        "question_id": "q001",
        "question": "Hạn mức là bao nhiêu?",
        "retrieved": [],
        "final_prompt": "",
        "answer": "Pre-logged answer",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "latency_ms": {"retrieval": 0, "generation": 0, "total": 0},
        "error": None,
    }
    with open(runs_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(pre_logged) + "\n")

    run_evaluation(
        config_id="A",
        eval_set_path=eval_file,
        run_id=run_id,
        concurrency=1,
        resume=True,
    )

    assert mock_ask.call_count == 1
    with open(runs_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    assert total_lines == 2


@patch("eval.run_eval.ask")
def test_run_eval_error_logging(mock_ask, temp_eval_dir):
    """Verify errors from pipeline.ask are recorded in log without crashing."""
    tmp_path, eval_file, runs_file = temp_eval_dir

    mock_ask.side_effect = RuntimeError("Chroma vector index corrupted")

    run_id = "test_run_error"
    run_evaluation(
        config_id="A",
        eval_set_path=eval_file,
        run_id=run_id,
        concurrency=1,
        limit=1,
    )

    with open(runs_file, "r", encoding="utf-8") as f:
        entry = json.loads(f.readline().strip())

    assert entry["run_id"] == run_id
    assert entry["error"] == "Chroma vector index corrupted"
    assert entry["answer"] == ""

