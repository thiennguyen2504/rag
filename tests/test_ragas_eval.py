"""Offline unit tests for RAGAS evaluation module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eval.ragas_eval import (
    build_ragas_dataset,
    compare_runs,
    evaluate_run,
    load_chunks_lookup,
    load_eval_set,
    load_run_entries,
)


@pytest.fixture
def mock_data(tmp_path: Path):
    """Create sample mock data for offline tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Mock chunk file
    chunks_file = data_dir / "chunks_A.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "c1", "text": "Nội dung quy định độ tuổi mở ví MoMo từ 15 tuổi.", "doc_id": "doc1"}) + "\n")
        f.write(json.dumps({"chunk_id": "c2", "text": "Hạn mức nạp tiền 50 triệu/ngày.", "doc_id": "doc2"}) + "\n")

    # Mock eval set
    eval_set_file = data_dir / "eval_set.jsonl"
    with open(eval_set_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "q_001",
            "question": "15 tuổi có được mở MoMo không?",
            "gold_answer": "Có, từ đủ 15 tuổi được mở ví.",
            "gold_evidence": ["Nội dung quy định độ tuổi mở ví MoMo từ 15 tuổi."],
        }) + "\n")
        f.write(json.dumps({
            "id": "q_002",
            "question": "Hạn mức nạp tiền là bao nhiêu?",
            "gold_answer": "Hạn mức tối đa 50 triệu.",
            "gold_evidence": ["Hạn mức nạp tiền 50 triệu/ngày."],
        }) + "\n")

    # Mock runs log
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    runs_file = logs_dir / "runs.jsonl"
    with open(runs_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": "run_test_1",
            "question_id": "q_001",
            "question": "15 tuổi có được mở MoMo không?",
            "answer": "Có thể mở tài khoản từ 15 tuổi.",
            "retrieved": [{"chunk_id": "c1", "score": 0.9}],
        }) + "\n")
        f.write(json.dumps({
            "run_id": "run_test_1",
            "question_id": "q_002",
            "question": "Hạn mức nạp tiền là bao nhiêu?",
            "answer": "Tối đa 50 triệu/ngày.",
            "retrieved": [{"chunk_id": "c2", "score": 0.85}],
        }) + "\n")
        # Run 2 for comparison
        f.write(json.dumps({
            "run_id": "run_test_2",
            "question_id": "q_001",
            "question": "15 tuổi có được mở MoMo không?",
            "answer": "Được mở ví từ 15 tuổi theo quy định.",
            "retrieved": [{"chunk_id": "c1", "score": 0.95}],
        }) + "\n")
        f.write(json.dumps({
            "run_id": "run_test_2",
            "question_id": "q_002",
            "question": "Hạn mức nạp tiền là bao nhiêu?",
            "answer": "Hạn mức 50 triệu một ngày.",
            "retrieved": [{"chunk_id": "c2", "score": 0.9}],
        }) + "\n")

    return {
        "data_dir": data_dir,
        "eval_set_path": eval_set_file,
        "runs_file": runs_file,
        "output_dir": tmp_path / "results",
    }


def test_load_chunks_lookup(mock_data):
    lookup = load_chunks_lookup(mock_data["data_dir"])
    assert "c1" in lookup
    assert "c2" in lookup
    assert "từ 15 tuổi" in lookup["c1"]["text"]


def test_load_eval_set(mock_data):
    eval_set = load_eval_set(mock_data["eval_set_path"])
    assert "q_001" in eval_set
    assert "q_002" in eval_set
    assert eval_set["q_001"]["gold_answer"] == "Có, từ đủ 15 tuổi được mở ví."


def test_load_run_entries(mock_data):
    entries = load_run_entries(mock_data["runs_file"], "run_test_1")
    assert len(entries) == 2
    assert entries[0]["question_id"] == "q_001"
    assert entries[1]["question_id"] == "q_002"


def test_build_ragas_dataset(mock_data):
    lookup = load_chunks_lookup(mock_data["data_dir"])
    eval_set = load_eval_set(mock_data["eval_set_path"])
    entries = load_run_entries(mock_data["runs_file"], "run_test_1")

    dataset, metadata = build_ragas_dataset(entries, eval_set, lookup)
    assert len(dataset) == 2
    assert len(metadata) == 2

    # Check sample 0
    s0 = dataset.samples[0]
    assert s0.user_input == "15 tuổi có được mở MoMo không?"
    assert s0.response == "Có thể mở tài khoản từ 15 tuổi."
    assert len(s0.retrieved_contexts) == 1
    assert "15 tuổi" in s0.retrieved_contexts[0]
    assert s0.reference == "Có, từ đủ 15 tuổi được mở ví."


@patch("eval.ragas_eval.evaluate")
def test_evaluate_run_offline(mock_ragas_eval, mock_data):
    # Mock ragas evaluate return value
    mock_result = MagicMock()
    df_mock = pd.DataFrame([
        {
            "faithfulness": 0.95,
            "answer_relevancy": 0.90,
            "context_precision": 0.85,
            "context_recall": 1.0,
        },
        {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
            "context_recall": 0.9,
        },
    ])
    mock_result.to_pandas.return_value = df_mock
    mock_ragas_eval.return_value = mock_result

    # Mock evaluators so no API calls happen
    dummy_llm = MagicMock()
    dummy_embeddings = MagicMock()

    res = evaluate_run(
        run_id="run_test_1",
        runs_log_path=mock_data["runs_file"],
        eval_set_path=mock_data["eval_set_path"],
        data_dir=mock_data["data_dir"],
        output_dir=mock_data["output_dir"] / "run_test_1",
        llm=dummy_llm,
        embeddings=dummy_embeddings,
        metrics=[MagicMock(), MagicMock()],
        local_mode=False,
    )

    summary = res["summary"]
    assert summary["run_id"] == "run_test_1"
    assert summary["total_questions"] == 2
    assert summary["faithfulness"] == 0.90  # average of 0.95 and 0.85
    assert summary["answer_relevancy"] == 0.85  # average of 0.90 and 0.80
    assert summary["context_precision"] == 0.80  # average of 0.85 and 0.75
    assert summary["context_recall"] == 0.95  # average of 1.0 and 0.9

    # Verify output files exist
    assert Path(res["summary_path"]).exists()
    assert Path(res["markdown_path"]).exists()

    with open(res["summary_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["faithfulness"] == 0.9


def test_evaluate_run_local_mode(mock_data):
    """Test fast local evaluation mode without any mocks."""
    res = evaluate_run(
        run_id="run_test_1",
        runs_log_path=mock_data["runs_file"],
        eval_set_path=mock_data["eval_set_path"],
        data_dir=mock_data["data_dir"],
        output_dir=mock_data["output_dir"] / "run_test_1_local",
        local_mode=True,
    )

    summary = res["summary"]
    assert summary["run_id"] == "run_test_1"
    assert summary["total_questions"] == 2
    assert 0.0 <= summary["faithfulness"] <= 1.0
    assert 0.0 <= summary["answer_relevancy"] <= 1.0
    assert 0.0 <= summary["context_precision"] <= 1.0
    assert 0.0 <= summary["context_recall"] <= 1.0


@patch("eval.ragas_eval.evaluate_run")
def test_compare_runs_offline(mock_eval_run, mock_data):
    # Mock evaluate_run for run_test_1 and run_test_2
    mock_eval_run.side_effect = [
        {
            "summary": {
                "run_id": "run_test_1",
                "faithfulness": 0.80,
                "answer_relevancy": 0.70,
                "context_precision": 0.75,
                "context_recall": 0.60,
            },
            "details": [],
        },
        {
            "summary": {
                "run_id": "run_test_2",
                "faithfulness": 0.90,
                "answer_relevancy": 0.85,
                "context_precision": 0.80,
                "context_recall": 0.90,
            },
            "details": [],
        },
    ]

    comp = compare_runs(
        run_id_a="run_test_1",
        run_id_b="run_test_2",
        output_dir=mock_data["output_dir"],
        runs_log_path=mock_data["runs_file"],
        eval_set_path=mock_data["eval_set_path"],
        data_dir=mock_data["data_dir"],
    )

    diff = comp["metrics_diff"]
    assert diff["faithfulness"]["delta"] == 0.10
    assert diff["faithfulness"]["percent_change"] == 12.5
    assert diff["context_recall"]["delta"] == 0.30
    assert diff["context_recall"]["percent_change"] == 50.0

    comp_md = mock_data["output_dir"] / "ragas_comparison_run_test_1_vs_run_test_2.md"
    assert comp_md.exists()

