"""Promptfoo Python Provider for MoMo RAG Q&A.

Allows Promptfoo to execute and test different RAG configurations (A, B, C)
and capture responses, token usage, cost, and latency for automated testing.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import ask

logger = logging.getLogger("promptfoo_provider")


def call_rag_pipeline(
    prompt: str,
    config_id: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute MoMo RAG pipeline for a given configuration and return Promptfoo result format."""
    query = prompt
    if context and "vars" in context:
        query = context["vars"].get("question") or context["vars"].get("query") or prompt

    try:
        result = ask(
            query=query,
            config_id=config_id,
            skip_log=True,
        )

        answer = result.get("answer", "")
        cost = result.get("cost_usd", 0.0)
        usage = result.get("usage", {})
        latency = result.get("latency_ms", {})
        retrieved = result.get("retrieved", [])

        return {
            "output": answer,
            "cost": cost,
            "tokenUsage": {
                "total": usage.get("total_tokens", 0),
                "prompt": usage.get("input_tokens", 0),
                "completion": usage.get("output_tokens", 0),
            },
            "metadata": {
                "config_id": config_id,
                "retrieved_chunk_ids": [r.get("chunk_id") for r in retrieved],
                "retrieved_docs": list({r.get("doc_id") for r in retrieved if r.get("doc_id")}),
                "latency_ms": latency,
            },
        }
    except Exception as e:
        logger.error("Lỗi khi chạy RAG Config %s: %s", config_id, e, exc_info=True)
        return {
            "error": str(e),
            "output": f"Lỗi hệ thống: {e}",
        }


def call_rag_A(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Promptfoo provider entrypoint for Config A."""
    return call_rag_pipeline(prompt, config_id="A", options=options, context=context)


def call_rag_B(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Promptfoo provider entrypoint for Config B."""
    return call_rag_pipeline(prompt, config_id="B", options=options, context=context)


def call_rag_C(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Promptfoo provider entrypoint for Config C."""
    return call_rag_pipeline(prompt, config_id="C", options=options, context=context)


def call_api(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Default provider: routes based on config_id in context vars or defaults to B."""
    config_id = "B"
    if context and "vars" in context and "config_id" in context["vars"]:
        config_id = context["vars"]["config_id"]
    return call_rag_pipeline(prompt, config_id=config_id, options=options, context=context)

