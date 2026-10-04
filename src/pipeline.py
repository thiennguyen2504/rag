"""End-to-end RAG Q&A Pipeline adhering to project contracts with metrics and caching."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import BASE_DIR, get_api_key, load_models_cfg, load_pipeline_cfg
from src.context import build_context
from src.llm_utils import compute_cost, extract_usage_metadata, retry_with_backoff
from src.retriever import Retriever

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# Module-level cache for pipeline instances: config_id -> RAGPipeline
_PIPELINE_CACHE: Dict[str, "RAGPipeline"] = {}


class RAGPipeline:
    """Orchestrates retrieval, context preparation, prompt compilation, and LLM answer generation."""

    def __init__(self, config_id: str):
        self.config_id = config_id
        self.config = load_pipeline_cfg(config_id)
        self.models_cfg = load_models_cfg()

        # Initialize Retriever with collection and settings from config
        self.retriever = Retriever(
            collection=self.config.collection,
            embedding_model=self.models_cfg.embedding_model,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
        )

        # Generator Model configuration
        self.model_name = self.config.llm.model or self.models_cfg.generator_model
        self.temperature = self.config.llm.temperature
        self.max_output_tokens = self.config.llm.max_output_tokens

        # Load prompt template
        prompt_path = BASE_DIR / self.config.prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file prompt: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        api_key = get_api_key()
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _invoke_llm(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)

    def ask(self, question: str) -> Dict[str, Any]:
        """Execute RAG pipeline on a question with separate retrieval and generation timing."""
        total_start = time.perf_counter()
        retrieval_ms = 0.0
        generation_ms = 0.0
        final_prompt = ""
        used_sources: List[Dict[str, Any]] = []

        try:
            # 1. Retrieval
            ret_start = time.perf_counter()
            retrieved_chunks = self.retriever.retrieve(question)
            retrieval_ms = (time.perf_counter() - ret_start) * 1000.0

            # 2. Context Building & Deduplication
            context_str, used_sources = build_context(
                retrieved_chunks,
                max_context_chars=self.config.max_context_chars,
            )

            # Render final prompt
            final_prompt = self.prompt_template.format(context=context_str, question=question)

            # Short-circuit check
            if self.config.short_circuit_on_empty_context and not used_sources:
                total_ms = (time.perf_counter() - total_start) * 1000.0
                return {
                    "question": question,
                    "config_id": self.config_id,
                    "answer": "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.",
                    "sources": [],
                    "final_prompt": final_prompt,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_usd": 0.0,
                    "latency_ms": {
                        "retrieval": round(retrieval_ms, 2),
                        "generation": 0.0,
                        "total": round(total_ms, 2),
                    },
                    "error": None,
                }

            # 3. Generation
            gen_start = time.perf_counter()
            response = self._invoke_llm(final_prompt)
            generation_ms = (time.perf_counter() - gen_start) * 1000.0

            answer = str(response.content).strip()
            usage = extract_usage_metadata(response)
            cost_usd = compute_cost(
                model=self.model_name,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                models_cfg=self.models_cfg,
            )

            total_ms = (time.perf_counter() - total_start) * 1000.0

            return {
                "question": question,
                "config_id": self.config_id,
                "answer": answer,
                "sources": used_sources,
                "final_prompt": final_prompt,
                "usage": usage,
                "cost_usd": cost_usd,
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "generation": round(generation_ms, 2),
                    "total": round(total_ms, 2),
                },
                "error": None,
            }

        except Exception as exc:
            logger.error("Lỗi trong pipeline %s: %s", self.config_id, exc, exc_info=True)
            total_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "question": question,
                "config_id": self.config_id,
                "answer": "",
                "sources": used_sources,
                "final_prompt": final_prompt,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "cost_usd": 0.0,
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "generation": round(generation_ms, 2),
                    "total": round(total_ms, 2),
                },
                "error": str(exc),
            }


def get_pipeline(config_id: str) -> RAGPipeline:
    """Retrieve or create and cache a RAGPipeline instance for config_id."""
    if config_id not in _PIPELINE_CACHE:
        logger.info("Khởi tạo pipeline mới cho cấu hình '%s'", config_id)
        _PIPELINE_CACHE[config_id] = RAGPipeline(config_id)
    return _PIPELINE_CACHE[config_id]


def ask(question: str, config_id: str = "A") -> Dict[str, Any]:
    """Top-level convenience function fulfilling the project pipeline contract."""
    pipeline = get_pipeline(config_id)
    return pipeline.ask(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="MoMo RAG Pipeline CLI.")
    parser.add_argument("--config", type=str, default="A", help="ID cấu hình pipeline ('A', 'B')")
    parser.add_argument("--q", "--question", dest="question", type=str, required=True, help="Câu hỏi cần giải đáp")
    args = parser.parse_args()

    result = ask(question=args.question, config_id=args.config)

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ HỎI ĐÁP RAG (Cấu hình: {args.config})")
    print("=" * 80)
    print(f"Câu hỏi: {result['question']}")

    if result.get("error"):
        print(f"\n[LỖI]: {result['error']}")
    else:
        print(f"\n[CÂU TRẢ LỜI]:\n{result['answer']}")

    print("\n" + "-" * 80)
    print("NGUỒN THAM KHẢO (SOURCES):")
    sources = result.get("sources", [])
    if not sources:
        print("  (Không có nguồn tham khảo nào được trích xuất)")
    else:
        for s in sources:
            idx = s.get("index", 1)
            score = s.get("score", 0.0)
            chunk_id = s.get("chunk_id", "N/A")
            title = s.get("title", "")
            preview = s.get("text", "").replace("\n", " ")[:150]
            print(f"  [{idx}] (Score: {score:.4f}) [{chunk_id}] {title}")
            print(f"      Trích đoạn: {preview}...")

    print("-" * 80)
    latency = result.get("latency_ms", {})
    usage = result.get("usage", {})
    cost = result.get("cost_usd", 0.0)
    print(f"ĐỘ TRỄ (Latency): Truy xuất: {latency.get('retrieval', 0.0):.1f}ms | Sinh câu trả lời: {latency.get('generation', 0.0):.1f}ms | Tổng: {latency.get('total', 0.0):.1f}ms")
    print(f"TOKENS         : Input: {usage.get('input_tokens', 0)} | Output: {usage.get('output_tokens', 0)} | Tổng: {usage.get('total_tokens', 0)}")
    print(f"CHI PHÍ (Cost) : ${cost:.6f} USD")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
