"""Check Gemini API connectivity for configured models."""

import sys
from typing import Any, Dict
from src.config import get_api_key, load_models_config
from src.llm_utils import retry_with_backoff

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def test_generation_model(model_name: str, api_key: str, prompt: str) -> Dict[str, Any]:
    """Test text generation with ChatGoogleGenerativeAI."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
    )
    response = llm.invoke(prompt)
    return {
        "content": response.content,
        "usage_metadata": response.usage_metadata or {},
    }


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def test_embedding_model(model_name: str, api_key: str, text: str) -> int:
    """Test text embedding with GoogleGenerativeAIEmbeddings."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # GoogleGenerativeAIEmbeddings accepts either "models/..." or raw name
    emb = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )
    vector = emb.embed_query(text)
    return len(vector)


def main() -> None:
    print("=" * 80)
    print("KIỂM TRA KẾT NỐI API GEMINI (src/check_api.py)")
    print("=" * 80)

    try:
        api_key = get_api_key()
    except Exception as exc:
        print(f"[FAIL] Lỗi API Key: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        models_cfg = load_models_config()
    except Exception as exc:
        print(f"[FAIL] Lỗi file cấu hình models.yaml: {exc}", file=sys.stderr)
        sys.exit(1)

    gen_model = models_cfg["generator_model"]
    judge_model = models_cfg["judge_model"]
    embed_model = models_cfg["embedding_model"]

    results = {}

    # 1. Generator Model
    print(f"\n1. Đang kiểm tra Generator Model: '{gen_model}'...")
    try:
        res = test_generation_model(gen_model, api_key, "Trả lời ngắn gọn: Bạn là ai?")
        content_preview = str(res["content"]).strip().replace("\n", " ")[:100]
        print(f"   [OK] Phản hồi: {content_preview}...")
        print(f"   [OK] Usage metadata: {res['usage_metadata']}")
        results["generator"] = True
    except Exception as exc:
        print(f"   [FAIL] Generator Model gặp lỗi: {exc}")
        results["generator"] = False

    # 2. Judge Model
    print(f"\n2. Đang kiểm tra Judge Model: '{judge_model}'...")
    try:
        res = test_generation_model(judge_model, api_key, "Trả lời ngắn gọn: Sẵn sàng đánh giá không?")
        content_preview = str(res["content"]).strip().replace("\n", " ")[:100]
        print(f"   [OK] Phản hồi: {content_preview}...")
        print(f"   [OK] Usage metadata: {res['usage_metadata']}")
        results["judge"] = True
    except Exception as exc:
        print(f"   [FAIL] Judge Model gặp lỗi: {exc}")
        results["judge"] = False

    # 3. Embedding Model
    print(f"\n3. Đang kiểm tra Embedding Model: '{embed_model}'...")
    try:
        dim = test_embedding_model(embed_model, api_key, "Kiểm tra vector embedding tiếng Việt.")
        print(f"   [OK] Vector dimension: {dim}")
        results["embedding"] = True
    except Exception as exc:
        print(f"   [FAIL] Embedding Model gặp lỗi: {exc}")
        results["embedding"] = False

    # Summary
    print("\n" + "=" * 80)
    print("TỔNG KẾT KIỂM TRA API:")
    all_ok = True
    for part, ok in results.items():
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  - {part.capitalize()}: [{status}]")
    print("=" * 80)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
