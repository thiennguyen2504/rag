"""Ingest chunks into Chroma vector store with idempotency, batching, and retry."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_DIR, DATA_DIR, get_api_key, load_models_config
from src.llm_utils import retry_with_backoff

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest")

MANIFEST_PATH = CHROMA_DIR / "ingest_manifest.json"


def load_chunks_file(strategy: str) -> List[Dict[str, Any]]:
    """Load chunks from data/chunks_{A,B}.jsonl."""
    file_path = DATA_DIR / f"chunks_{strategy}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {file_path}. Vui lòng chạy 'python -m src.chunking' trước."
        )

    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))
    return chunks


def load_ingest_manifest() -> Dict[str, Any]:
    """Load chroma_db/ingest_manifest.json if exists."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_ingest_manifest(embedding_model: str, collections_info: Dict[str, int]) -> None:
    """Save chroma_db/ingest_manifest.json."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "embedding_model": embedding_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collections": collections_info,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


@retry_with_backoff(max_retries=5, initial_delay=2.0)
def ingest_batch(
    vector_store: Chroma,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    """Ingest a single batch of documents into Chroma with retry."""
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def ingest_strategy(
    strategy: str,
    embedding_function: Any,
    batch_size: int = 50,
    rebuild: bool = False,
) -> int:
    """Ingest chunks for a given strategy ('A' or 'B')."""
    collection_name = f"momo_{strategy}"
    chunks = load_chunks_file(strategy)
    logger.info("Chiến lược %s: Tìm thấy %d chunks trong file jsonl.", strategy, len(chunks))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if rebuild:
        logger.info("Tùy chọn --rebuild được bật: Xóa collection %s và tạo lại...", collection_name)
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # Idempotency check: retrieve existing chunk_ids
    existing_data = vector_store.get()
    existing_ids = set(existing_data.get("ids", []))
    logger.info("Collection %s hiện có %d chunks đã được index.", collection_name, len(existing_ids))

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"[OK] Collection '{collection_name}': Toàn bộ {len(chunks)} chunks đã tồn tại. Không cần embed lại.")
    else:
        print(f"Đang index {len(new_chunks)} chunks mới vào '{collection_name}' (batch_size={batch_size})...")

        for i in tqdm(range(0, len(new_chunks), batch_size), desc=f"Ingest {collection_name}"):
            batch = new_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "source_url": c["source_url"],
                    "section": c["section"],
                    "strategy": c["strategy"],
                }
                for c in batch
            ]

            ingest_batch(vector_store, texts, metadatas, ids)
            time.sleep(0.5)  # brief pause between batches to respect rate limits

    # Verify count
    final_data = vector_store.get()
    final_count = len(final_data.get("ids", []))
    if final_count != len(chunks):
        raise RuntimeError(
            f"Lỗi lệch số lượng chunk cho {collection_name}: "
            f"Collection có {final_count} chunks nhưng file jsonl có {len(chunks)} chunks!"
        )

    print(f"[OK] Collection '{collection_name}' hoàn tất: {final_count}/{len(chunks)} chunks.")
    return final_count


def run_ingest(
    rebuild: bool = False,
    only_strategy: Optional[str] = None,
    batch_size: int = 50,
) -> None:
    """Main ingestion coordinator."""
    # 1. Config and API key check
    api_key = get_api_key()
    models_cfg = load_models_config()
    embedding_model = models_cfg["embedding_model"]

    # 2. Check manifest compatibility
    manifest = load_ingest_manifest()
    if manifest and manifest.get("embedding_model") != embedding_model:
        if not rebuild:
            raise ValueError(
                f"Embedding model hiện tại ('{embedding_model}') khác với model đã lưu trong "
                f"ingest_manifest.json ('{manifest.get('embedding_model')}'). "
                f"Vector không tương thích. Vui lòng chạy lại với cờ --rebuild để xóa và tạo lại index."
            )

    # 3. Initialize embeddings
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key,
    )

    collections_info: Dict[str, int] = manifest.get("collections", {})

    strategies = ["A", "B"] if not only_strategy else [only_strategy.upper()]

    for strat in strategies:
        if strat not in ["A", "B"]:
            raise ValueError(f"Chiến lược không hợp lệ: '{strat}'. Chỉ chấp nhận 'A' hoặc 'B'.")
        count = ingest_strategy(
            strategy=strat,
            embedding_function=embedding_function,
            batch_size=batch_size,
            rebuild=rebuild,
        )
        collections_info[f"momo_{strat}"] = count

    # Save manifest
    save_ingest_manifest(embedding_model, collections_info)
    print(f"\n[OK] Đã cập nhật thành công {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest chunks into Chroma vector database.")
    parser.add_argument("--rebuild", action="store_true", help="Xóa collection hiện tại và tạo lại từ đầu")
    parser.add_argument("--only", type=str, choices=["A", "B", "a", "b"], default=None, help="Chỉ ingest một strategy cụ thể (A hoặc B)")
    parser.add_argument("--batch-size", type=int, default=50, help="Số lượng chunks mỗi batch (mặc định 50)")
    args = parser.parse_args()

    print("=" * 80)
    print("BẮT ĐẦU INGEST VÀO CHROMA (src/ingest.py)")
    print("=" * 80)

    try:
        run_ingest(
            rebuild=args.rebuild,
            only_strategy=args.only,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"\n[LỖI INGEST] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
