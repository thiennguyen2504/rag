"""Retriever module for querying Chroma collections with caching and similarity threshold filtering."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_DIR, get_api_key
from src.llm_utils import retry_with_backoff

logger = logging.getLogger(__name__)

# Module-level cache for Chroma vector store instances: (collection_name, embedding_model) -> Chroma
_CHROMA_CACHE: Dict[Tuple[str, str], Chroma] = {}


def get_cached_chroma(collection_name: str, embedding_model: str) -> Chroma:
    """Retrieve or create and cache a Chroma vector store instance."""
    cache_key = (collection_name, embedding_model)
    if cache_key not in _CHROMA_CACHE:
        api_key = get_api_key()
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
        )
        logger.info("Khởi tạo kết nối Chroma mới cho collection '%s' với model '%s'", collection_name, embedding_model)
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
        )
        _CHROMA_CACHE[cache_key] = vector_store

    return _CHROMA_CACHE[cache_key]


class Retriever:
    """Retriever for Chroma collection with cosine similarity and threshold filtering."""

    def __init__(
        self,
        collection: str,
        embedding_model: str,
        top_k: int = 3,
        similarity_threshold: Optional[float] = None,
    ):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_store = get_cached_chroma(self.collection, self.embedding_model)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chunks from Chroma, convert cosine distance to similarity, and filter by threshold.

        Returns:
            List of dicts: {chunk_id, doc_id, title, source_url, section, text, score}
            sorted by score descending.
        """
        k = top_k if top_k is not None else self.top_k
        raw_results = self.vector_store.similarity_search_with_score(query, k=k)

        items: List[Dict[str, Any]] = []
        for doc, dist in raw_results:
            similarity = 1.0 - float(dist)

            # Filter by similarity_threshold if set
            if self.similarity_threshold is not None and similarity < self.similarity_threshold:
                continue

            metadata = doc.metadata or {}
            items.append({
                "chunk_id": metadata.get("chunk_id", ""),
                "doc_id": metadata.get("doc_id", ""),
                "title": metadata.get("title", ""),
                "source_url": metadata.get("source_url", ""),
                "section": metadata.get("section", ""),
                "text": doc.page_content,
                "score": round(similarity, 4),
            })

        # Sort by score descending
        items.sort(key=lambda x: x["score"], reverse=True)
        return items


# Backwards compatibility alias
MoMoRetriever = Retriever
