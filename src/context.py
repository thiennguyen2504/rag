"""Context assembly module for formatting retrieved chunks and enforcing token/character bounds."""

from typing import Any, Dict, List, Set, Tuple


def build_context(
    chunks: List[Dict[str, Any]],
    max_context_chars: int = 6000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build formatted context string and return the list of included sources.

    Rules:
    - Deduplicates chunks with identical text content.
    - Format per source: "[{i}] ({title} - {section})\\n{text}", separated by a blank line.
      If section is empty, formats as "[{i}] ({title})\\n{text}".
    - Numbered from 1 (1-indexed).
    - Stops adding chunks when adding the chunk would exceed max_context_chars (never cuts mid-chunk).
    - Returns (context_str, used_sources), where each item in used_sources has 'index' matching {i}.

    Args:
        chunks: List of chunk dictionaries from retriever.
        max_context_chars: Maximum character limit for context_str.

    Returns:
        Tuple of (context_str, used_sources).
    """
    if not chunks:
        return "", []

    seen_texts: Set[str] = set()
    used_sources: List[Dict[str, Any]] = []
    context_blocks: List[str] = []
    current_char_count = 0

    idx = 1
    for c in chunks:
        text_content = c.get("text", "").strip()
        if not text_content:
            continue

        # Deduplicate identical text
        if text_content in seen_texts:
            continue

        title = c.get("title", "").strip()
        section = c.get("section", "").strip()
        header_desc = f"{title} - {section}" if section else title
        block = f"[{idx}] ({header_desc})\n{text_content}"

        # Calculate length if added (account for "\n\n" separator)
        added_len = len(block) if not context_blocks else 2 + len(block)

        if current_char_count + added_len > max_context_chars:
            # Stop adding further chunks when exceeding limit (do not cut mid-chunk)
            break

        seen_texts.add(text_content)
        context_blocks.append(block)
        current_char_count += added_len

        # Record used source with exact matching index
        source_entry = {
            "index": idx,
            "chunk_id": c.get("chunk_id", ""),
            "doc_id": c.get("doc_id", ""),
            "title": c.get("title", ""),
            "source_url": c.get("source_url", ""),
            "score": c.get("score", 0.0),
            "text": text_content,
        }
        if "section" in c:
            source_entry["section"] = c["section"]

        used_sources.append(source_entry)
        idx += 1

    context_str = "\n\n".join(context_blocks)
    return context_str, used_sources
