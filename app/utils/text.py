"""Text chunking utilities for document preparation."""

from __future__ import annotations


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[str]:
    """Split *text* into overlapping chunks of roughly *max_chars* characters.

    Uses paragraph boundaries when possible, falling back to sentence then
    hard character splits.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # try to split at paragraph boundary
        split_pos = text.rfind("\n\n", start, end)
        if split_pos == -1 or split_pos <= start:
            # try sentence boundary
            split_pos = text.rfind(". ", start, end)
        if split_pos == -1 or split_pos <= start:
            # hard split
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        start = max(split_pos - overlap, start + 1)

    return chunks
