"""
Recursive Character Text Splitter
==================================

Splits long document text into smaller, overlapping chunks. This ensures that
retrieved RAG context is focused, fits cleanly in the LLM's context window,
and avoids overloading the local generation model.
"""
from typing import List


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively by trying a list of separators (e.g. paragraphs,
    newlines, spaces) until chunks are smaller than ``chunk_size``.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Try splitting by paragraph, then single newline, then space
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Split text into a list of strings within chunk_size, retaining overlap.
        """
        if not text:
            return []

        raw_splits = []
        self._split_recursive(text, self.separators, raw_splits)
        return self._merge_splits(raw_splits)

    def _split_recursive(self, text: str, separators: List[str], splits: List[str]) -> None:
        if len(text) <= self.chunk_size:
            splits.append(text)
            return

        if not separators:
            # Fallback: force chunking by character index if no separators left
            for i in range(0, len(text), self.chunk_size):
                splits.append(text[i : i + self.chunk_size])
            return

        separator = separators[0]
        parts = text.split(separator)

        # If splitting by this separator yielded multiple parts, process each
        if len(parts) > 1:
            for part in parts:
                if len(part) <= self.chunk_size:
                    splits.append(part)
                else:
                    self._split_recursive(part, separators[1:], splits)
        else:
            self._split_recursive(text, separators[1:], splits)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        merged_chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            clean_split = split.strip()
            if not clean_split:
                continue

            split_len = len(clean_split)

            # If adding this split exceeds chunk_size, save current_chunk and roll back overlap
            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    merged_chunks.append(" ".join(current_chunk))

                    # Retain overlap context
                    overlap_items = []
                    overlap_length = 0
                    for prev in reversed(current_chunk):
                        if overlap_length + len(prev) < self.chunk_overlap:
                            overlap_items.insert(0, prev)
                            overlap_length += len(prev)
                        else:
                            break
                    current_chunk = overlap_items
                    current_length = overlap_length

            current_chunk.append(clean_split)
            current_length += split_len

        if current_chunk:
            merged_chunks.append(" ".join(current_chunk))

        return merged_chunks
