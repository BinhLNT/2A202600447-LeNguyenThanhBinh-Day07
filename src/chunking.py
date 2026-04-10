from __future__ import annotations

import math
import re

# Định nghĩa hàm _dot để hỗ trợ tính similarity
def _dot(a: list[float], b: list[float]) -> float:
    """Tính tích vô hướng của hai vector."""
    return sum(x * y for x, y in zip(a, b))

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    """
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Tách câu dựa trên dấu chấm, hỏi, cảm thán theo sau là khoảng trắng
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.
    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text]

        sep = remaining_separators[0]
        # Nếu separator là rỗng "", cắt theo ký tự
        parts = current_text.split(sep) if sep != "" else list(current_text)
        
        final_chunks = []
        buffer = ""

        for part in parts:
            join_sep = sep if buffer else ""
            if len(buffer) + len(join_sep) + len(part) <= self.chunk_size:
                buffer = buffer + join_sep + part
            else:
                if buffer:
                    final_chunks.append(buffer)
                
                if len(part) > self.chunk_size:
                    final_chunks.extend(self._split(part, remaining_separators[1:]))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            final_chunks.append(buffer)
        return final_chunks

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_prod / (norm_a * norm_b)

class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        for name, chunker in [
            ("fixed_size", FixedSizeChunker(chunk_size=chunk_size)),
            ("by_sentences", SentenceChunker(max_sentences_per_chunk=3)),
            ("recursive", RecursiveChunker(chunk_size=chunk_size))
        ]:
            chunks = chunker.chunk(text)
            results[name] = {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
        return results