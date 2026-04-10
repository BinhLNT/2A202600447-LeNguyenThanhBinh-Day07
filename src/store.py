from __future__ import annotations
from typing import Any, Callable
from .chunking import _dot, compute_similarity, RecursiveChunker
from .embeddings import _mock_embed
from .models import Document

class EmbeddingStore:
    """
    A vector store for text chunks.
    Tries to use ChromaDB if available; falls back to an in-memory store.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None

        try:
            import chromadb
            # Khởi tạo ephemeral client (lưu tạm thời trong RAM)
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document, chunk_text: str, index: int) -> dict[str, Any]:
        """Tạo một bản ghi chuẩn hóa để lưu trữ."""
        return {
            "id": f"{doc.id}_chunk_{index}",
            "content": chunk_text,
            "embedding": self._embedding_fn(chunk_text),
            "metadata": {**doc.metadata, "doc_id": doc.id}
        }

    def add_documents(self, docs: list[Document]) -> None:
        """Chia nhỏ tài liệu, tạo embedding và lưu vào store."""
        chunker = RecursiveChunker(chunk_size=500)
        
        for doc in docs:
            chunks = chunker.chunk(doc.content)
            
            if self._use_chroma:
                ids = [f"{doc.id}_{i}" for i in range(len(chunks))]
                embeddings = [self._embedding_fn(c) for c in chunks]
                metadatas = [{**doc.metadata, "doc_id": doc.id} for _ in chunks]
                self._collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                for i, chunk_text in enumerate(chunks):
                    record = self._make_record(doc, chunk_text, i)
                    self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Tìm kiếm top_k đoạn văn bản tương đồng nhất."""
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            # Format lại kết quả cho đồng nhất
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1.0 - results['distances'][0][i] # Chuyển distance thành similarity
                })
            return formatted
        else:
            query_vec = self._embedding_fn(query)
            scored_records = []
            for record in self._store:
                score = compute_similarity(query_vec, record["embedding"])
                scored_records.append({**record, "score": score})
            
            return sorted(scored_records, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_collection_size(self) -> int:
        """Trả về tổng số lượng chunk đang lưu trữ."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Tìm kiếm có kèm theo lọc metadata (Pre-filtering)."""
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            # Format tương tự hàm search
            return [{"content": d, "metadata": m, "score": 1.0 - dist} 
                    for d, m, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]
        else:
            # Lọc trước (Pre-filtering) cho bộ nhớ in-memory
            filtered_records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            
            query_vec = self._embedding_fn(query)
            scored = []
            for r in filtered_records:
                score = compute_similarity(query_vec, r["embedding"])
                scored.append({**r, "score": score})
            
            return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """Xóa toàn bộ các chunk thuộc về một doc_id."""
        if self._use_chroma:
            count_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < count_before
        else:
            initial_size = len(self._store)
            self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_size