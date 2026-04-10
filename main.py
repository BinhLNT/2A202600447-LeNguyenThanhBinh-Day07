from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    GeminiEmbedder, # Thêm Gemini vào đây
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# 1. Cập nhật danh sách file truyện cổ tích của nhóm
# main.py
FAIRY_TALE_FILES = [
    "data/sodua.txt",
    "data/thachsanh.txt",
    "data/hoguom.txt",
    "data/nguulangchucnu.txt",
    "data/caykhe.txt",
]

def get_metadata_for_story(file_stem: str) -> dict:
    """Tự động tạo metadata dựa trên tên file truyện để phục vụ filtering."""
    metadata_map = {
        "sodua": {
            "story_title": ["Sọ Dừa"], 
            "story_type": "cổ tích", 
            "origin": "Việt Nam",
            "main_characters": ["Sọ Dừa", "Phú Ông", "cô út"],
            "themes": ["phép thuật", "tình yêu"]
        },
        "thachsanh": {
            "story_title": ["Thạch Sanh"], 
            "story_type": "cổ tích", 
            "origin": "Việt Nam",
            "main_characters": ["Thạch Sanh", "Lý Thông", "công chúa"],
            "themes": ["anh hùng", "thiện ác"]
        },
        "hoguom": {
            "story_title": ["Sự tích Hồ Gươm"], 
            "story_type": "truyền thuyết", 
            "origin": "Việt Nam",
            "main_characters": ["Lê Lợi", "Rùa Vàng"],
            "themes": ["lịch sử", "yêu nước"]
        },
        "nguulangchucnu": {
            "story_title": ["Ngưu Lang Chức Nữ"], 
            "story_type": "truyền thuyết", 
            "origin": "Trung Quốc",
            "main_characters": ["Ngưu Lang", "Chức Nữ"],
            "themes": ["tình yêu", "chia ly"]
        },
        "caykhe": {
            "story_title": ["Cây Khế"], 
            "story_type": "cổ tích", 
            "origin": "Việt Nam",
            "main_characters": ["người anh", "người em"],
            "themes": ["tham lam", "thiện ác"]
        }
    }
    return metadata_map.get(file_stem.lower(), {"category": "general"})

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        # Gán metadata chi tiết cho từng truyện
        metadata = get_metadata_for_story(path.stem)
        metadata["source"] = str(path)
        
        documents.append(
            Document(id=path.stem, content=content, metadata=metadata)
        )
    return documents

def real_llm_call(prompt: str) -> str:
    """Nếu em có API key cho Gemini LLM, hãy tích hợp vào đây. 
    Nếu không, hàm này sẽ trả về prompt để em xem AI nhận context gì."""
    return "[AGENT] Câu trả lời sẽ được tạo dựa trên context truyện đã tìm thấy."

def run_manual_demo(question: str | None = None):
    # Khởi tạo môi trường
    load_dotenv(override=False)
    
    print("=== Hệ thống RAG Truyện Cổ Tích Việt Nam ===")
    docs = load_documents_from_files(FAIRY_TALE_FILES)
    
    # Lựa chọn Embedder
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "gemini":
        embedder = GeminiEmbedder()
    elif provider == "local":
        embedder = LocalEmbedder()
    else:
        embedder = _mock_embed

    print(f"Sử dụng backend: {getattr(embedder, '_backend_name', 'Mock')}")

    # Khởi tạo Store và nạp dữ liệu
    store = EmbeddingStore(embedding_fn=embedder)
    store.add_documents(docs)
    print(f"Đã lưu trữ {store.get_collection_size()} chunks văn bản.")

    # Chạy các câu hỏi Benchmark
    benchmark_queries = [
        "Sọ Dừa có ngoại hình như thế nào từ khi sinh ra?",
        "Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại?",
        "Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào?",
        "Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ đâu?",
        "Bài học rõ nét nhất từ câu chuyện Cây Khế?"
    ]

    query_to_run = [question] if question else benchmark_queries

    agent = KnowledgeBaseAgent(store=store, llm_fn=real_llm_call)

    for q in query_to_run:
        print(f"\n--- Query: {q} ---")
        # In kết quả retrieval để lấy số liệu cho báo cáo
        results = store.search(q, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"Top-{i} Chunk (Score: {res['score']:.4f}): {res['content'][:100]}...")
        
        # Agent trả lời
        # print(f"Agent Answer: {agent.answer(q)}")

def main():
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    run_manual_demo(question)

if __name__ == "__main__":
    main()