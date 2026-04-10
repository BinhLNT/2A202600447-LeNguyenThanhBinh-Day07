from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """
    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
            # TODO: store references to store and llm_fn
            # Đã hoàn thành: Lưu trữ tham chiếu đến store và hàm gọi LLM
            self.store = store
            self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # Đã hoàn thành: Triển khai luồng RAG

        # 1. Retrieval: Lấy top_k các đoạn văn bản liên quan nhất từ vector store
        search_results = self.store.search(question, top_k=top_k)
        
        # Kết hợp các nội dung tìm được thành một chuỗi văn bản làm ngữ cảnh
        context_text = "\n\n".join([res["content"] for res in search_results])
        
        # 2. Build Prompt: Xây dựng chỉ dẫn cho LLM kèm theo ngữ cảnh vừa truy xuất
        # Mục tiêu là yêu cầu AI chỉ trả lời dựa trên thông tin có trong context để tránh "hallucination"
        prompt = f"""Bạn là một trợ lý hữu ích. Hãy trả lời câu hỏi của người dùng dựa TRỰC TIẾP và DUY NHẤT vào phần ngữ cảnh được cung cấp dưới đây.
Nếu thông tin không có trong ngữ cảnh, hãy trả lời rằng bạn không biết, tuyệt đối không tự bịa ra thông tin.

Ngữ cảnh:
---------------------
{context_text}
---------------------

Câu hỏi: {question}

Trả lời:"""

        # 3. Generation: Gọi hàm LLM với prompt đã xây dựng và trả về kết quả
        return self.llm_fn(prompt)
