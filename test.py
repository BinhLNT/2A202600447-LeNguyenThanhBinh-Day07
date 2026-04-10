from src.chunking import compute_similarity
from src.embeddings import _mock_embed

# Thay đổi các câu bên dưới theo bảng của em
sent_a = "Sọ Dừa không chân tay, tròn như quả dừa."
sent_b = "Đứa bé không tay chân, mình mẩy tròn lông lốc."

vec_a = _mock_embed(sent_a)
vec_b = _mock_embed(sent_b)

score = compute_similarity(vec_a, vec_b)
print(f"Actual Score: {score:.4f}")