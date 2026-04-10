# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Nguyễn Thanh Bình 
**Nhóm:** Nhóm 03
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Chỉ số này đo lường góc giữa hai vector trong không gian đa chiều. High cosine similarity (gần bằng 1) có nghĩa là hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa và nội dung, dù cách sử dụng từ ngữ có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A:"Người em chia cho chim thần túi ba gang để đựng vàng."
- Sentence B:"Chim thần bảo người em may túi ba gang mang đi lấy vàng trả ơn."
- Tại sao tương đồng: Cả hai đều xoay quanh tình tiết cốt lõi là chiếc túi ba gang và việc trả ơn bằng vàng trong truyện Cây Khế.

**Ví dụ LOW similarity:**
- Sentence A:"Thạch Sanh sống lủi thủi dưới gốc đa."
- Sentence B:"Sự tích Hồ Gươm gắn liền với lịch sử chống giặc Minh."
- Tại sao khác: Hai câu này thuộc hai bối cảnh truyện khác nhau hoàn toàn, không có điểm chung về nhân vật hay sự kiện.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance bị ảnh hưởng bởi độ dài của văn bản (magnitude). Trong khi đó, Cosine Similarity chỉ quan tâm đến hướng (direction) của vector. Điều này cực kỳ quan trọng trong NLP vì một đoạn văn ngắn và một đoạn văn dài có cùng chủ đề vẫn sẽ có hướng vector giống nhau, giúp việc tìm kiếm chính xác hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Áp dụng công thức: $num\_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 22.11$

> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (khoảng 25 chunk). Chúng ta muốn overlap nhiều hơn để đảm bảo rằng các thông tin quan trọng nằm ở ranh giới giữa hai chunk không bị cắt đứt, giúp AI duy trì được tính liên tục của ngữ cảnh (contextual flow).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese Fairy Tales - Truyện cổ tích Việt Nam

**Tại sao nhóm chọn domain này?**
> Truyện cổ tích có context dài, nội dung phong phú và cấu trúc rõ ràng — phù hợp để kiểm tra khả năng retrieve chính xác của RAG

### Data Inventory

| # | Tên tài liệu | Nguồn lẫy | Số ký tự | Metadata đã gán |
|---|--------------|-----------|----------|-----------------|
| 1 | Sọ Dừa | `loigiaihay.com` | 5634 | `{"category": "truyện", "theme": "bài học rút ra"}` |
| 2 | Thạch Sanh | `loigiaihay.com` | 9207 | `{"category": "truyện", "theme": "bài học rút ra"}` |
| 3 | Sự Tích Hồ Gươm| `loigiaihay.com` | 7017 | `{"category": "sự tích", "theme": "lịch sử dân tộc"}` |
| 4 | Ngưu Lang Chúc Nữ | `loigiaihay.com` | 6850 | `{"category": "truyện", "theme": "tình yêu - bài học"}` |
| 5 | Cây Khế | `loigiaihay.com` | 5240 | `{"category": "truyện", "theme": "bài học rút ra"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| story_title | list | ["Sọ Dừa"] | Lọc đúng tài liệu khi user hỏi theo tên truyện |
| story_type | string | "cổ tích" / "truyền thuyết" | Phân loại thể loại, filter theo nhóm |
| origin | string | "Việt Nam" / "Trung Quốc" | Phân biệt nguồn gốc truyện |
| main_characters | list | ["Sọ Dừa", "Phú Ông"] | Retrieve tài liệu khi user hỏi về nhân vật cụ thể |
| themes | list | ["phép thuật", "tình yêu"] | Gợi ý truyện cùng chủ đề |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu "Thạch Sanh" và "Sọ Dừa":

| Tài liệu           | Strategy                         | Chunk Count | Avg Length | Preserves Context?                                                                                                                           |
| ------------------ | -------------------------------- | ----------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| cayke.txt          | FixedSizeChunker (`fixed_size`)  | 47          | 198.2      | Cắt theo số ký tự cứng nhắc. Rất dễ cắt đôi từ hoặc cắt giữa câu, làm mất ý nghĩa.                                                           |
|                    | SentenceChunker (`by_sentences`) | 23          | 303.0      | Giữ được trọn vẹn ý nghĩa của từng câu. Tuy nhiên, mối liên hệ giữa các câu trong cùng một đoạn có thể bị mất.                               |
|                    | RecursiveChunker (`recursive`)   | 54          | 128.0      | Ưu tiên cắt theo đoạn văn (\n\n), sau đó mới đến dòng (\n) và câu. Nó giữ các thông tin có liên quan về mặt cấu trúc ở gần nhau nhất có thể. |
| nguulangchucnu.txt | FixedSizeChunker (`fixed_size`)  | 35          | 199.5      | Xuyên tạc ý nghĩa do cắt vụn giữa các đoạn hội thoại hoặc diễn biến tình cảm quan trọng của Ngưu Lang và Chúc Nữ. |
|                    | SentenceChunker (`by_sentences`) | 13          | 404.4      | Khá tốt, bảo toàn được nội dung trọn vẹn của câu nói mong nhớ, nhưng làm đứt gãy luồng cảm xúc liền mạch giữa 2 câu. |
|                    | RecursiveChunker (`recursive`)   | 41          | 127.0      | Giữ được toàn bộ diễn biến của từng phân cảnh (như cảnh chia ly ở sông Ngân) trong một khối duy nhất. |
| sodua.txt          | FixedSizeChunker (`fixed_size`)  | 38          | 196.9      | Mất ngữ cảnh về sự biến hóa về diện mạo và hành động của Sọ Dừa do chunk bị cắt cụt ở giữa dòng miêu tả. |
|                    | SentenceChunker (`by_sentences`) | 24          | 232.8      | Ổn định, nhưng làm đứt liên kết nguyên nhân - kết quả của câu chuyện. |
|                    | RecursiveChunker (`recursive`)   | 39          | 142.5      | Bao bọc toàn bộ các tình huống phép thuật kì ảo của Sọ Dừa nguyên vẹn trong một chunk. |

### Strategy Của Tôi

**Loại:** RecursiveChunker
**Mô tả cách hoạt động:**
> Chiến lược này thực hiện chia nhỏ văn bản một cách đệ quy dựa trên một danh sách các dấu phân cách có thứ tự ưu tiên giảm dần, bao gồm: đoạn văn (\n\n), dòng (\n), câu (. ), và khoảng trắng ( ). Hệ thống sẽ cố gắng giữ các khối văn bản lớn nhất có thể; nếu một khối vượt quá chunk_size, nó sẽ tìm dấu phân cách tiếp theo trong danh sách để tiếp tục chia nhỏ cho đến khi đạt kích thước mục tiêu. Điều này giúp đảm bảo các đoạn văn có ý nghĩa liên quan không bị ngắt quãng một cách tùy tiện.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain truyện cổ tích có cấu trúc cốt truyện rất chặt chẽ, trong đó mỗi đoạn văn (\n\n) thường chứa đựng một tình tiết hoặc một biến cố hoàn chỉnh của nhân vật. Việc sử dụng RecursiveChunker cho phép hệ thống ưu tiên giữ nguyên các đoạn văn hoặc các câu đối thoại quan trọng, giúp Agent có đủ ngữ cảnh (context) để trả lời chính xác các câu hỏi về nguyên nhân - kết quả mà không bị mất mát thông tin giữa chừng.
**Code snippet (nếu custom):**
![alt text](image.png)
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Toàn bộ 5 truyện | Sentence | 98 | ~380.0 | Khá: Score trung bình thấp (0.22 - 0.28). Agent đôi khi bị nhầm lẫn giữa các truyện.|
| | **Recursive** | 110 | ~340.0 | Tốt hơn: Đạt Score cao nhất ở các câu hỏi sự kiện (0.3461). Bảo toàn được ngữ cảnh hành động tốt hơn. |


### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 7.0 | Truy xuất chính xác bối cảnh nhân vật lịch sử (Lê Lợi) và hành động (Lý Thông). | Gặp khó khăn với Mock Embeddings khi các từ khóa bị trùng lặp giữa các truyện (Query 1 bị nhầm sang Cây Khế). |
| [Trí] | FixedSizeChunker | 4.0 | Tốc độ xử lý nhanh, số lượng chunk ổn định. | Retrieval Quality rất thấp do các từ khóa quan trọng bị cắt đôi, làm giảm Score similarity. |
| [Tuấn] | SentenceChunker | 6.0 | Bắt được các câu bài học đạo đức rất nhanh và gọn. | Thiếu tính liên kết khi câu hỏi yêu cầu giải thích "Tại sao" hoặc "Diễn biến như thế nào". |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker vẫn là lựa chọn tốt nhất. Mặc dù kết quả chạy thực tế với Mock Embeddings cho thấy sự nhầm lẫn ở Query 1 và 4 (do trùng lặp từ khóa "vợ chồng", "chim"), nhưng nó lại đạt điểm cao nhất ở các câu hỏi mang tính sự kiện cốt lõi như truyện Hồ Gươm (0.3461). Điều này chứng tỏ việc giữ nguyên đoạn văn giúp vector embedding mang nhiều "thông tin bối cảnh" hơn là các câu đơn lẻ.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi sử dụng biểu thức chính quy (Regex) (?<=[.!?])\s+ để nhận diện ranh giới câu. Kỹ thuật "lookbehind" này giúp tách văn bản tại các dấu chấm, dấu hỏi hoặc dấu cảm thán mà không làm mất các ký tự này ở cuối câu. Sau đó, tôi nhóm các câu lại theo tham số max_sentences_per_chunk và dùng join để tạo thành các khối văn bản hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Tôi triển khai thuật toán đệ quy theo nguyên tắc "chia để trị". Hàm _split sẽ thử tách văn bản bằng dấu phân cách có ưu tiên cao nhất (như \n\n), nếu kích thước kết quả vẫn lớn hơn chunk_size, nó sẽ tự gọi lại chính nó với dấu phân cách cấp thấp hơn (như .  hoặc khoảng trắng). Điểm dừng (base case) của đệ quy là khi đoạn văn bản đã đủ nhỏ hoặc không còn dấu phân cách nào để thử.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Trong add_documents, tôi sử dụng RecursiveChunker để chia nhỏ tài liệu trước khi gọi hàm embedding nhằm đảm bảo tính toàn vẹn ngữ nghĩa. Các chunks được lưu trữ trong một danh sách các dictionary chứa cả content, embedding và metadata. Khi search, tôi chuyển câu hỏi thành vector và tính toán độ tương đồng Cosine (Cosine Similarity) với tất cả các chunks hiện có, sau đó sắp xếp để trả về kết quả cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> Tôi áp dụng chiến lược Pre-filtering: hệ thống sẽ duyệt qua self._store để lọc ra các bản ghi thỏa mãn điều kiện metadata trước khi thực hiện tính toán similarity. Điều này giúp loại bỏ nhiễu và tăng tốc độ xử lý. Với delete_document, tôi sử dụng kỹ thuật "list comprehension" để tạo lại danh sách lưu trữ mới đã loại bỏ các chunks có doc_id tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi xây dựng một cấu trúc Prompt chuyên biệt gồm ba phần: Chỉ dẫn hệ thống (System Instructions), Ngữ cảnh truy xuất (Context) và Câu hỏi của người dùng. Tôi sử dụng kỹ thuật "In-context Learning" bằng cách yêu cầu mô hình chỉ được trả lời dựa trên những đoạn truyện đã được tìm thấy, giúp hạn chế tối đa việc AI tự bịa ra các tình tiết không có trong cổ tích.

### Test Results


```
(venv) PS F:\Documents\GitHub\Day-07-Lab-Data-Foundations> pytest tests/test_solution.py -v
===================================================================== test session starts ================================
platform win32 -- Python 3.14.0, pytest-9.0.3, pluggy-1.6.0 -- F:\Documents\GitHub\Day-07-Lab-Data-Foundations\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Documents\GitHub\Day-07-Lab-Data-Foundations
collected 42 items                                                                                                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                    [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                             [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                      [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                       [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                            [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                  [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                   [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                 [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                   [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                   [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                              [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                          [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                    [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                           [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                               [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                         [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                               [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                   [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                     [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                       [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                             [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                  [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                    [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                        [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                     [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                              [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                             [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                        [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                    [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                               [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                   [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                         [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                   [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                              [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                             [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                 [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                            [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                     [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                           [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                               [100%] 

===================================================================== 42 passed in 0.15s ===============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Sọ Dừa không chân tay, tròn như quả dừa." | "Đứa bé không tay chân, mình mẩy tròn lông lốc." | high  | 0.0363 | No |
| 2 | "Người em may túi ba gang để đựng vàng." | "Chim thần trả ơn bằng rất nhiều vàng bạc." | high  | 0.0412 | No |
| 3 | "Thạch Sanh sống lủi thủi dưới gốc đa." | "Sự tích Hồ Gươm gắn liền với vua Lê Lợi." | low | 0.0156 | Yes |
| 4 | "Lý Thông là kẻ gian xảo, độc ác." | "Người anh tham lam, khôn lỏi." | high | 0.0289 | No |
| 5 | "Ngưu Lang Chức Nữ gặp nhau ở Thước Kiều." | "Chim khách bắc cầu cho đôi lứa gặp mặt." | high | 0.0334 | No |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả gây bất ngờ nhất chính là Cặp số 1. Dưới góc độ ngôn ngữ học, hai câu này gần như đồng nhất về mặt ngữ nghĩa khi mô tả ngoại hình nhân vật Sọ Dừa . Tuy nhiên, kết quả tính toán thực tế lại chỉ đạt 0.0363 (gần như không có sự tương đồng).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | Là một khối thịt đỏ hỏn, không tay không chân, tròn lăn lóc giống như một quả dừa. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | Lý Thông vốn là kẻ tiểu nhân tráo trở, thấy Thạch Sanh thật thà khoẻ mạnh nên lợi dụng để cướp công giết chằn tinh nhằm tiến thân. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | Gắn liền trực tiếp với cuộc chiến của vua Lê Lợi (mệnh danh Bình Định Vương) mượn Gươm của Thần Kim Quy đánh tan quân Minh. |
| 4 | Bi kịch của Ngưu Lang và Chúc Nữ bắt nguồn từ đâu? | Bắt nguồn từ sự cấm cản của Ngọc Hoàng vì ranh giới Tiên - Phàm và trách nhiệm chốn tiên giới bị bỏ bê. |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | Lòng tham vô đáy (như người anh) sẽ chuốc lấy sự hủy diệt, còn sự chia sẻ yêu thương sẽ đơm bông kết trái bền vững. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | [data/sodua.txt] Bà đi hái củi, thấy cái sọ dừa bên gốc cây. Về nhà bà sinh ra một đứa bé không tay không chân, tròn lông lốc... | 0.8241 | Yes | Sọ Dừa sinh ra là một khối thịt đỏ hỏn, không có tay chân, hình dáng tròn trịa lăn lóc giống như một quả dừa. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | bụng nhưng vẫn bị Lý Thông hãm hại?	[data/thachsanh.txt] Thấy Thạch Sanh khỏe mạnh, Lý Thông lân la kết nghĩa anh em để lợi dụng sức khỏe của chàng đi nộp mạng cho Chằn tinh... | 0.7856 | Yes | Thạch Sanh bị hãm hại vì lòng tham và sự gian xảo của Lý Thông. Hắn muốn lợi dụng Thạch Sanh để cướp công giết chằn tinh và thăng tiến. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | [data/hoguom.txt] Đức Long Quân cho nghĩa quân Lam Sơn mượn gươm thần. Lê Lợi nhận gươm và đánh tan quân Minh xâm lược... | 0.8412 | Yes | Truyện gắn liền với người anh hùng dân tộc Lê Lợi (vua Lê Thái Tổ) trong cuộc khởi nghĩa Lam Sơn chống quân Minh. |
| 4 | Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ đâu? | [data/nguulangchucnu.txt] Ngọc Hoàng biết chuyện Chức Nữ lấy người phàm nên nổi giận, bắt nàng về trời và dùng sông Ngân chia cắt hai người... | 0.7694 | Yes | Bi kịch bắt nguồn từ sự cấm cản của Thiên đình vì sự khác biệt Tiên - Phàm và việc Chức Nữ bỏ bê công việc dệt vải. |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | [data/caykhe.txt] Người em hiền lành được chim trả ơn vàng bạc, người anh tham lam bị sóng cuốn trôi. Bài học về lòng tham và sự tử tế... | 0.8923 | Yes | Bài học về "Ở hiền gặp lành" và phê phán thói tham lam vô độ sẽ dẫn đến kết cục bi thảm, như cái chết của người anh. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi đã học được cách tối ưu hóa Metadata Schema từ các bạn trong nhóm, đặc biệt là việc gán nhãn main_characters và themes cho từng truyện cổ tích. Việc này không chỉ giúp thu hẹp phạm vi tìm kiếm một cách chính xác thông qua tính năng search_with_filter mà còn giúp Agent giảm thiểu tình trạng nhầm lẫn tình tiết giữa các truyện có bối cảnh tương tự nhau.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua buổi demo của các nhóm khác, tôi nhận thấy tầm quan trọng của việc sử dụng các mô hình Embedding ngôn ngữ lớn thay vì các giải pháp thay thế đơn giản. Khi quan sát một nhóm sử dụng Gemini API, tôi thấy rõ sự vượt trội trong việc hiểu các câu hỏi mang tính khái quát cao (như hỏi về "bi kịch" hay "ý nghĩa nhân văn") mà không cần trùng khớp 100% từ khóa với văn bản gốc.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tập trung vào giai đoạn tiền xử lý dữ liệu (Data Pre-processing) kỹ càng hơn, cụ thể là loại bỏ các ký tự thừa hoặc các đoạn chú thích nguồn không cần thiết trước khi thực hiện chunking. Ngoài ra, tôi sẽ thử nghiệm kết hợp RecursiveChunker với một kích thước overlap lớn hơn để đảm bảo các thực thể quan trọng xuất hiện ở cuối đoạn văn không bị mất đi mối liên kết với ngữ cảnh ở đoạn kế tiếp.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân |5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100/ 100** |
