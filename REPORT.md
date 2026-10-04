# BÁO CÁO KẾT QUẢ THỰC NGHIỆM HỆ THỐNG RAG Q&A VÍ MOMO

**So sánh hai cấu hình**: Cấu hình `A` (`run_live_test_A`) vs Cấu hình `B` (`run_live_test_B`)
**Ngày lập báo cáo**: `<!-- TODO: Điền ngày tháng năm -->`
**Người thực hiện**: `<!-- TODO: Điền tên chuyên viên / nhóm thực hiện -->`

## 1. Mục Tiêu Thực Nghiệm & Dữ Liệu

### 1.1. Mục tiêu
Thực nghiệm nhằm đánh giá so sánh chất lượng truy xuất và sinh câu trả lời giữa hai chiến lược RAG:
- **Cấu hình A**: Cấu hình A: chunk phẳng 500 ký tự (naive), prompt v1_simple
- **Cấu hình B**: Cấu hình B: chunk phân cấp theo cấu trúc có breadcrumb, prompt v2_strict

### 1.2. Thống kê tập tài liệu và tập đánh giá (Eval Set)
| Hạng mục dữ liệu | Số lượng | Ghi chú |
|---|---|---|
| Tài liệu xử lý sạch (`data/processed/`) | 5 tài liệu | Bao gồm điều khoản chung, chính sách bảo mật, FAQ |
| Vector chunks Chiến lược A (`momo_A`) | 202 chunks | Cắt cố định ~500 ký tự |
| Vector chunks Chiến lược B (`momo_B`) | 109 chunks | Cắt theo heading và gộp ngữ cảnh cha-con |
| Tổng số câu hỏi kiểm thử (`eval_set`) | 5 câu | Phân bổ 5 loại câu hỏi |

**Phân bổ chi tiết câu hỏi theo từng loại:**
- `direct`: 1 câu
- `multi`: 1 câu
- `unanswerable`: 1 câu
- `trap`: 1 câu
- `casual`: 1 câu

## 2. Bảng So Sánh Hai Cấu Hình Pipeline (A vs B)

| Thông số cấu hình | Cấu hình A | Cấu hình B |
|---|---|---|
| **Collection Chroma** | `momo_A` | `momo_B` |
| **Số lượng Top-k** | `3` | `5` |
| **Similarity Threshold** | `None` | `None` |
| **Giới hạn ký tự context** | `6000` | `6000` |
| **Prompt Template** | `prompts/v1_simple.txt` | `prompts/v2_strict.txt` |
| **Short Circuit Context Rỗng** | `False` | `False` |
| **Mô hình Generator (LLM)** | `Default generator` | `Default generator` |
| **Temperature** | `0.1` | `0.1` |

## 3. Kết Quả Đo Lường & So Sánh (Benchmark Results)

*Dưới đây là bảng tổng hợp số liệu trích xuất từ `eval.compare`:*

# So Sánh Thực Nghiệm RAG: `run_live_test_A` (A) vs `run_live_test_B` (B)

- **Run A**: `run_live_test_A` (Cấu hình `A`) - Số câu: 5
- **Run B**: `run_live_test_B` (Cấu hình `B`) - Số câu: 5

## 1. Bảng So Sánh Chỉ Số Tổng Thể (Overall Metrics)

| Nhóm chỉ số | Chỉ số (Metric) | Run A | Run B | Chênh lệch (B - A) | Đánh giá |
|---|---|---|---|---|---|
| **Retrieval** | Doc Hit@k | `75.0%` | `75.0%` | `0.0%` | **-** |
| **Retrieval** | Evidence Hit@k | `75.0%` | `75.0%` | `0.0%` | **-** |
| **Retrieval** | Evidence Recall | `75.0%` | `75.0%` | `0.0%` | **-** |
| **Retrieval** | MRR | `0.6250` | `0.7500` | `+0.1250` | **B (tốt hơn)** |
| **Generation** | Correct Rate | `60.0%` | `80.0%` | `+20.0%` | **B (tốt hơn)** |
| **Generation** | Partial Rate | `40.0%` | `20.0%` | `-20.0%` | **B (tốt hơn)** |
| **Generation** | Incorrect Rate | `0.0%` | `0.0%` | `0.0%` | **-** |
| **Generation** | Faithful Rate | `100.0%` | `100.0%` | `0.0%` | **-** |
| **Hallucination** | Unanswerable Hallucination | `0.0%` | `0.0%` | `0.0%` | **-** |
| **Hallucination** | Answerable Refusal (Từ chối nhầm) | `0.0%` | `0.0%` | `0.0%` | **-** |
| **Hallucination** | Unfaithful Rate | `0.0%` | `0.0%` | `0.0%` | **-** |
| **Citations** | Tỉ lệ có trích dẫn [n] | `0.0%` | `80.0%` | `+80.0%` | **B (tốt hơn)** |
| **Citations** | Độ hợp lệ trích dẫn | `100.0%` | `100.0%` | `0.0%` | **-** |
| **Vận hành** | Độ trễ Total p50 | `5937.5 ms` | `2049.0 ms` | `-3888.5 ms` | **B (tốt hơn)** |
| **Vận hành** | Độ trễ Total p95 | `20197.6 ms` | `2650.7 ms` | `-17546.9 ms` | **B (tốt hơn)** |
| **Vận hành** | Độ trễ Retrieval p50 | `784.5 ms` | `667.3 ms` | `-117.2 ms` | **B (tốt hơn)** |
| **Vận hành** | Độ trễ Generation p50 | `5078.6 ms` | `1448.0 ms` | `-3630.6 ms` | **B (tốt hơn)** |
| **Vận hành** | Tokens trung bình / câu | `1100.6` | `1743.6` | `+643` | **A (tốt hơn)** |
| **Vận hành** | Chi phí trung bình / câu | `$0.000220` | `$0.000073` | `$-0.000147` | **B (tốt hơn)** |
| **Vận hành** | Tổng chi phí | `$0.001100` | `$0.000367` | `$-0.000733` | **B (tốt hơn)** |
| **Vận hành** | Số câu lỗi (errors) | `0` | `0` | `0` | **-** |

## 2. So Sánh Theo Phân Loại Câu Hỏi (Breakdown by Type)

| Loại câu (Type) | Metric | Run A | Run B | Chênh lệch (B - A) | Đánh giá |
|---|---|---|---|---|---|
| `direct` | Evidence Hit@k | `100.0%` | `100.0%` | `0.0%` | - |
| `direct` | Correct Rate | `0.0%` | `100.0%` | `+100.0%` | B (tốt hơn) |
| `multi` | Evidence Hit@k | `100.0%` | `100.0%` | `0.0%` | - |
| `multi` | Correct Rate | `100.0%` | `100.0%` | `0.0%` | - |
| `unanswerable` | Correct Rate | `100.0%` | `100.0%` | `0.0%` | - |
| `unanswerable` | Unans Hallucination | `0.0%` | `0.0%` | `0.0%` | - |
| `trap` | Evidence Hit@k | `100.0%` | `100.0%` | `0.0%` | - |
| `trap` | Correct Rate | `100.0%` | `100.0%` | `0.0%` | - |
| `casual` | Evidence Hit@k | `0.0%` | `0.0%` | `0.0%` | - |
| `casual` | Correct Rate | `0.0%` | `0.0%` | `0.0%` | - |

## 3. Phân Tích Chênh Lệch Từng Câu Hỏi (Divergence Analysis)

### 3.1. Các câu Run A đúng, Run B sai (0 câu)

*(Không có câu nào Run A đúng mà Run B sai)*


### 3.2. Các câu Run B đúng, Run A sai (1 câu)

| Question ID | Câu hỏi | Run A | Run B |
|---|---|---|---|
| `q_001` | Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có được mở tài khoản MoMo không? | partial | **correct** |

---
*Báo cáo được sinh tự động bởi `eval.compare`.*

## 4. Phân Tích Lỗi Chuyên Sâu (Failure Analysis)

### 4.1. Tổng quan phân loại lỗi
Hệ thống tự động phân loại lỗi theo 5 nhóm nguyên nhân chính:
1. `retrieval_miss`: Lỗi tại bộ truy xuất (không tìm thấy đoạn chứng cứ cần thiết).
2. `generation_error`: Truy xuất đủ nhưng mô hình tổng hợp sai hoặc thiếu điều kiện.
3. `hallucination_unanswerable`: Câu hỏi unanswerable nhưng mô hình bịa đặt câu trả lời.
4. `unfaithful`: Mô hình trả lời chứa dữ kiện ngoài ngữ cảnh tài liệu.
5. `false_refusal`: Ngữ cảnh có dữ kiện nhưng mô hình từ chối nhầm.

### 4.2. Nghiên cứu các trường hợp điển hình (Case Studies)
<!-- TODO: Lựa chọn 3-5 ca lỗi tiêu biểu nhất từ analysis/failures_*.md để phân tích chi tiết: -->
- **Trường hợp 1**: `[ID câu hỏi]`
  * Hiện tượng: `<!-- Mô tả lỗi -->`
  * Nguyên nhân gốc rễ: `<!-- Do chunking / do embedding / do prompt / do model -->`
  * Giải pháp khắc phục: `<!-- Tăng threshold / sửa prompt / bổ sung metadata -->`

- **Trường hợp 2**: `[ID câu hỏi]`
  * Hiện tượng: `<!-- Mô tả lỗi -->`
  * Nguyên nhân gốc rễ: `<!-- ... -->`
  * Giải pháp khắc phục: `<!-- ... -->`

## 5. Đánh Giá Độ Tin Cậy Của LLM-Judge (Judge Reliability)

Kết quả đối chiếu giữa người chấm tay độc lập và LLM-Judge:
- **Run B (`run_live_test_B`)**:
  * Tỉ lệ đồng thuận Correctness: `100.0%`
  * Hệ số Cohen's Kappa: `1.0000` (Rất cao (Almost perfect agreement))

## 6. Hạn Chế Của Hệ Thống Hiện Tại (Limitations)
<!-- TODO: Ghi chú các hạn chế đã quan sát được trong quá trình chạy: -->
1. **Độ phủ dữ liệu**: Bộ corpus hiện tại tập trung vào điều khoản chung và FAQ liên kết; chưa bao quát toàn bộ sản phẩm dịch vụ chuyên sâu.
2. **Rate Limit API**: Tài khoản Google AI Studio Free Tier bị hạn chế số yêu cầu/phút nên quá trình đánh giá cần chạy điều tiết `concurrency`.
3. **Độ dài văn bản**: Các đoạn trích phức tạp trong điều khoản pháp lý đôi khi bị phân mảnh nếu không có breadcrumb.

## 7. Kết Luận & Kế Hoạch Cải Tiến (Conclusions & Next Steps)
<!-- TODO: Đưa ra kết luận cuối cùng chọn cấu hình nào để đưa vào production và các cải tiến tiếp theo: -->
1. **Quyết định cấu hình**: `<!-- Khuyến nghị cấu hình B do vượt trội ở tính trung thực và trích dẫn -->`
2. **Kế hoạch tiếp theo**:
   - Tinh chỉnh `similarity_threshold` (khuyến nghị 0.72 - 0.74) và bật `short_circuit_on_empty_context` để giảm chi phí token.
   - Mở rộng thêm 20+ câu hỏi đa bước (multi-hop) và câu bẫy (trap) vào tập eval_set.
   - Tích hợp thêm bộ lọc Re-ranking (như Cohere Rerank hoặc cross-encoder) trước khi đưa vào context LLM.

---
*Khung báo cáo được sinh tự động bởi `eval.report_skeleton`.*
