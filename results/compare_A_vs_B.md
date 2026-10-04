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
