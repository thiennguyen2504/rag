# Báo Cáo Đánh Giá Thực Nghiệm RAG: `run_live_test_B`

- **Cấu hình (Config ID)**: `B`
- **Tổng số câu hỏi đánh giá**: `5`
- **Số câu hoàn tất chấm (Judged)**: `5`
- **Số lỗi hệ thống**: `0` (`0.0%`)

## 1. Bảng Chỉ Số Tổng Thể (Overall Performance)

| Nhóm chỉ số | Chỉ số (Metric) | Kết quả | Ghi chú |
|---|---|---|---|
| **Retrieval** | Doc Hit@k | `75.0%` | Tỉ lệ tìm thấy đúng tài liệu nguồn |
| | Evidence Hit@k | `75.0%` | Tỉ lệ tìm thấy ít nhất 1 đoạn chứng cứ |
| | Evidence Recall | `75.0%` | Độ phủ chứng cứ (đặc biệt cho multi-hop) |
| | MRR | `0.7500` | Thứ hạng trung bình của chunk chứa evidence |
| **Generation** | Correct Rate | `80.0%` | Tỉ lệ trả lời đúng hoàn toàn |
| | Partial Rate | `20.0%` | Tỉ lệ trả lời đúng một phần / thiếu ý |
| | Incorrect Rate | `0.0%` | Tỉ lệ trả lời sai hoặc bịa đặt |
| | Faithful Rate | `100.0%` | Mọi khẳng định đều có cơ sở trong context |
| **Hallucination** | Unanswerable Hallucination | `0.0%` | Trả lời bừa khi câu hỏi unanswerable |
| | Answerable Refusal | `0.0%` | Từ chối nhầm câu hỏi có đáp án |
| | Unfaithful Rate | `0.0%` | Tỉ lệ câu có thông tin ngoài lề |
| **Citations** | Trích dẫn nguồn `[n]` | `80.0%` | Tỉ lệ câu có dấu trích dẫn |
| | Độ hợp lệ trích dẫn | `100.0%` | Tỉ lệ trích dẫn đúng số hiệu nguồn |

## 2. Phân Tích Chi Tiết Theo Loại Câu Hỏi (Breakdown by Type)

| Loại câu hỏi (Type) | Số câu | Doc Hit@k | Evidence Hit@k | Correct | Partial | Incorrect | Faithful | Refused |
|---|---|---|---|---|---|---|---|---|
| `direct` | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| `multi` | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| `unanswerable` | 1 | N/A | N/A | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| `trap` | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| `casual` | 1 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |

## 3. Chỉ Số Vận Hành & Chi Phí (Operational & Cost)

| Chỉ số | Retrieval | Generation | Total / Chi tiết |
|---|---|---|---|
| **Độ trễ p50 (ms)** | `667.3` ms | `1448.0` ms | `2049.0` ms |
| **Độ trễ p95 (ms)** | `695.8` ms | `1977.4` ms | `2650.7` ms |
| **Token trung bình** | Input: `1670.8` | Output: `72.8` | Total: `1743.6` tokens |
| **Chi phí API (USD)** | - | - | TB: `$0.000073` / câu - Tổng: `$0.000367` |

---
*Báo cáo được khởi tạo tự động bởi `eval.metrics`.*
