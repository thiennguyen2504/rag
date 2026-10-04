# Báo cáo kiểm thử tự động Promptfoo: So sánh cấu hình MoMo RAG

Tổng số ca kiểm thử (test cases): **50**

## 1. Bảng so sánh giữa các Cấu hình / Provider

| Cấu hình / Provider | Số test | Đạt (Pass) | Hỏng (Fail) | Tỷ lệ Đạt (%) | Độ trễ TB (ms) | Tổng chi phí ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoMo RAG - Config A (Chunk 500, k=3)** | 50 | 42 | 8 | **84.0%** | 3603.76 ms | $0.000000 |
| **MoMo RAG - Config B (Chunk 1000, k=5)** | 50 | 43 | 7 | **86.0%** | 9941.2 ms | $0.000000 |

## 2. Tiêu chí kiểm thử (Assertions)
- **Factual groundness (Từ khóa quan trọng)**: Kiểm tra thông tin cốt lõi (độ tuổi, hạn mức, chính sách) có xuất hiện trong câu trả lời.
- **Hallucination Prevention & Refusal**: Khi gặp câu hỏi ngoài ngữ cảnh (ví dụ: visa du lịch Nhật Bản), mô hình phải từ chối rõ ràng, không bịa đặt.
- **Chi phí & Độ trễ**: Đo đạc latency thực tế và tổng chi phí tiêu thụ token theo bảng giá niêm yết.
