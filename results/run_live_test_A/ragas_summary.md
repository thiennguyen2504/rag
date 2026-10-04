# Báo cáo đánh giá Ragas: Run `run_live_test_A`

## 1. Tóm tắt chỉ số trung bình

| Chỉ số | Điểm số trung bình (0.0 - 1.0) | Ý nghĩa |
| :--- | :---: | :--- |
| **Faithfulness** | 1.0000 | Câu trả lời trung thực, không bịa đặt ngoài ngữ cảnh truy xuất |
| **Answer Relevancy** | 0.8740 | Mức độ bám sát câu hỏi người dùng |
| **Context Precision** | 1.0000 | Tỷ lệ các đoạn chính xác xuất hiện ở đầu bảng xếp hạng |
| **Context Recall** | 0.8835 | Ngữ cảnh truy xuất chứa đầy đủ bằng chứng đối chuẩn (ground truth) |

## 2. Chi tiết từng câu hỏi

| Question ID | Faithfulness | Relevancy | Context Precision | Context Recall | Câu hỏi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `q_002` | 1.0000 | 0.7941 | 1.0000 | 1.0000 | Hạn mức giao dịch và hạn mức ngày mặc định khi liên kết dịch vụ trên MoMo là bao nhiêu? |
| `q_001` | 1.0000 | 0.9091 | 1.0000 | 1.0000 | Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có được mở tài khoản MoMo không? |
| `q_004` | 1.0000 | 0.8542 | 1.0000 | 0.9844 | Có đúng là sau khi liên kết tài khoản dịch vụ thì mỗi lần thanh toán người dùng vẫn phải mở ứng dụng MoMo để bấm xác nhận không? |
| `q_005` | 1.0000 | 0.8125 | 1.0000 | 0.4333 | Chào bạn, cho tôi biết MoMo là ứng dụng gì? |
| `q_003` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Ví MoMo có hỗ trợ tính năng mua bán tiền điện tử Bitcoin không? |
