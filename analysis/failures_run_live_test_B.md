# Báo Cáo Phân Tích Lỗi Thực Nghiệm: `run_live_test_B`

- **Cấu hình**: `B`
- **Tổng số câu hỏi**: 5
- **Tổng số ca lỗi**: 1 (`20.0%`)

## 1. Bảng Tổng Hợp Nhóm Lỗi (Failure Taxonomy)

| STT | Nhóm lỗi (Category) | Mô tả | Số ca | Tỉ lệ (%) |
|---|---|---|---|---|
| 1 | `retrieval_miss` | Không truy xuất được evidence cần thiết | **1** | 100.0% |
| 2 | `generation_error` | Truy xuất đủ evidence nhưng trả lời sai/thiếu ý | **0** | 0.0% |
| 3 | `hallucination_unanswerable` | Câu hỏi ngoài phạm vi nhưng vẫn bịa đặt trả lời | **0** | 0.0% |
| 4 | `unfaithful` | Câu trả lời chứa dữ kiện ngoài ngữ cảnh (unfaithful) | **0** | 0.0% |
| 5 | `false_refusal` | Có dữ kiện trong ngữ cảnh nhưng từ chối nhầm | **0** | 0.0% |
| 6 | `other` | Các lỗi khác | **0** | 0.0% |

## 2. Chi Tiết Các Ca Lỗi Điển Hình (Tối đa 5 ca mỗi nhóm)

### Nhóm: `retrieval_miss` - Không truy xuất được evidence cần thiết (1 ca)

#### Ca 1: [q_005] (Type: `casual`)
- **Câu hỏi**: Chào bạn, cho tôi biết MoMo là ứng dụng gì?
- **Đáp án chuẩn (Gold)**: MoMo là ứng dụng Ví điện tử thuộc quyền sở hữu của Công ty Cổ phần Dịch vụ Di Động Trực Tuyến (M_Service).
- **Câu trả lời của hệ thống**: MoMo là nền tảng thanh toán điện tử và tài chính số do Công ty Cổ phần Dịch vụ Di động Trực tuyến ("MoMo/Công ty") phát triển và vận hành, cung cấp các dịch vụ ví điện tử, chuyển tiền, thanh toán, sản phẩm tài chính và hệ sinh thái dịch vụ số [4].
- **Đánh giá của Judge**: `partial` | Faithful: `True` | Refused: `False`
- **Lý do từ Judge**: Câu trả lời cung cấp thông tin chính xác về MoMo là nền tảng thanh toán điện tử và tài chính số, tuy nhiên, đáp án chuẩn nhấn mạnh MoMo là ứng dụng Ví điện tử thuộc sở hữu của Công ty Cổ phần Dịch vụ Di Động Trực Tuyến (M_Service), thông tin này không được đề cập trực tiếp trong câu trả lời của hệ thống.
- **Các đoạn trích đã truy xuất**:
  * `[momo_faq_basic-B-001]` (Score: `0.7659`): Hỏi đáp - Thông tin cơ bản MoMo - Ứng dụng tài chính Dịch vụ Về MoMo Tin tức Trợ giúp Đối tác Blog MoMo Hỏi đáp Thông tin cơ bản Quản lý ứng dụng và tài khoản Tải, xóa, cập nhật ứng dụng Đăng ký, đăng...
  * `[momo_faq_finance-B-001]` (Score: `0.7622`): Hỏi đáp - Câu hỏi chung về tài chính MoMo - Ứng dụng tài chính Dịch vụ Về MoMo Tin tức Trợ giúp Đối tác Blog MoMo Hỏi đáp Tài chính Thông tin cơ bản Quản lý ứng dụng và tài khoản Tải, xóa, cập nhật ứn...
  * `[momo_faq_basic-B-002]` (Score: `0.7585`): Hỏi đáp - Thông tin cơ bản Câu hỏi chung về Du Lịch - Đi Lại Vé máy bay Vé tàu hỏa Vé xe khách Dịch vụ đi lại khác Sống tốt cùng MoMo Heo Đất MoMo Trái Tim MoMo Mua thẻ Game Sự cố và khiếu nại Sự cố k...
  * `[momo_privacy-B-001]` (Score: `0.7578`): Chính sách quyền riêng tư > Chính sách quyền riêng tư *Cập nhật ngày 01/04/2026*  Ứng dụng MoMo là nền tảng thanh toán điện tử và tài chính số do Công ty Cổ phần Dịch vụ Di động Trực tuyến ("MoMo/Công...
  * `[momo_faq_finance-B-002]` (Score: `0.7491`): Hỏi đáp - Câu hỏi chung về tài chính Câu hỏi chung về Du Lịch - Đi Lại Vé máy bay Vé tàu hỏa Vé xe khách Dịch vụ đi lại khác Sống tốt cùng MoMo Heo Đất MoMo Trái Tim MoMo Mua thẻ Game Sự cố và khiếu n...

**Nguyên nhân (điền tay):** _________________________________________________________________


---
*Báo cáo được sinh tự động bởi `eval.failures`.*
