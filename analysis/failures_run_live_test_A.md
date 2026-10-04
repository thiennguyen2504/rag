# Báo Cáo Phân Tích Lỗi Thực Nghiệm: `run_live_test_A`

- **Cấu hình**: `A`
- **Tổng số câu hỏi**: 5
- **Tổng số ca lỗi**: 2 (`40.0%`)

## 1. Bảng Tổng Hợp Nhóm Lỗi (Failure Taxonomy)

| STT | Nhóm lỗi (Category) | Mô tả | Số ca | Tỉ lệ (%) |
|---|---|---|---|---|
| 1 | `retrieval_miss` | Không truy xuất được evidence cần thiết | **1** | 50.0% |
| 2 | `generation_error` | Truy xuất đủ evidence nhưng trả lời sai/thiếu ý | **1** | 50.0% |
| 3 | `hallucination_unanswerable` | Câu hỏi ngoài phạm vi nhưng vẫn bịa đặt trả lời | **0** | 0.0% |
| 4 | `unfaithful` | Câu trả lời chứa dữ kiện ngoài ngữ cảnh (unfaithful) | **0** | 0.0% |
| 5 | `false_refusal` | Có dữ kiện trong ngữ cảnh nhưng từ chối nhầm | **0** | 0.0% |
| 6 | `other` | Các lỗi khác | **0** | 0.0% |

## 2. Chi Tiết Các Ca Lỗi Điển Hình (Tối đa 5 ca mỗi nhóm)

### Nhóm: `retrieval_miss` - Không truy xuất được evidence cần thiết (1 ca)

#### Ca 1: [q_005] (Type: `casual`)
- **Câu hỏi**: Chào bạn, cho tôi biết MoMo là ứng dụng gì?
- **Đáp án chuẩn (Gold)**: MoMo là ứng dụng Ví điện tử thuộc quyền sở hữu của Công ty Cổ phần Dịch vụ Di Động Trực Tuyến (M_Service).
- **Câu trả lời của hệ thống**: Chào bạn, MoMo là ứng dụng trên nền tảng di động do MoMo phát triển và vận hành, cung cấp dịch vụ Ví Điện Tử và các dịch vụ trung gian thanh toán khác.
- **Đánh giá của Judge**: `partial` | Faithful: `True` | Refused: `False`
- **Lý do từ Judge**: Câu trả lời đã xác định MoMo là một ứng dụng di động cung cấp dịch vụ Ví Điện Tử và các dịch vụ trung gian thanh toán khác, tuy nhiên, nó thiếu thông tin về đơn vị sở hữu ứng dụng (Công ty Cổ phần Dịch vụ Di Động Trực Tuyến - M_Service) như trong đáp án chuẩn.
- **Các đoạn trích đã truy xuất**:
  * `[momo_terms-A-004]` (Score: `0.7802`): **1.2. Ứng Dụng MoMo:** là ứng dụng trên nền tảng di động do MoMo phát triển và vận hành để cung cấp các Sản Phẩm/Dịch Vụ cho Người Sử Dụng, bao gồm dịch vụ Ví Điện Tử và các dịch vụ trung gian thanh ...
  * `[momo_faq_basic-A-004]` (Score: `0.7729`): Thanh toán khoản vay, thẻ tín dụng Túi Thần Tài Ứng dụng MoMo Thông tin cơ bản Dành cho người mới bắt đầu Thông tin cơ bản MoMo có những dịch vụ gì? Sử dụng MoMo có an toàn? Làm thế nào để biết giao d...
  * `[momo_faq_basic-A-005]` (Score: `0.7693`): Quy trình khiếu nại với bộ phận CSKH MoMo là gì? Làm thế nào để kiểm tra thông tin đơn hàng và tìm trợ giúp sau khi mua hàng? Rút tiền từ MoMo về tài khoản ngân hàng liên kết có mất phí không? Dịch vụ...

**Nguyên nhân (điền tay):** _________________________________________________________________

### Nhóm: `generation_error` - Truy xuất đủ evidence nhưng trả lời sai/thiếu ý (1 ca)

#### Ca 1: [q_001] (Type: `direct`)
- **Câu hỏi**: Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có được mở tài khoản MoMo không?
- **Đáp án chuẩn (Gold)**: Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự được đăng ký, mở và sử dụng tài khoản theo quy định của MoMo phù hợp pháp luật.
- **Câu trả lời của hệ thống**: Có, người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có thể mở tài khoản MoMo theo điều kiện, phương thức và quy trình áp dụng của MoMo phù hợp
- **Đánh giá của Judge**: `partial` | Faithful: `True` | Refused: `False`
- **Lý do từ Judge**: Câu trả lời của hệ thống đúng một phần nhưng thiếu điều kiện quan trọng về việc người sử dụng không bị mất, hạn chế năng lực hành vi dân sự, điều này được nêu rõ trong đáp án chuẩn và ngữ cảnh [1].
- **Các đoạn trích đã truy xuất**:
  * `[momo_terms-A-018]` (Score: `0.8643`): Trường hợp Người Sử Dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự, việc đăng ký, mở và sử dụng tài khoản được thực hiện theo điều kiện, phương thức và quy trì...
  * `[momo_privacy-A-007]` (Score: `0.8388`): MoMo không hướng đến người dưới 15 tuổi. Việc xử lý dữ liệu cá nhân của người dưới 15 tuổi, người bị hạn chế hoặc mất năng lực hành vi dân sự được thực hiện thông qua người đại diện theo pháp luật the...
  * `[momo_terms-A-014]` (Score: `0.7555`): **1.15. Hồ Sơ Mở Tài Khoản:** là các tài liệu, thông tin, dữ liệu để định danh và xác minh thông tin nhận biết khách hàng mà Người Sử Dụng cung cấp khi mở Tài Khoản MoMo/Tài Khoản Ví Điện Tử, gồm tối ...

**Nguyên nhân (điền tay):** _________________________________________________________________


---
*Báo cáo được sinh tự động bởi `eval.failures`.*
