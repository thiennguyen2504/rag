# Báo cáo đánh giá Ragas: Run `run_B_50`

## 1. Tóm tắt chỉ số trung bình

| Chỉ số | Điểm số trung bình (0.0 - 1.0) | Ý nghĩa |
| :--- | :---: | :--- |
| **Faithfulness** | 0.9183 | Câu trả lời trung thực, không bịa đặt ngoài ngữ cảnh truy xuất |
| **Answer Relevancy** | 0.7036 | Mức độ bám sát câu hỏi người dùng |
| **Context Precision** | 0.9990 | Tỷ lệ các đoạn chính xác xuất hiện ở đầu bảng xếp hạng |
| **Context Recall** | 0.9901 | Ngữ cảnh truy xuất chứa đầy đủ bằng chứng đối chuẩn (ground truth) |

## 2. Chi tiết từng câu hỏi

| Question ID | Faithfulness | Relevancy | Context Precision | Context Recall | Câu hỏi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `q_001` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có được mở tài khoản MoMo không? |
| `q_002` | 0.7500 | 0.8529 | 1.0000 | 1.0000 | Hạn mức giao dịch và hạn mức ngày mặc định khi liên kết dịch vụ trên MoMo là bao nhiêu? |
| `q_003` | 1.0000 | 0.5500 | 1.0000 | 1.0000 | Ví MoMo có hỗ trợ tính năng mua bán tiền điện tử Bitcoin không? |
| `q_004` | 1.0000 | 0.8125 | 1.0000 | 1.0000 | Có đúng là sau khi liên kết tài khoản dịch vụ thì mỗi lần thanh toán người dùng vẫn phải mở ứng dụng MoMo để bấm xác nhận không? |
| `q_005` | 1.0000 | 0.6875 | 1.0000 | 0.9667 | Chào bạn, cho tôi biết MoMo là ứng dụng gì? |
| `q_006` | 1.0000 | 0.9167 | 1.0000 | 1.0000 | Số dư tối đa của Tài Khoản Ví Điện Tử MoMo tại mọi thời điểm là bao nhiêu? |
| `q_007` | 1.0000 | 0.8913 | 1.0000 | 1.0000 | Tổng hạn mức giao dịch chuyển tiền và thanh toán qua các tài khoản Ví cá nhân của một người dùng tối đa là bao nhiêu một tháng? |
| `q_008` | 1.0000 | 0.7292 | 1.0000 | 1.0000 | Hạn mức bổ sung tối đa cho nhóm giao dịch thanh toán đặc thù (điện, nước, viễn thông, học phí, viện phí) là bao nhiêu một tháng? |
| `q_009` | 1.0000 | 0.5556 | 1.0000 | 0.9286 | Công ty chủ quản sở hữu và vận hành ví MoMo có tên đầy đủ là gì? |
| `q_010` | 1.0000 | 0.7273 | 1.0000 | 1.0000 | Hồ sơ mở tài khoản MoMo tối thiểu bao gồm những thông tin nào? |
| `q_011` | 1.0000 | 0.6875 | 1.0000 | 0.9667 | Hạn mức giao dịch liên kết tối thiểu và tối đa trên MoMo có thể cài đặt là bao nhiêu? |
| `q_012` | 1.0000 | 0.6250 | 1.0000 | 1.0000 | Hạn mức ngày tối thiểu khi liên kết tài khoản đối tác trên MoMo là bao nhiêu? |
| `q_013` | 1.0000 | 0.7500 | 1.0000 | 1.0000 | Người dùng có thể hủy dịch vụ liên kết tài khoản ở đâu? |
| `q_014` | 1.0000 | 0.8889 | 1.0000 | 1.0000 | MoMo có hướng đến người dùng dưới 15 tuổi hay không? |
| `q_015` | 1.0000 | 0.6923 | 1.0000 | 1.0000 | Việc xử lý dữ liệu cá nhân của người dưới 15 tuổi trên MoMo được thực hiện qua ai? |
| `q_016` | 1.0000 | 0.7857 | 1.0000 | 1.0000 | Chủ thể dữ liệu có quyền yêu cầu xóa dữ liệu cá nhân của mình trên MoMo không? |
| `q_017` | 1.0000 | 0.7500 | 1.0000 | 1.0000 | Chủ thể dữ liệu có quyền rút lại sự đồng ý cho phép xử lý dữ liệu cá nhân trên MoMo không? |
| `q_018` | 1.0000 | 0.7143 | 1.0000 | 1.0000 | Người sử dụng nạp tiền vào Tài khoản ví điện tử MoMo thông qua những kênh nào? |
| `q_019` | 1.0000 | 0.6667 | 1.0000 | 1.0000 | Ngày làm việc theo quy định trong Điều khoản chung của MoMo là những ngày nào? |
| `q_020` | 1.0000 | 0.8333 | 1.0000 | 1.0000 | Người sử dụng có bắt buộc phải duy trì liên kết với tài khoản ngân hàng trong suốt thời gian sử dụng ví MoMo không? |
| `q_021` | 1.0000 | 0.7632 | 1.0000 | 1.0000 | Biện pháp xác thực sinh trắc học trên MoMo đối chiếu khớp đúng thông tin với những nguồn nào? |
| `q_022` | 1.0000 | 0.6250 | 1.0000 | 1.0000 | Điều khoản chung mở và sử dụng tài khoản MoMo áp dụng chính thức từ ngày nào? |
| `q_023` | 1.0000 | 0.5417 | 1.0000 | 1.0000 | MoMo thu thập những loại dữ liệu cá nhân cơ bản nào của khách hàng? |
| `q_024` | 1.0000 | 0.8889 | 1.0000 | 0.9815 | M_Service có chịu trách nhiệm về chất lượng sản phẩm do Nhà cung cấp dịch vụ cung cấp khi liên kết không? |
| `q_025` | 1.0000 | 0.7353 | 0.9500 | 1.0000 | Chào bạn, cho mình hỏi MoMo hỗ trợ chăm sóc khách hàng qua số tổng đài hotline nào vậy? |
| `q_026` | 0.5000 | 0.8333 | 1.0000 | 1.0000 | Người từ đủ 15 đến dưới 18 tuổi cần đáp ứng điều kiện gì về năng lực hành vi và người đại diện để đăng ký sử dụng MoMo? |
| `q_027` | 0.5000 | 0.6000 | 1.0000 | 1.0000 | Nếu giấy tờ tùy thân của khách hàng hết hiệu lực hoặc không cập nhật thông tin định danh thì MoMo có quyền áp dụng những biện pháp gì? |
| `q_028` | 0.5000 | 0.6364 | 1.0000 | 1.0000 | Người dùng có thể nạp tiền vào ví MoMo từ những nguồn nào và sau khi nạp có thể dùng số dư để chi trả cho những mục đích nào? |
| `q_029` | 0.5000 | 0.7885 | 1.0000 | 1.0000 | Khi hủy dịch vụ liên kết tài khoản đối tác, quan hệ thanh toán qua MoMo và các cam kết riêng với đối tác bị ảnh hưởng như thế nào? |
| `q_030` | 0.5000 | 0.6944 | 1.0000 | 0.9733 | Trong trường hợp nào MoMo có quyền trích nợ, khấu trừ hoặc thu hồi tiền từ tài khoản người dùng? |
| `q_031` | 0.5000 | 0.7917 | 1.0000 | 1.0000 | MoMo chia sẻ dữ liệu cá nhân của người dùng trong những trường hợp nào? |
| `q_032` | 0.5000 | 0.7105 | 1.0000 | 0.9524 | Người dùng cần làm những bước gì để từ một tài khoản MoMo cơ bản trở thành Tài khoản Ví điện tử MoMo chính thức? |
| `q_033` | 0.6667 | 0.7045 | 1.0000 | 0.9841 | Trách nhiệm bồi thường tổn thất trong trường hợp xảy ra giao dịch trái phép trên MoMo được phân định như thế nào? |
| `q_034` | 1.0000 | 0.5333 | 1.0000 | 1.0000 | Thủ tục xin cấp visa định cư tại Canada qua ứng dụng MoMo gồm những giấy tờ gì? |
| `q_035` | 1.0000 | 0.5882 | 1.0000 | 1.0000 | Ví MoMo có cung cấp dịch vụ đổi trực tiếp tiền mặt Euro sang tiền Yên Nhật tại quầy không? |
| `q_036` | 1.0000 | 0.5000 | 1.0000 | 1.0000 | Lãi suất tiền gửi tiết kiệm kỳ hạn 12 tháng tại Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam hiện tại là bao nhiêu? |
| `q_037` | 1.0000 | 0.5000 | 1.0000 | 1.0000 | Làm thế nào để kích hoạt tính năng đào tiền ảo Ethereum trên nền tảng MoMo? |
| `q_038` | 1.0000 | 0.5263 | 1.0000 | 1.0000 | Quy định về thời hạn cấp phép xây dựng nhà ở riêng lẻ tại TP.HCM theo Luật Xây dựng là bao nhiêu ngày? |
| `q_039` | 1.0000 | 0.5278 | 1.0000 | 1.0000 | Chính sách bảo hành và sửa chữa xe ô tô điện Tesla tại trạm sạc MoMo được quy định ở điều khoản nào? |
| `q_040` | 1.0000 | 0.5714 | 1.0000 | 1.0000 | MoMo có hỗ trợ mở tài khoản ngân hàng tại Thụy Sĩ cho công dân Việt Nam không? |
| `q_041` | 1.0000 | 0.5000 | 1.0000 | 1.0000 | Hướng dẫn cách kết nối ví MoMo trực tiếp với mạng vệ tinh Starlink của SpaceX? |
| `q_042` | 1.0000 | 0.5000 | 1.0000 | 1.0000 | Mức xử phạt hành chính đối với hành vi điều khiển xe máy vượt đèn đỏ theo Nghị định 100 là bao nhiêu? |
| `q_043` | 1.0000 | 0.7000 | 1.0000 | 1.0000 | Hi MoMo, cho mình hỏi ngoài hotline thì người dùng có thể gửi khiếu nại qua tính năng nào trên ứng dụng? |
| `q_044` | 1.0000 | 0.7857 | 1.0000 | 0.9828 | Người từ đủ 12 tuổi đến dưới 15 tuổi có được tự đứng tên đăng ký mở tài khoản ví điện tử MoMo không? |
| `q_045` | 1.0000 | 0.6818 | 1.0000 | 0.9455 | MoMo có cam kết hoàn trả 100% tiền thanh toán khi người dùng hủy mua hàng trên ứng dụng của bên thứ ba không? |
| `q_046` | 1.0000 | 0.6500 | 1.0000 | 1.0000 | Số dư trong tài khoản ví MoMo có được phép vượt quá 500 triệu đồng nếu khách hàng đã xác thực sinh trắc học không? |
| `q_047` | 1.0000 | 0.8824 | 1.0000 | 0.9778 | Có phải hạn mức thanh toán và chuyển tiền qua ví MoMo của mỗi cá nhân là không giới hạn mỗi tháng? |
| `q_048` | 1.0000 | 0.8542 | 1.0000 | 0.8462 | Khi đã liên kết tài khoản MoMo với đối tác, người dùng có được phép hủy liên kết hay bắt buộc phải duy trì vĩnh viễn? |
| `q_049` | 1.0000 | 0.8333 | 1.0000 | 1.0000 | MoMo có được tự ý chia sẻ thông tin cá nhân của người dùng cho bất kỳ bên thứ ba nào mà không cần sự đồng ý hay căn cứ pháp lý không? |
| `q_050` | 1.0000 | 0.6154 | 1.0000 | 1.0000 | Xin chào trợ lý, bạn có thể cho tôi biết dịch vụ Ví điện tử MoMo là gì không? |
