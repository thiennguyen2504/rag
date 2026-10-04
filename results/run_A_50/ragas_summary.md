# Báo cáo đánh giá Ragas: Run `run_A_50`

## 1. Tóm tắt chỉ số trung bình

| Chỉ số | Điểm số trung bình (0.0 - 1.0) | Ý nghĩa |
| :--- | :---: | :--- |
| **Faithfulness** | 0.8000 | Câu trả lời trung thực, không bịa đặt ngoài ngữ cảnh truy xuất |
| **Answer Relevancy** | 0.7076 | Mức độ bám sát câu hỏi người dùng |
| **Context Precision** | 1.0000 | Tỷ lệ các đoạn chính xác xuất hiện ở đầu bảng xếp hạng |
| **Context Recall** | 0.9771 | Ngữ cảnh truy xuất chứa đầy đủ bằng chứng đối chuẩn (ground truth) |

## 2. Chi tiết từng câu hỏi

| Question ID | Faithfulness | Relevancy | Context Precision | Context Recall | Câu hỏi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `q_001` | 1.0000 | 0.7353 | 1.0000 | 1.0000 | Bạn trẻ đang là học sinh khoảng 16 tuổi có được tự đứng tên đăng ký và sử dụng tài khoản ví MoMo hay không? |
| `q_002` | 1.0000 | 0.6724 | 1.0000 | 0.9792 | Theo thiết lập mặc định của MoMo, khi liên kết với dịch vụ đối tác thì số tiền tối đa có thể trả cho một lần thanh toán và cho cả một ngày là bao nhiêu? |
| `q_003` | 1.0000 | 0.7692 | 1.0000 | 1.0000 | Chính sách quản lý rủi ro của MoMo quy định lượng tiền tối đa được phép lưu giữ lại trong tài khoản ví điện tử tại mọi thời điểm là bao nhiêu? |
| `q_004` | 1.0000 | 0.7037 | 1.0000 | 1.0000 | Trong một tháng, một khách hàng cá nhân được chuyển tiền và chi tiêu thanh toán qua các tài khoản ví MoMo tổng cộng không vượt quá bao nhiêu tiền? |
| `q_005` | 1.0000 | 0.5833 | 1.0000 | 1.0000 | Đối với các khoản chi đặc thù như tiền điện, nước, cước viễn thông, học phí hay viện phí thì hạn mức bổ sung hàng tháng của MoMo tối đa là bao nhiêu? |
| `q_006` | 1.0000 | 0.6406 | 1.0000 | 1.0000 | Khi người dùng chủ động điều chỉnh cài đặt hạn mức liên kết đối tác trên MoMo, mức tối thiểu được phép cài đặt cho mỗi giao dịch đơn lẻ là bao nhiêu tiền? |
| `q_007` | 1.0000 | 0.6042 | 1.0000 | 1.0000 | Mức thanh toán tối thiểu tính theo ngày mà người dùng có thể thiết lập cho một tài khoản dịch vụ liên kết với MoMo là bao nhiêu? |
| `q_008` | 1.0000 | 0.7800 | 1.0000 | 1.0000 | Các ngày thứ Bảy, Chủ Nhật cùng các đợt nghỉ Tết hay nghỉ lễ quốc gia có được MoMo tính là Ngày Làm Việc theo quy định không? |
| `q_009` | 1.0000 | 0.7292 | 1.0000 | 1.0000 | Người dùng có bắt buộc phải liên kết ví MoMo với tài khoản ngân hàng và duy trì kết nối đó trong toàn bộ quá trình sử dụng dịch vụ không? |
| `q_010` | 1.0000 | 0.5682 | 1.0000 | 1.0000 | Một cá nhân khi tạo lập ví MoMo cần phải cung cấp những thông tin cơ bản tối thiểu nào trong hồ sơ nhận diện khách hàng? |
| `q_011` | 1.0000 | 0.6000 | 1.0000 | 1.0000 | Khi tiến hành xác thực khuôn mặt hoặc vân tay, hệ thống MoMo sẽ đối chiếu dữ liệu sinh trắc học của khách hàng với những nguồn nào? |
| `q_012` | 1.0000 | 0.7353 | 1.0000 | 0.9000 | Ứng dụng MoMo có chủ trương định hướng phục vụ nhóm đối tượng khách hàng dưới 15 tuổi hay không? |
| `q_013` | 1.0000 | 0.6667 | 1.0000 | 1.0000 | Chủ tài khoản có quyền gửi yêu cầu đề nghị MoMo xóa bỏ dữ liệu cá nhân của mình trong hệ thống hay không? |
| `q_014` | 1.0000 | 0.6562 | 1.0000 | 1.0000 | Sau khi đã chấp thuận cho MoMo xử lý dữ liệu cá nhân, khách hàng có được quyền rút lại sự đồng ý đó không? |
| `q_015` | 1.0000 | 0.7045 | 1.0000 | 0.9487 | Những loại dữ liệu cơ bản nào liên quan đến thông tin định danh và liên lạc được MoMo thu thập để vận hành tài khoản? |
| `q_016` | 1.0000 | 0.8696 | 1.0000 | 1.0000 | MoMo có được chia sẻ thông tin khách hàng cho bên thứ ba nhằm mục đích tiếp thị trực tiếp của bên đó không, và có ngoại lệ nào không? |
| `q_017` | 1.0000 | 0.6136 | 1.0000 | 0.8000 | Bản Điều khoản và điều kiện mở và sử dụng tài khoản MoMo hiện hành được công bố bắt đầu có hiệu lực áp dụng từ ngày tháng năm nào? |
| `q_018` | 1.0000 | 0.6765 | 1.0000 | 0.9839 | Khách hàng có thể nạp tiền vào ví điện tử MoMo thông qua những kênh giao dịch hợp pháp nào? |
| `q_019` | 1.0000 | 0.6136 | 1.0000 | 1.0000 | Người dùng có thể thực hiện thao tác ngắt kết nối liên kết thanh toán MoMo với tài khoản đối tác ở những đâu? |
| `q_020` | 1.0000 | 0.7115 | 1.0000 | 0.9815 | Khi xảy ra lỗi hư hỏng hay tranh chấp về sản phẩm của Nhà cung cấp dịch vụ đối tác, M_Service có chịu trách nhiệm bảo hành hay hoàn tiền không? |
| `q_021` | 1.0000 | 0.7400 | 1.0000 | 0.9710 | Thiếu niên từ 15 đến dưới 18 tuổi muốn mở và dùng MoMo cần đồng thời thỏa mãn những điều kiện gì về năng lực dân sự và thủ tục với người đại diện? |
| `q_022` | 1.0000 | 0.6935 | 1.0000 | 0.9296 | Hạn mức thanh toán các loại phí công ích, điện nước, viện phí, học phí trên MoMo khác gì so với tổng hạn mức giao dịch chuyển tiền và chi tiêu cá nhân mỗi tháng? |
| `q_023` | 1.0000 | 0.6207 | 1.0000 | 1.0000 | Nếu giấy tờ tùy thân của chủ ví MoMo bị hết hiệu lực và khách hàng không chịu cập nhật xác minh lại thì MoMo có quyền áp dụng những biện pháp chế tài nào? |
| `q_024` | 1.0000 | 0.7069 | 1.0000 | 0.9630 | Các con số hạn mức mặc định và hạn mức tối thiểu khi liên kết thanh toán MoMo với tài khoản đối tác khác nhau ra sao giữa từng giao dịch và cả ngày? |
| `q_025` | 1.0000 | 0.7069 | 1.0000 | 0.9747 | Trong những trường hợp cụ thể nào MoMo được quyền tự ý trích nợ, khấu trừ hoặc thu hồi tiền từ tài khoản của người dùng mà không cần người dùng xác nhận lại? |
| `q_026` | 1.0000 | 0.7143 | 1.0000 | 1.0000 | Trách nhiệm đối với giao dịch trái phép được phân chia thế nào và người dùng có phải chịu thiệt hại nếu nguyên nhân xuất phát từ sự cố hệ thống của MoMo không? |
| `q_027` | 1.0000 | 0.6897 | 1.0000 | 0.9167 | Để một tài khoản MoMo thông thường được nâng cấp và kích hoạt đầy đủ thành Tài Khoản Ví Điện Tử MoMo chính thức thì người dùng cần hoàn thành những thủ tục nào? |
| `q_028` | 1.0000 | 0.7500 | 1.0000 | 0.9419 | Khi người dùng bấm hủy liên kết thanh toán MoMo với một bên thứ ba thì quyền thanh toán qua ví và các nghĩa vụ hợp đồng riêng với đối tác đó sẽ ra sao? |
| `q_029` | 1.0000 | 0.6458 | 1.0000 | 0.9242 | MoMo được phép cung cấp hoặc chia sẻ thông tin dữ liệu của khách hàng cho các cơ quan, tổ chức bên ngoài trong những tình huống nào? |
| `q_030` | 0.0000 | 0.8542 | 1.0000 | 1.0000 | Biểu phí rút tiền từ số dư ví MoMo về tài khoản ngân hàng liên kết khi vượt quá định mức miễn phí trong tháng là bao nhiêu phần trăm? |
| `q_031` | 0.0000 | 0.8750 | 1.0000 | 1.0000 | Chính sách chuyển tiền giữa hai tài khoản ví MoMo với nhau hiện nay cho phép miễn phí tối đa bao nhiêu lượt chuyển mỗi tháng? |
| `q_032` | 0.0000 | 0.8036 | 1.0000 | 1.0000 | Tỷ suất sinh lời hay lãi suất tiền gửi của sản phẩm tích lũy Túi Thần Tài trên ứng dụng MoMo hiện tại là bao nhiêu phần trăm một năm? |
| `q_033` | 0.0000 | 0.8409 | 1.0000 | 1.0000 | Hạn mức chi tiêu ban đầu được cấp khi người dùng đăng ký mở dịch vụ Ví Trả Sau trên MoMo tối đa là bao nhiêu triệu đồng? |
| `q_034` | 0.0000 | 0.7917 | 1.0000 | 1.0000 | Mức phí duy trì dịch vụ hoặc phí quản lý tài khoản của Ví Trả Sau MoMo là bao nhiêu tiền mỗi tháng nếu có phát sinh giao dịch? |
| `q_035` | 0.0000 | 0.8800 | 1.0000 | 1.0000 | Thời gian ân hạn miễn lãi tối đa cho các hóa đơn mua sắm thanh toán qua Ví Trả Sau của MoMo là bao nhiêu ngày kể từ ngày chốt sao kê? |
| `q_036` | 0.0000 | 0.8478 | 1.0000 | 1.0000 | Để tích lũy và đổi được một Heo Vàng quyên góp trong chương trình Heo Đất MoMo thì người dùng cần bao nhiêu gam thức ăn? |
| `q_037` | 0.0000 | 0.7500 | 1.0000 | 1.0000 | Tỷ lệ quy đổi điểm thưởng MoMo Rewards sang tiền mặt hoặc mã giảm giá trên ứng dụng MoMo là bao nhiêu điểm tương đương 1 đồng? |
| `q_038` | 0.0000 | 0.7308 | 1.0000 | 1.0000 | Hạn mức vay tiền mặt tiêu dùng nhanh FastMoney do đối tác tài chính liên kết trên ứng dụng MoMo cung cấp tối đa là bao nhiêu triệu đồng? |
| `q_039` | 0.0000 | 0.8478 | 1.0000 | 1.0000 | Khách hàng sử dụng thẻ tín dụng quốc tế Visa hoặc Mastercard để nạp tiền vào ví MoMo có bị trừ phí nạp không và mức phí cụ thể là bao nhiêu? |
| `q_040` | 1.0000 | 0.6500 | 1.0000 | 0.9836 | Một học sinh 14 tuổi đã có CCCD gắn chip thì có được tự mình đăng ký và mở tài khoản ví điện tử MoMo mà không cần người giám hộ không? |
| `q_041` | 1.0000 | 0.7273 | 1.0000 | 1.0000 | Sau khi đã cài đặt liên kết ví MoMo với ứng dụng của bên thứ ba, có phải mỗi lần đặt đơn hàng người dùng vẫn phải mở app MoMo nhập mã OTP để xác nhận trừ tiền không? |
| `q_042` | 1.0000 | 0.6071 | 1.0000 | 0.9714 | Nếu một khách hàng đã hoàn thành xác thực sinh trắc học cấp cao và nâng cấp tài khoản thì số dư trong ví MoMo có được phép vượt quá 300 triệu đồng không? |
| `q_043` | 1.0000 | 0.8810 | 1.0000 | 0.9792 | Có phải hạn mức thanh toán và chuyển tiền qua ví MoMo của mỗi cá nhân là không giới hạn mỗi tháng nếu số dư tài khoản đủ tiền? |
| `q_044` | 1.0000 | 0.6724 | 1.0000 | 0.9149 | Sau khi đã bấm xác nhận liên kết tài khoản MoMo với nhà cung cấp dịch vụ, người dùng có bắt buộc phải duy trì liên kết này vĩnh viễn không? |
| `q_045` | 1.0000 | 0.6923 | 1.0000 | 0.9167 | Khi người dùng hủy mua vé xem phim hoặc đồ ăn trên ứng dụng của bên thứ ba đã liên kết, MoMo có nghĩa vụ bồi hoàn lại toàn bộ số tiền thanh toán không? |
| `q_046` | 1.0000 | 0.6852 | 1.0000 | 1.0000 | MoMo có quyền tự ý chuyển giao dữ liệu cá nhân của khách hàng cho các nhãn hàng bên ngoài để họ chạy quảng cáo tiếp thị trực tiếp mà không cần xin phép không? |
| `q_047` | 1.0000 | 0.5278 | 1.0000 | 0.8750 | Xin chào bạn, cho mình hỏi tên gọi đầy đủ và pháp nhân chính thức của công ty vận hành ví MoMo là gì? |
| `q_048` | 1.0000 | 0.5227 | 1.0000 | 1.0000 | Chào MoMo, nếu mình cần hỗ trợ kỹ thuật hoặc khiếu nại khẩn cấp thì số điện thoại tổng đài hotline chính thức là bao nhiêu? |
| `q_049` | 1.0000 | 0.6786 | 1.0000 | 1.0000 | Chào trợ lý, ngoài gọi điện lên hotline thì người dùng có thể gửi yêu cầu hỗ trợ hoặc phản ánh về quyền dữ liệu qua những kênh nào trên ứng dụng? |
| `q_050` | 1.0000 | 0.6111 | 1.0000 | 1.0000 | Hi bạn, cho mình hỏi định nghĩa ngắn gọn ứng dụng MoMo là gì theo quy định của điều khoản dịch vụ? |
