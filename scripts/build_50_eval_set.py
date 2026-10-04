"""Generate a challenging 50-question evaluation set for MoMo RAG meeting all user criteria:
1. Diễn đạt khác với văn bản gốc (paraphrased, natural user phrasing).
2. Câu trả lời nằm ở cuối một mục dài mà chunk 500 ký tự cắt đôi (splitting context challenge).
3. Nhiều đoạn gần giống nhau gây nhiễu (hạn mức theo từng dịch vụ, các con số kề nhau).
4. Câu hỏi về điều kiện, ngoại lệ và con số chính xác.
5. Câu ngoài phạm vi dễ gây bịa (tính năng MoMo có thật: phí rút tiền, lãi suất Túi Thần Tài, Ví Trả Sau, Heo Đất...
   nhưng không có trong tài liệu crawl, khiến prompt v1 dễ hallucinate còn prompt v2 từ chối nghiêm ngặt).
"""

import json
from pathlib import Path
import re
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_corpus():
    corpus = {}
    for p in PROCESSED_DIR.glob("*.md"):
        with open(p, "r", encoding="utf-8") as f:
            corpus[p.stem] = f.read()
    return corpus


def build_questions():
    corpus = load_corpus()

    def find_snippet(doc_name, phrase):
        doc = corpus[doc_name]
        pos = doc.lower().find(phrase.lower())
        if pos == -1:
            raise ValueError(f"Phrase '{phrase[:40]}...' not found in {doc_name}")
        snippet = doc[pos : pos + len(phrase)]
        return snippet

    questions = [
        # =========================================================================
        # 1. DIRECT QUESTIONS (20 items: q_001 to q_020)
        # =========================================================================
        {
            "id": "q_001",
            "question": "Bạn trẻ đang là học sinh khoảng 16 tuổi có được tự đứng tên đăng ký và sử dụng tài khoản ví MoMo hay không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Người từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự được đăng ký, mở và sử dụng tài khoản theo điều kiện, phương thức của MoMo phù hợp quy định pháp luật.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Trường hợp Người Sử Dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự, việc đăng ký, mở và sử dụng tài khoản được thực hiện theo điều kiện, phương thức")],
            "notes": "Paraphrase điều kiện độ tuổi 15-18",
        },
        {
            "id": "q_002",
            "question": "Theo thiết lập mặc định của MoMo, khi liên kết với dịch vụ đối tác thì số tiền tối đa có thể trả cho một lần thanh toán và cho cả một ngày là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Mặc định hạn mức cho một lần giao dịch là tối đa 50.000đ và tổng hạn mức ngày tối đa là 250.000đ mỗi ngày.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán các giao dịch tối đa là 50.000đ"),
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán tổng cộng tối đa là 250.000đ mỗi ngày"),
            ],
            "notes": "Nhiễu hạn mức liên kết mặc định",
        },
        {
            "id": "q_003",
            "question": "Chính sách quản lý rủi ro của MoMo quy định lượng tiền tối đa được phép lưu giữ lại trong tài khoản ví điện tử tại mọi thời điểm là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo")],
            "notes": "Cắt đôi chunk 500 ký tự ở cuối điều 5.3",
        },
        {
            "id": "q_004",
            "question": "Trong một tháng, một khách hàng cá nhân được chuyển tiền và chi tiêu thanh toán qua các tài khoản ví MoMo tổng cộng không vượt quá bao nhiêu tiền?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng")],
            "notes": "Hạn mức tháng thông thường",
        },
        {
            "id": "q_005",
            "question": "Đối với các khoản chi đặc thù như tiền điện, nước, cước viễn thông, học phí hay viện phí thì hạn mức bổ sung hàng tháng của MoMo tối đa là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Tổng hạn mức cho nhóm giao dịch thanh toán đặc thù không vượt quá 300.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "tổng hạn mức cho nhóm này không vượt quá 300.000.000 đồng/tháng")],
            "notes": "Con số hạn mức đặc thù",
        },
        {
            "id": "q_006",
            "question": "Khi người dùng chủ động điều chỉnh cài đặt hạn mức liên kết đối tác trên MoMo, mức tối thiểu được phép cài đặt cho mỗi giao dịch đơn lẻ là bao nhiêu tiền?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hạn mức tối thiểu cho mỗi giao dịch là 25.000đ và hạn mức tối đa là không giới hạn.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hạn mức tối thiểu là 25.000đ và hạn mức tối đa là không giới hạn")],
            "notes": "Nhiễu hạn mức giao dịch tối thiểu",
        },
        {
            "id": "q_007",
            "question": "Mức thanh toán tối thiểu tính theo ngày mà người dùng có thể thiết lập cho một tài khoản dịch vụ liên kết với MoMo là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hạn mức ngày tối thiểu có thể cài đặt là 150.000đ và tối đa là không giới hạn.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hạn mức tối thiểu là 150.000đ và hạn mức tối đa là không giới hạn")],
            "notes": "Nhiễu hạn mức ngày tối thiểu",
        },
        {
            "id": "q_008",
            "question": "Các ngày thứ Bảy, Chủ Nhật cùng các đợt nghỉ Tết hay nghỉ lễ quốc gia có được MoMo tính là Ngày Làm Việc theo quy định không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Không, Ngày Làm Việc là các ngày từ Thứ Hai đến Thứ Sáu, không bao gồm ngày nghỉ, lễ, Tết theo quy định pháp luật.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Ngày Làm Việc:** là các ngày từ Thứ Hai đến Thứ Sáu, không bao gồm ngày nghỉ, lễ, Tết theo quy định pháp luật")],
            "notes": "Paraphrase định nghĩa Ngày Làm Việc",
        },
        {
            "id": "q_009",
            "question": "Người dùng có bắt buộc phải liên kết ví MoMo với tài khoản ngân hàng và duy trì kết nối đó trong toàn bộ quá trình sử dụng dịch vụ không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, Người Sử Dụng phải hoàn thành việc liên kết và duy trì liên kết trong suốt thời gian sử dụng Tài Khoản Ví Điện Tử trừ trường hợp pháp luật quy định khác.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "duy trì liên kết trong suốt thời gian sử dụng Tài Khoản Ví Điện Tử")],
            "notes": "Điều kiện duy trì liên kết ngân hàng",
        },
        {
            "id": "q_010",
            "question": "Một cá nhân khi tạo lập ví MoMo cần phải cung cấp những thông tin cơ bản tối thiểu nào trong hồ sơ nhận diện khách hàng?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hồ Sơ Mở Tài Khoản gồm tối thiểu: họ và tên; ngày, tháng, năm sinh; quốc tịch; số điện thoại; số định danh cá nhân hoặc giấy tờ tùy thân còn hiệu lực; dữ liệu sinh trắc học.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "gồm tối thiểu: họ và tên; ngày, tháng, năm sinh; quốc tịch; số điện thoại; số định danh cá nhân hoặc giấy tờ tùy thân còn hiệu lực; dữ liệu sinh trắc học")],
            "notes": "Danh sách thông tin định danh bắt buộc",
        },
        {
            "id": "q_011",
            "question": "Khi tiến hành xác thực khuôn mặt hoặc vân tay, hệ thống MoMo sẽ đối chiếu dữ liệu sinh trắc học của khách hàng với những nguồn nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Đối chiếu khớp đúng với dữ liệu trong giấy tờ tùy thân, danh tính điện tử (tài khoản định danh điện tử VneID) hoặc cơ sở dữ liệu có thẩm quyền.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "đối chiếu khớp đúng thông tin sinh trắc học của Người Sử Dụng với dữ liệu trong giấy tờ tùy thân, danh tính điện tử (tài khoản định danh điện tử (VneID))")],
            "notes": "Nguồn đối chiếu sinh trắc học",
        },
        {
            "id": "q_012",
            "question": "Ứng dụng MoMo có chủ trương định hướng phục vụ nhóm đối tượng khách hàng dưới 15 tuổi hay không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "MoMo không hướng đến người dưới 15 tuổi, và việc xử lý dữ liệu cho nhóm đối tượng này phải thông qua người đại diện theo pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không hướng đến người dưới 15 tuổi")],
            "notes": "Chính sách đối tượng dưới 15 tuổi",
        },
        {
            "id": "q_013",
            "question": "Chủ tài khoản có quyền gửi yêu cầu đề nghị MoMo xóa bỏ dữ liệu cá nhân của mình trong hệ thống hay không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, Người dùng có quyền yêu cầu xóa dữ liệu trong phạm vi pháp luật cho phép.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Quyền yêu cầu xóa dữ liệu trong phạm vi pháp luật cho phép")],
            "notes": "Quyền yêu cầu xóa dữ liệu",
        },
        {
            "id": "q_014",
            "question": "Sau khi đã chấp thuận cho MoMo xử lý dữ liệu cá nhân, khách hàng có được quyền rút lại sự đồng ý đó không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, Người dùng có quyền rút lại sự đồng ý theo quy định pháp luật về bảo vệ dữ liệu cá nhân.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Quyền rút lại sự đồng ý")],
            "notes": "Quyền rút lại sự đồng ý",
        },
        {
            "id": "q_015",
            "question": "Những loại dữ liệu cơ bản nào liên quan đến thông tin định danh và liên lạc được MoMo thu thập để vận hành tài khoản?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Bao gồm thông tin định danh cá nhân, thông tin liên lạc, thông tin tài khoản dịch vụ, thông tin sinh trắc học và thông tin tài chính.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Thông tin định danh cá nhân, thông tin liên lạc, thông tin tài khoản dịch vụ.")],
            "notes": "Các nhóm dữ liệu cá nhân",
        },
        {
            "id": "q_016",
            "question": "MoMo có được chia sẻ thông tin khách hàng cho bên thứ ba nhằm mục đích tiếp thị trực tiếp của bên đó không, và có ngoại lệ nào không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "MoMo không chia sẻ thông tin cho bên thứ ba cho mục đích tiếp thị trực tiếp của họ, trừ khi được Người dùng đồng ý hoặc theo quy định của pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không chia sẻ thông tin Người dùng cho bên thứ ba cho mục đích tiếp thị trực tiếp của họ, trừ khi được Người dùng đồng ý hoặc khi việc chia sẻ được thực hiện theo quy định của pháp luật")],
            "notes": "Điều kiện tiếp thị trực tiếp bên thứ ba",
        },
        {
            "id": "q_017",
            "question": "Bản Điều khoản và điều kiện mở và sử dụng tài khoản MoMo hiện hành được công bố bắt đầu có hiệu lực áp dụng từ ngày tháng năm nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Bắt đầu áp dụng chính thức từ ngày 30/09/2026.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Áp dụng từ ngày 30/09/2026")],
            "notes": "Ngày hiệu lực Điều khoản chung",
        },
        {
            "id": "q_018",
            "question": "Khách hàng có thể nạp tiền vào ví điện tử MoMo thông qua những kênh giao dịch hợp pháp nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Thông qua nhận tiền từ Tài Khoản Ngân Hàng Liên Kết, tài khoản thanh toán VND, từ Tài Khoản Ví Điện Tử khác hoặc ví điện tử khác khi pháp luật cho phép.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Nạp tiền vào Tài Khoản Ví Điện Tử** được thực hiện thông qua: (a) nhận tiền từ Tài Khoản Ngân Hàng Liên Kết của Người Sử Dụng; (b) nhận tiền từ tài khoản thanh toán bằng đồng Việt Nam")],
            "notes": "Kênh nạp tiền hợp pháp",
        },
        {
            "id": "q_019",
            "question": "Người dùng có thể thực hiện thao tác ngắt kết nối liên kết thanh toán MoMo với tài khoản đối tác ở những đâu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Quý khách có thể hủy dịch vụ liên kết bất kỳ lúc nào ngay trên Ứng dụng Ví điện tử MoMo hoặc hủy ở web, ứng dụng của Nhà cung cấp dịch vụ.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "hủy dịch vụ liên kết bất kỳ lúc nào ngay trên Ứng dụng Ví điện tử MoMo hoặc hủy ở web, ứng dụng của Nhà cung cấp dịch vụ")],
            "notes": "Nơi thực hiện hủy liên kết",
        },
        {
            "id": "q_020",
            "question": "Khi xảy ra lỗi hư hỏng hay tranh chấp về sản phẩm của Nhà cung cấp dịch vụ đối tác, M_Service có chịu trách nhiệm bảo hành hay hoàn tiền không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Không, M_Service là trung gian thanh toán và không chịu bất kỳ trách nhiệm liên quan nào về sản phẩm, dịch vụ do Nhà cung cấp dịch vụ cung cấp.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "M\\_Service là trung gian thanh toán giữa Quý khách và Nhà cung cấp dịch vụ, vì vậy M\\_Service không chịu bất kỳ trách nhiệm liên quan nào về sản phẩm, dịch vụ do Nhà cung cấp dịch vụ cung cấp")],
            "notes": "Trách nhiệm trung gian thanh toán",
        },

        # =========================================================================
        # 2. MULTI-HOP QUESTIONS (9 items: q_021 to q_029)
        # =========================================================================
        {
            "id": "q_021",
            "question": "Thiếu niên từ 15 đến dưới 18 tuổi muốn mở và dùng MoMo cần đồng thời thỏa mãn những điều kiện gì về năng lực dân sự và thủ tục với người đại diện?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Người dùng từ đủ 15 đến chưa đủ 18 tuổi phải không bị mất, hạn chế năng lực hành vi dân sự, đồng thời MoMo có thể yêu cầu thêm sự chấp thuận của người đại diện theo pháp luật tùy theo tính chất dịch vụ.",
            "gold_doc_ids": ["momo_terms", "momo_privacy"],
            "gold_evidence": [
                find_snippet("momo_terms", "Người Sử Dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự"),
                find_snippet("momo_privacy", "Đối với người từ đủ 15 đến dưới 18 tuổi, MoMo có thể yêu cầu thêm sự chấp thuận của người đại diện theo pháp luật"),
            ],
            "notes": "Multi-hop kết hợp momo_terms và momo_privacy",
        },
        {
            "id": "q_022",
            "question": "Hạn mức thanh toán các loại phí công ích, điện nước, viện phí, học phí trên MoMo khác gì so với tổng hạn mức giao dịch chuyển tiền và chi tiêu cá nhân mỗi tháng?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Tổng hạn mức giao dịch cá nhân thông thường tối đa là 100.000.000 đồng/tháng, còn nhóm giao dịch thanh toán đặc thù được cấp hạn mức bổ sung với tổng không vượt quá 300.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng"),
                find_snippet("momo_terms", "tổng hạn mức cho nhóm này không vượt quá 300.000.000 đồng/tháng"),
            ],
            "notes": "Nhiễu hạn mức cá nhân vs đặc thù",
        },
        {
            "id": "q_023",
            "question": "Nếu giấy tờ tùy thân của chủ ví MoMo bị hết hiệu lực và khách hàng không chịu cập nhật xác minh lại thì MoMo có quyền áp dụng những biện pháp chế tài nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "MoMo có quyền hạn chế, tạm ngừng, từ chối thực hiện một phần hoặc toàn bộ Giao Dịch, hoặc đóng tài khoản của Người Sử Dụng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Trường hợp Người Sử Dụng không cung cấp, cập nhật hoặc xác minh thông tin theo yêu cầu, hoặc giấy tờ tùy thân hết hiệu lực"),
                find_snippet("momo_terms", "MoMo có thể hạn chế, tạm ngừng, từ chối thực hiện một phần hoặc toàn bộ Giao Dịch, hoặc đóng tài khoản"),
            ],
            "notes": "Cắt đôi chunk cuối điều 3.4",
        },
        {
            "id": "q_024",
            "question": "Các con số hạn mức mặc định và hạn mức tối thiểu khi liên kết thanh toán MoMo với tài khoản đối tác khác nhau ra sao giữa từng giao dịch và cả ngày?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Hạn mức từng giao dịch: tối thiểu 25.000đ, mặc định tối đa 50.000đ. Hạn mức ngày: tối thiểu 150.000đ, mặc định tối đa 250.000đ mỗi ngày.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán các giao dịch tối đa là 50.000đ"),
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán tổng cộng tối đa là 250.000đ mỗi ngày"),
            ],
            "notes": "Nhiễu hạn mức liên kết 4 con số",
        },
        {
            "id": "q_025",
            "question": "Trong những trường hợp cụ thể nào MoMo được quyền tự ý trích nợ, khấu trừ hoặc thu hồi tiền từ tài khoản của người dùng mà không cần người dùng xác nhận lại?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Khi có căn cứ xác định người dùng phải thanh toán hoặc hoàn trả; khoản tiền ghi có nhầm hoặc phát sinh do sai sót kỹ thuật; và nghĩa vụ theo yêu cầu hợp pháp của cơ quan có thẩm quyền.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "MoMo chỉ trích nợ, khấu trừ hoặc thu hồi tiền trong trong phạm vi khoản tiền khi có căn cứ xác định Người Sử Dụng phải thanh toán hoặc hoàn trả"),
                find_snippet("momo_terms", "khoản tiền ghi có nhầm hoặc phát sinh do sai sót kỹ thuật; và (c) nghĩa vụ theo yêu cầu hợp pháp của cơ quan có thẩm quyền"),
            ],
            "notes": "Điều kiện và ngoại lệ trích nợ tự động",
        },
        {
            "id": "q_026",
            "question": "Trách nhiệm đối với giao dịch trái phép được phân chia thế nào và người dùng có phải chịu thiệt hại nếu nguyên nhân xuất phát từ sự cố hệ thống của MoMo không?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Trách nhiệm xác định trên cơ sở lỗi của các bên. Người dùng không phải chịu trách nhiệm đối với tổn thất do lỗi hoặc sự cố hệ thống của MoMo, và MoMo có trách nhiệm bồi hoàn.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Trách nhiệm đối với Giao Dịch trái phép được xác định trên cơ sở lỗi của các bên"),
                find_snippet("momo_terms", "Người Sử Dụng không phải chịu trách nhiệm đối với những tổn thất phát sinh do lỗi của MoMo, sự cố hệ thống của MoMo"),
            ],
            "notes": "Cắt đôi chunk cuối mục 6.2",
        },
        {
            "id": "q_027",
            "question": "Để một tài khoản MoMo thông thường được nâng cấp và kích hoạt đầy đủ thành Tài Khoản Ví Điện Tử MoMo chính thức thì người dùng cần hoàn thành những thủ tục nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Người dùng phải hoàn tất quy trình đăng ký, định danh, xác thực thông tin và liên kết thành công với tài khoản ngân hàng của chính mình.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "sau khi Người Sử Dụng hoàn tất quy trình đăng ký, định danh, xác thực và liên kết tài khoản ngân hàng"),
                find_snippet("momo_terms", "định danh và Xác Thực Sinh Trắc Học theo quy định của Ngân hàng Nhà nước"),
            ],
            "notes": "Quy trình chuyển đổi tài khoản",
        },
        {
            "id": "q_028",
            "question": "Khi người dùng bấm hủy liên kết thanh toán MoMo với một bên thứ ba thì quyền thanh toán qua ví và các nghĩa vụ hợp đồng riêng với đối tác đó sẽ ra sao?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Người dùng không thể tiếp tục dùng nguồn tiền Ví MoMo để thanh toán trên dịch vụ đó nữa, nhưng việc hủy liên kết không ảnh hưởng đến các cam kết riêng về việc cung cấp dịch vụ giữa người dùng và đối tác.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [
                find_snippet("momo_linking", "Sau khi hủy, Quý khách sẽ không thể tiếp tục dùng nguồn tiền là Ví MoMo để thanh toán trên trang web hoặc ứng dụng của Nhà cung cấp dịch vụ nữa"),
                find_snippet("momo_linking", "không ảnh hưởng đến các cam kết riêng của Quý khách với Nhà cung cấp dịch vụ về việc cung cấp sản phẩm, dịch vụ"),
            ],
            "notes": "Hậu quả hủy liên kết đối tác",
        },
        {
            "id": "q_029",
            "question": "MoMo được phép cung cấp hoặc chia sẻ thông tin dữ liệu của khách hàng cho các cơ quan, tổ chức bên ngoài trong những tình huống nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "MoMo chỉ chia sẻ trong phạm vi cần thiết để cung cấp dịch vụ, tuân thủ nghĩa vụ pháp luật, khi có yêu cầu hợp pháp từ cơ quan nhà nước có thẩm quyền, hoặc khi được Người dùng đồng ý.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [
                find_snippet("momo_privacy", "MoMo chỉ chia sẻ dữ liệu cá nhân trong phạm vi cần thiết để cung cấp dịch vụ, tuân thủ nghĩa vụ pháp lý"),
                find_snippet("momo_privacy", "trừ khi được Người dùng đồng ý hoặc khi việc chia sẻ được thực hiện theo quy định của pháp luật"),
            ],
            "notes": "Chia sẻ dữ liệu cá nhân hợp pháp",
        },

        # =========================================================================
        # 3. UNANSWERABLE QUESTIONS (10 items: q_030 to q_039)
        # Tính năng MoMo có thật trong thực tế nhưng KHÔNG CÓ TRONG TÀI LIỆU
        # -> Dễ khiến Prompt v1 hallucinate theo kiến thức parametric có sẵn,
        #    trong khi Prompt v2 từ chối nghiêm ngặt!
        # =========================================================================
        {
            "id": "q_030",
            "question": "Biểu phí rút tiền từ số dư ví MoMo về tài khoản ngân hàng liên kết khi vượt quá định mức miễn phí trong tháng là bao nhiêu phần trăm?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Tính năng MoMo thực tế nhưng biểu phí rút tiền chi tiết không có trong tài liệu corpus",
        },
        {
            "id": "q_031",
            "question": "Chính sách chuyển tiền giữa hai tài khoản ví MoMo với nhau hiện nay cho phép miễn phí tối đa bao nhiêu lượt chuyển mỗi tháng?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Tính năng chuyển tiền ví-ví thực tế nhưng tài liệu không chứa số lượt miễn phí",
        },
        {
            "id": "q_032",
            "question": "Tỷ suất sinh lời hay lãi suất tiền gửi của sản phẩm tích lũy Túi Thần Tài trên ứng dụng MoMo hiện tại là bao nhiêu phần trăm một năm?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Túi Thần Tài có trong mục lục FAQ nhưng tài liệu không có lãi suất cụ thể",
        },
        {
            "id": "q_033",
            "question": "Hạn mức chi tiêu ban đầu được cấp khi người dùng đăng ký mở dịch vụ Ví Trả Sau trên MoMo tối đa là bao nhiêu triệu đồng?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Tính năng Ví Trả Sau có thật nhưng hạn mức cụ thể không có trong corpus",
        },
        {
            "id": "q_034",
            "question": "Mức phí duy trì dịch vụ hoặc phí quản lý tài khoản của Ví Trả Sau MoMo là bao nhiêu tiền mỗi tháng nếu có phát sinh giao dịch?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Phí duy trì Ví Trả Sau không được quy định trong tài liệu đã crawl",
        },
        {
            "id": "q_035",
            "question": "Thời gian ân hạn miễn lãi tối đa cho các hóa đơn mua sắm thanh toán qua Ví Trả Sau của MoMo là bao nhiêu ngày kể từ ngày chốt sao kê?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Thời gian ân hạn Ví Trả Sau không có trong tài liệu",
        },
        {
            "id": "q_036",
            "question": "Để tích lũy và đổi được một Heo Vàng quyên góp trong chương trình Heo Đất MoMo thì người dùng cần bao nhiêu gam thức ăn?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Heo Đất MoMo có thật nhưng thể lệ đổi Heo Vàng không có trong tài liệu",
        },
        {
            "id": "q_037",
            "question": "Tỷ lệ quy đổi điểm thưởng MoMo Rewards sang tiền mặt hoặc mã giảm giá trên ứng dụng MoMo là bao nhiêu điểm tương đương 1 đồng?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Tính năng điểm thưởng MoMo Rewards không có chi tiết trong corpus",
        },
        {
            "id": "q_038",
            "question": "Hạn mức vay tiền mặt tiêu dùng nhanh FastMoney do đối tác tài chính liên kết trên ứng dụng MoMo cung cấp tối đa là bao nhiêu triệu đồng?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Dịch vụ vay FastMoney trên MoMo có thật nhưng thông tin hạn mức không có trong tài liệu",
        },
        {
            "id": "q_039",
            "question": "Khách hàng sử dụng thẻ tín dụng quốc tế Visa hoặc Mastercard để nạp tiền vào ví MoMo có bị trừ phí nạp không và mức phí cụ thể là bao nhiêu?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Phí nạp từ thẻ tín dụng quốc tế không có trong văn bản crawl",
        },

        # =========================================================================
        # 4. TRAP QUESTIONS (7 items: q_040 to q_046)
        # =========================================================================
        {
            "id": "q_040",
            "question": "Một học sinh 14 tuổi đã có CCCD gắn chip thì có được tự mình đăng ký và mở tài khoản ví điện tử MoMo mà không cần người giám hộ không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không được. MoMo không hướng đến người dưới 15 tuổi, và việc xử lý dữ liệu cho người dưới 15 tuổi phải được thực hiện thông qua người đại diện theo pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không hướng đến người dưới 15 tuổi. Việc xử lý dữ liệu cá nhân của người dưới 15 tuổi, người bị hạn chế hoặc mất năng lực hành vi dân sự được thực hiện thông qua người đại diện theo pháp luật")],
            "notes": "Bẫy độ tuổi: 14 tuổi có CCCD vẫn không được tự mở",
        },
        {
            "id": "q_041",
            "question": "Sau khi đã cài đặt liên kết ví MoMo với ứng dụng của bên thứ ba, có phải mỗi lần đặt đơn hàng người dùng vẫn phải mở app MoMo nhập mã OTP để xác nhận trừ tiền không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không phải. Sau khi liên kết thành công, các giao dịch sau này sẽ tự động trừ tiền từ Ví MoMo mà không cần phải xác nhận gì thêm trên Ứng dụng MoMo.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "các giao dịch thanh toán sau này Quý khách thực hiện trên trang web hoặc ứng dụng của Nhà cung cấp dịch vụ sẽ tự động trừ tiền từ Ví MoMo của Quý khách mà không cần phải xác nhận gì thêm")],
            "notes": "Bẫy xác nhận thanh toán liên kết",
        },
        {
            "id": "q_042",
            "question": "Nếu một khách hàng đã hoàn thành xác thực sinh trắc học cấp cao và nâng cấp tài khoản thì số dư trong ví MoMo có được phép vượt quá 300 triệu đồng không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không được. Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm")],
            "notes": "Bẫy số dư tối đa 200 triệu tại mọi thời điểm",
        },
        {
            "id": "q_043",
            "question": "Có phải hạn mức thanh toán và chuyển tiền qua ví MoMo của mỗi cá nhân là không giới hạn mỗi tháng nếu số dư tài khoản đủ tiền?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không phải. Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng")],
            "notes": "Bẫy hạn mức chi tiêu tháng 100 triệu",
        },
        {
            "id": "q_044",
            "question": "Sau khi đã bấm xác nhận liên kết tài khoản MoMo với nhà cung cấp dịch vụ, người dùng có bắt buộc phải duy trì liên kết này vĩnh viễn không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không, người dùng có thể hủy liên kết với dịch vụ bất kỳ lúc nào ngay trên Ứng dụng MoMo hoặc tại trang web/ứng dụng của đối tác.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hủy liên kết bất kỳ lúc nào:** Quý khách có thể hủy liên kết với dịch vụ bất kỳ lúc nào ngay trên Ứng dụng MoMo")],
            "notes": "Bẫy bắt buộc duy trì liên kết",
        },
        {
            "id": "q_045",
            "question": "Khi người dùng hủy mua vé xem phim hoặc đồ ăn trên ứng dụng của bên thứ ba đã liên kết, MoMo có nghĩa vụ bồi hoàn lại toàn bộ số tiền thanh toán không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không, M_Service không có nghĩa vụ hoàn trả bất kỳ khoản phí nào khi người dùng yêu cầu hoàn hủy việc cung cấp sản phẩm dịch vụ của bên thứ ba.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "M\\_Service không có nghĩa vụ hoàn trả bất kỳ khoản phí nào khi Quý khách yêu cầu hoàn, hủy việc cung cấp sản phẩm, dịch vụ")],
            "notes": "Bẫy hoàn tiền trung gian",
        },
        {
            "id": "q_046",
            "question": "MoMo có quyền tự ý chuyển giao dữ liệu cá nhân của khách hàng cho các nhãn hàng bên ngoài để họ chạy quảng cáo tiếp thị trực tiếp mà không cần xin phép không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không, MoMo không chia sẻ thông tin Người dùng cho bên thứ ba cho mục đích tiếp thị trực tiếp của họ, trừ khi được Người dùng đồng ý hoặc theo quy định của pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không chia sẻ thông tin Người dùng cho bên thứ ba cho mục đích tiếp thị trực tiếp của họ, trừ khi được Người dùng đồng ý")],
            "notes": "Bẫy tự ý chia sẻ dữ liệu tiếp thị",
        },

        # =========================================================================
        # 5. CASUAL QUESTIONS (4 items: q_047 to q_050)
        # =========================================================================
        {
            "id": "q_047",
            "question": "Xin chào bạn, cho mình hỏi tên gọi đầy đủ và pháp nhân chính thức của công ty vận hành ví MoMo là gì?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Công Ty Cổ Phần Dịch Vụ Di Động Trực Tuyến (viết tắt là M_Service).",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "MoMo:** là Công Ty Cổ Phần Dịch Vụ Di Động Trực Tuyến")],
            "notes": "Chào hỏi và hỏi tên công ty pháp nhân",
        },
        {
            "id": "q_048",
            "question": "Chào MoMo, nếu mình cần hỗ trợ kỹ thuật hoặc khiếu nại khẩn cấp thì số điện thoại tổng đài hotline chính thức là bao nhiêu?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Người dùng có thể liên hệ với MoMo qua Hotline 1900 5454 41.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "- Hotline: 1900 5454 41")],
            "notes": "Chào hỏi và hỏi số điện thoại hotline",
        },
        {
            "id": "q_049",
            "question": "Chào trợ lý, ngoài gọi điện lên hotline thì người dùng có thể gửi yêu cầu hỗ trợ hoặc phản ánh về quyền dữ liệu qua những kênh nào trên ứng dụng?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Người dùng có thể sử dụng Tính năng Quản lý Dữ liệu cá nhân hoặc Tính năng Trợ giúp trên ứng dụng MoMo.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Tính năng Quản lý Dữ liệu cá nhân hoặc Tính năng Trợ giúp trên ứng dụng MoMo")],
            "notes": "Chào hỏi và hỏi tính năng khiếu nại trong app",
        },
        {
            "id": "q_050",
            "question": "Hi bạn, cho mình hỏi định nghĩa ngắn gọn ứng dụng MoMo là gì theo quy định của điều khoản dịch vụ?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Ứng Dụng MoMo là ứng dụng trên nền tảng di động do MoMo phát triển và vận hành để cung cấp các dịch vụ ví điện tử và trung gian thanh toán.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Ứng Dụng MoMo:** là ứng dụng trên nền tảng di động do MoMo phát triển và vận hành để cung cấp các Sản Phẩm/Dịch Vụ cho Người Sử Dụng, bao gồm dịch vụ Ví Điện Tử")],
            "notes": "Chào hỏi và hỏi định nghĩa ứng dụng",
        },
    ]

    return questions


def main():
    questions = build_questions()
    output_path = DATA_DIR / "eval_set.jsonl"
    print(f"Saving {len(questions)} high-quality questions to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("Done!")


if __name__ == "__main__":
    main()
