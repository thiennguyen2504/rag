"""Generate a robust 50-question evaluation set for MoMo RAG and validate verbatim evidence."""

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
    momo_terms = corpus["momo_terms"]
    momo_privacy = corpus["momo_privacy"]
    momo_linking = corpus["momo_linking"]

    # Helpers to find exact verbatim snippet
    def find_snippet(doc_name, phrase):
        doc = corpus[doc_name]
        pos = doc.lower().find(phrase.lower())
        if pos == -1:
            raise ValueError(f"Phrase '{phrase[:40]}...' not found in {doc_name}")
        # return snippet around it up to ~150 chars
        snippet = doc[pos : pos + len(phrase)]
        return snippet

    questions = [
        # q_001 to q_005 (Original 5 questions)
        {
            "id": "q_001",
            "question": "Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi có được mở tài khoản MoMo không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Người sử dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự được đăng ký, mở và sử dụng tài khoản theo quy định của MoMo phù hợp pháp luật.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Trường hợp Người Sử Dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự, việc đăng ký, mở và sử dụng tài khoản được thực hiện theo điều kiện, phương thức")],
            "notes": "Căn cứ khoản 2.1 Điều 2 momo_terms",
        },
        {
            "id": "q_002",
            "question": "Hạn mức giao dịch và hạn mức ngày mặc định khi liên kết dịch vụ trên MoMo là bao nhiêu?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Hạn mức giao dịch mặc định tối đa là 50.000đ và hạn mức ngày mặc định tối đa là 250.000đ mỗi ngày cho các giao dịch từ một tài khoản đối tác đã liên kết.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán các giao dịch tối đa là 50.000đ"),
                find_snippet("momo_linking", "mặc định Quý khách chỉ được thanh toán tổng cộng tối đa là 250.000đ mỗi ngày"),
            ],
            "notes": "Căn cứ Mục III momo_linking",
        },
        {
            "id": "q_003",
            "question": "Ví MoMo có hỗ trợ tính năng mua bán tiền điện tử Bitcoin không?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "MoMo không có tính năng mua bán tiền điện tử Bitcoin.",
        },
        {
            "id": "q_004",
            "question": "Có đúng là sau khi liên kết tài khoản dịch vụ thì mỗi lần thanh toán người dùng vẫn phải mở ứng dụng MoMo để bấm xác nhận không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không đúng. Sau khi liên kết thành công, các giao dịch thanh toán sau này sẽ tự động trừ tiền từ Ví MoMo mà không cần phải xác nhận gì thêm trên Ứng dụng MoMo.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "các giao dịch thanh toán sau này Quý khách thực hiện trên trang web hoặc ứng dụng của Nhà cung cấp dịch vụ sẽ tự động trừ tiền từ Ví MoMo của Quý khách mà không cần phải xác nhận gì thêm")],
            "notes": "Căn cứ Mục Lưu ý momo_linking",
        },
        {
            "id": "q_005",
            "question": "Chào bạn, cho tôi biết MoMo là ứng dụng gì?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "MoMo là ứng dụng Ví điện tử thuộc quyền sở hữu của Công ty Cổ phần Dịch vụ Di Động Trực Tuyến (M_Service).",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "MoMo:** là ứng dụng Ví điện tử thuộc quyền sở hữu của Công ty Cổ phần Dịch vụ Di Động Trực Tuyến (M\\_Service)")],
            "notes": "Chào hỏi và hỏi định nghĩa cơ bản",
        },

        # --- DIRECT QUESTIONS (q_006 to q_025) ---
        {
            "id": "q_006",
            "question": "Số dư tối đa của Tài Khoản Ví Điện Tử MoMo tại mọi thời điểm là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo")],
            "notes": "Khoản 5.3 Điều 5 momo_terms",
        },
        {
            "id": "q_007",
            "question": "Tổng hạn mức giao dịch chuyển tiền và thanh toán qua các tài khoản Ví cá nhân của một người dùng tối đa là bao nhiêu một tháng?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng")],
            "notes": "Khoản 5.3 Điều 5 momo_terms",
        },
        {
            "id": "q_008",
            "question": "Hạn mức bổ sung tối đa cho nhóm giao dịch thanh toán đặc thù (điện, nước, viễn thông, học phí, viện phí) là bao nhiêu một tháng?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Tổng hạn mức cho nhóm giao dịch thanh toán đặc thù không vượt quá 300.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "tổng hạn mức cho nhóm này không vượt quá 300.000.000 đồng/tháng")],
            "notes": "Khoản 5.3 Điều 5 momo_terms",
        },
        {
            "id": "q_009",
            "question": "Công ty chủ quản sở hữu và vận hành ví MoMo có tên đầy đủ là gì?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Công Ty Cổ Phần Dịch Vụ Di Động Trực Tuyến (M_Service).",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "MoMo:** là Công Ty Cổ Phần Dịch Vụ Di Động Trực Tuyến")],
            "notes": "Khoản 1.1 Điều 1 momo_terms",
        },
        {
            "id": "q_010",
            "question": "Hồ sơ mở tài khoản MoMo tối thiểu bao gồm những thông tin nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hồ Sơ Mở Tài Khoản gồm tối thiểu: họ và tên; ngày, tháng, năm sinh; quốc tịch; số điện thoại; số định danh cá nhân hoặc giấy tờ tùy thân còn hiệu lực; dữ liệu sinh trắc học.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "gồm tối thiểu: họ và tên; ngày, tháng, năm sinh; quốc tịch; số điện thoại; số định danh cá nhân hoặc giấy tờ tùy thân còn hiệu lực; dữ liệu sinh trắc học")],
            "notes": "Khoản 1.15 Điều 1 momo_terms",
        },
        {
            "id": "q_011",
            "question": "Hạn mức giao dịch liên kết tối thiểu và tối đa trên MoMo có thể cài đặt là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hạn mức tối thiểu là 25.000đ và hạn mức tối đa là không giới hạn (áp dụng theo hạn mức tối đa của Ví MoMo).",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hạn mức tối thiểu là 25.000đ và hạn mức tối đa là không giới hạn")],
            "notes": "Mục III momo_linking",
        },
        {
            "id": "q_012",
            "question": "Hạn mức ngày tối thiểu khi liên kết tài khoản đối tác trên MoMo là bao nhiêu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Hạn mức tối thiểu là 150.000đ và hạn mức tối đa là không giới hạn.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hạn mức tối thiểu là 150.000đ và hạn mức tối đa là không giới hạn")],
            "notes": "Mục III momo_linking",
        },
        {
            "id": "q_013",
            "question": "Người dùng có thể hủy dịch vụ liên kết tài khoản ở đâu?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Quý khách có thể hủy dịch vụ liên kết bất kỳ lúc nào ngay trên Ứng dụng Ví điện tử MoMo hoặc hủy ở web, ứng dụng của Nhà cung cấp dịch vụ.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "hủy dịch vụ liên kết bất kỳ lúc nào ngay trên Ứng dụng Ví điện tử MoMo hoặc hủy ở web, ứng dụng của Nhà cung cấp dịch vụ")],
            "notes": "Mục IV momo_linking",
        },
        {
            "id": "q_014",
            "question": "MoMo có hướng đến người dùng dưới 15 tuổi hay không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "MoMo không hướng đến người dưới 15 tuổi.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không hướng đến người dưới 15 tuổi")],
            "notes": "Chính sách quyền riêng tư momo_privacy",
        },
        {
            "id": "q_015",
            "question": "Việc xử lý dữ liệu cá nhân của người dưới 15 tuổi trên MoMo được thực hiện qua ai?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Được thực hiện thông qua người đại diện theo pháp luật theo quy định.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "được thực hiện thông qua người đại diện theo pháp luật theo quy định")],
            "notes": "Chính sách quyền riêng tư momo_privacy",
        },
        {
            "id": "q_016",
            "question": "Chủ thể dữ liệu có quyền yêu cầu xóa dữ liệu cá nhân của mình trên MoMo không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, chủ thể dữ liệu có quyền yêu cầu xóa dữ liệu trong phạm vi pháp luật cho phép.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Quyền yêu cầu xóa dữ liệu trong phạm vi pháp luật cho phép")],
            "notes": "Mục Quyền chủ thể dữ liệu momo_privacy",
        },
        {
            "id": "q_017",
            "question": "Chủ thể dữ liệu có quyền rút lại sự đồng ý cho phép xử lý dữ liệu cá nhân trên MoMo không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, người dùng có quyền rút lại sự đồng ý theo quy định pháp luật về bảo vệ dữ liệu cá nhân.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Quyền rút lại sự đồng ý")],
            "notes": "Mục Quyền chủ thể dữ liệu momo_privacy",
        },
        {
            "id": "q_018",
            "question": "Người sử dụng nạp tiền vào Tài khoản ví điện tử MoMo thông qua những kênh nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Thông qua nhận tiền từ Tài Khoản Ngân Hàng Liên Kết, nhận tiền từ tài khoản thanh toán ngân hàng, từ Tài Khoản Ví Điện Tử khác hoặc ví điện tử khác khi pháp luật cho phép.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Nạp tiền vào Tài Khoản Ví Điện Tử** được thực hiện thông qua: (a) nhận tiền từ Tài Khoản Ngân Hàng Liên Kết của Người Sử Dụng; (b) nhận tiền từ tài khoản thanh toán bằng đồng Việt Nam")],
            "notes": "Khoản 5.1 Điều 5 momo_terms",
        },
        {
            "id": "q_019",
            "question": "Ngày làm việc theo quy định trong Điều khoản chung của MoMo là những ngày nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Là các ngày từ Thứ Hai đến Thứ Sáu, không bao gồm ngày nghỉ, lễ, Tết theo quy định pháp luật.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Ngày Làm Việc:** là các ngày từ Thứ Hai đến Thứ Sáu, không bao gồm ngày nghỉ, lễ, Tết theo quy định pháp luật")],
            "notes": "Khoản 1.16 Điều 1 momo_terms",
        },
        {
            "id": "q_020",
            "question": "Người sử dụng có bắt buộc phải duy trì liên kết với tài khoản ngân hàng trong suốt thời gian sử dụng ví MoMo không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Có, Người Sử Dụng phải hoàn thành việc liên kết và duy trì liên kết trong suốt thời gian sử dụng Tài Khoản Ví Điện Tử trừ trường hợp pháp luật quy định khác.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "duy trì liên kết trong suốt thời gian sử dụng Tài Khoản Ví Điện Tử")],
            "notes": "Điều 4 momo_terms",
        },
        {
            "id": "q_021",
            "question": "Biện pháp xác thực sinh trắc học trên MoMo đối chiếu khớp đúng thông tin với những nguồn nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Đối chiếu khớp đúng thông tin sinh trắc học với dữ liệu trong giấy tờ tùy thân, danh tính điện tử (tài khoản định danh điện tử VneID) hoặc cơ sở dữ liệu có thẩm quyền.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "đối chiếu khớp đúng thông tin sinh trắc học của Người Sử Dụng với dữ liệu trong giấy tờ tùy thân, danh tính điện tử (tài khoản định danh điện tử (VneID))")],
            "notes": "Khoản 1.12 Điều 1 momo_terms",
        },
        {
            "id": "q_022",
            "question": "Điều khoản chung mở và sử dụng tài khoản MoMo áp dụng chính thức từ ngày nào?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Áp dụng từ ngày 30/09/2026.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Áp dụng từ ngày 30/09/2026")],
            "notes": "Đầu văn bản momo_terms",
        },
        {
            "id": "q_023",
            "question": "MoMo thu thập những loại dữ liệu cá nhân cơ bản nào của khách hàng?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Bao gồm thông tin định danh cá nhân, thông tin liên lạc, thông tin tài khoản dịch vụ, thông tin sinh trắc học và thông tin tài chính.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Thông tin định danh cá nhân, thông tin liên lạc, thông tin tài khoản dịch vụ.")],
            "notes": "Mục Dữ liệu thu thập momo_privacy",
        },
        {
            "id": "q_024",
            "question": "M_Service có chịu trách nhiệm về chất lượng sản phẩm do Nhà cung cấp dịch vụ cung cấp khi liên kết không?",
            "type": "direct",
            "answerable": True,
            "gold_answer": "Không, M_Service là trung gian thanh toán và không chịu bất kỳ trách nhiệm liên quan nào về sản phẩm, dịch vụ do Nhà cung cấp dịch vụ cung cấp.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "M\\_Service là trung gian thanh toán giữa Quý khách và Nhà cung cấp dịch vụ, vì vậy M\\_Service không chịu bất kỳ trách nhiệm liên quan nào về sản phẩm, dịch vụ do Nhà cung cấp dịch vụ cung cấp")],
            "notes": "Mục II momo_linking",
        },
        {
            "id": "q_025",
            "question": "Chào bạn, cho mình hỏi MoMo hỗ trợ chăm sóc khách hàng qua số tổng đài hotline nào vậy?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "MoMo hỗ trợ chăm sóc khách hàng qua Hotline 1900 5454 41.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "- Hotline: 1900 5454 41")],
            "notes": "Chào hỏi và hỏi số điện thoại hotline hỗ trợ",
        },

        # --- MULTI-HOP QUESTIONS (q_026 to q_033) ---
        {
            "id": "q_026",
            "question": "Người từ đủ 15 đến dưới 18 tuổi cần đáp ứng điều kiện gì về năng lực hành vi và người đại diện để đăng ký sử dụng MoMo?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Người sử dụng từ đủ 15 đến dưới 18 tuổi phải không bị mất hoặc hạn chế năng lực hành vi dân sự, đồng thời tùy theo dịch vụ có thể cần thêm sự chấp thuận của người đại diện theo pháp luật.",
            "gold_doc_ids": ["momo_terms", "momo_privacy"],
            "gold_evidence": [
                find_snippet("momo_terms", "Người Sử Dụng từ đủ 15 tuổi đến chưa đủ 18 tuổi và không bị mất, hạn chế năng lực hành vi dân sự"),
                find_snippet("momo_privacy", "Đối với người từ đủ 15 đến dưới 18 tuổi, MoMo có thể yêu cầu thêm sự chấp thuận của người đại diện theo pháp luật"),
            ],
            "notes": "Kết hợp momo_terms và momo_privacy",
        },
        {
            "id": "q_027",
            "question": "Nếu giấy tờ tùy thân của khách hàng hết hiệu lực hoặc không cập nhật thông tin định danh thì MoMo có quyền áp dụng những biện pháp gì?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "MoMo có thể hạn chế, tạm ngừng, từ chối thực hiện một phần hoặc toàn bộ giao dịch, hoặc đóng tài khoản của khách hàng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Trường hợp Người Sử Dụng không cung cấp, cập nhật hoặc xác minh thông tin theo yêu cầu, hoặc giấy tờ tùy thân hết hiệu lực"),
                find_snippet("momo_terms", "MoMo có thể hạn chế, tạm ngừng, từ chối thực hiện một phần hoặc toàn bộ Giao Dịch, hoặc đóng tài khoản"),
            ],
            "notes": "Khoản 3.4 Điều 3 momo_terms",
        },
        {
            "id": "q_028",
            "question": "Người dùng có thể nạp tiền vào ví MoMo từ những nguồn nào và sau khi nạp có thể dùng số dư để chi trả cho những mục đích nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Nạp tiền từ tài khoản ngân hàng liên kết hoặc ví điện tử khác; sau đó có thể rút về tài khoản ngân hàng, chuyển tiền hoặc thanh toán sản phẩm, dịch vụ và nộp dịch vụ công.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Nạp tiền vào Tài Khoản Ví Điện Tử** được thực hiện thông qua: (a) nhận tiền từ Tài Khoản Ngân Hàng Liên Kết của Người Sử Dụng"),
                find_snippet("momo_terms", "Tài Khoản Ví Điện Tử** được sử dụng cho các mục đích: (a) rút tiền về Tài Khoản Ngân Hàng Liên Kết"),
            ],
            "notes": "Khoản 5.1 và 5.2 Điều 5 momo_terms",
        },
        {
            "id": "q_029",
            "question": "Khi hủy dịch vụ liên kết tài khoản đối tác, quan hệ thanh toán qua MoMo và các cam kết riêng với đối tác bị ảnh hưởng như thế nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Sau khi hủy, người dùng không thể tiếp tục dùng Ví MoMo để thanh toán trên dịch vụ đó, nhưng việc hủy không ảnh hưởng đến các cam kết riêng của người dùng với Nhà cung cấp dịch vụ.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [
                find_snippet("momo_linking", "Sau khi hủy, Quý khách sẽ không thể tiếp tục dùng nguồn tiền là Ví MoMo để thanh toán trên trang web hoặc ứng dụng của Nhà cung cấp dịch vụ nữa"),
                find_snippet("momo_linking", "không ảnh hưởng đến các cam kết riêng của Quý khách với Nhà cung cấp dịch vụ về việc cung cấp sản phẩm, dịch vụ"),
            ],
            "notes": "Mục Những điều lưu ý và Mục IV momo_linking",
        },
        {
            "id": "q_030",
            "question": "Trong trường hợp nào MoMo có quyền trích nợ, khấu trừ hoặc thu hồi tiền từ tài khoản người dùng?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Khi có giao dịch và phí người dùng đã xác nhận, hoặc tiền ghi có nhầm do sai sót kỹ thuật, hoặc nghĩa vụ theo yêu cầu hợp pháp của cơ quan nhà nước có thẩm quyền.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "MoMo chỉ trích nợ, khấu trừ hoặc thu hồi tiền trong trong phạm vi khoản tiền khi có căn cứ xác định Người Sử Dụng phải thanh toán hoặc hoàn trả"),
                find_snippet("momo_terms", "khoản tiền ghi có nhầm hoặc phát sinh do sai sót kỹ thuật; và (c) nghĩa vụ theo yêu cầu hợp pháp của cơ quan có thẩm quyền"),
            ],
            "notes": "Khoản 5.4 Điều 5 momo_terms",
        },
        {
            "id": "q_031",
            "question": "MoMo chia sẻ dữ liệu cá nhân của người dùng trong những trường hợp nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "MoMo chỉ chia sẻ dữ liệu cá nhân trong phạm vi cần thiết để cung cấp dịch vụ, tuân thủ nghĩa vụ pháp luật, hoặc khi được Người dùng đồng ý.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [
                find_snippet("momo_privacy", "MoMo chỉ chia sẻ dữ liệu cá nhân trong phạm vi cần thiết để cung cấp dịch vụ, tuân thủ nghĩa vụ pháp lý"),
                find_snippet("momo_privacy", "trừ khi được Người dùng đồng ý hoặc khi việc chia sẻ được thực hiện theo quy định của pháp luật"),
            ],
            "notes": "Mục Chia sẻ dữ liệu momo_privacy",
        },
        {
            "id": "q_032",
            "question": "Người dùng cần làm những bước gì để từ một tài khoản MoMo cơ bản trở thành Tài khoản Ví điện tử MoMo chính thức?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Người dùng cần hoàn tất quy trình đăng ký, định danh, xác thực thông tin (bao gồm sinh trắc học) và liên kết với tài khoản ngân hàng của chính mình.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "sau khi Người Sử Dụng hoàn tất quy trình đăng ký, định danh, xác thực và liên kết tài khoản ngân hàng"),
                find_snippet("momo_terms", "định danh và Xác Thực Sinh Trắc Học theo quy định của Ngân hàng Nhà nước"),
            ],
            "notes": "Khoản 1.5 Điều 1 và Điều 3 momo_terms",
        },
        {
            "id": "q_033",
            "question": "Trách nhiệm bồi thường tổn thất trong trường hợp xảy ra giao dịch trái phép trên MoMo được phân định như thế nào?",
            "type": "multi",
            "answerable": True,
            "gold_answer": "Được xác định trên cơ sở lỗi của các bên. Người dùng không chịu trách nhiệm nếu tổn thất do lỗi hoặc sự cố hệ thống của MoMo, và MoMo có trách nhiệm bồi hoàn.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [
                find_snippet("momo_terms", "Trách nhiệm đối với Giao Dịch trái phép được xác định trên cơ sở lỗi của các bên"),
                find_snippet("momo_terms", "Người Sử Dụng không phải chịu trách nhiệm đối với những tổn thất phát sinh do lỗi của MoMo, sự cố hệ thống của MoMo"),
            ],
            "notes": "Khoản 6.2 Điều 6 momo_terms",
        },

        # --- UNANSWERABLE QUESTIONS (q_034 to q_043) ---
        {
            "id": "q_034",
            "question": "Thủ tục xin cấp visa định cư tại Canada qua ứng dụng MoMo gồm những giấy tờ gì?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Không có thông tin về xin visa Canada",
        },
        {
            "id": "q_035",
            "question": "Ví MoMo có cung cấp dịch vụ đổi trực tiếp tiền mặt Euro sang tiền Yên Nhật tại quầy không?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Không có dịch vụ đổi ngoại tệ ngoại quốc tại quầy",
        },
        {
            "id": "q_036",
            "question": "Lãi suất tiền gửi tiết kiệm kỳ hạn 12 tháng tại Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam hiện tại là bao nhiêu?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Thông tin ngoài phạm vi tài liệu MoMo",
        },
        {
            "id": "q_037",
            "question": "Làm thế nào để kích hoạt tính năng đào tiền ảo Ethereum trên nền tảng MoMo?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "MoMo không hỗ trợ đào tiền ảo",
        },
        {
            "id": "q_038",
            "question": "Quy định về thời hạn cấp phép xây dựng nhà ở riêng lẻ tại TP.HCM theo Luật Xây dựng là bao nhiêu ngày?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Ngoài phạm vi tài liệu",
        },
        {
            "id": "q_039",
            "question": "Chính sách bảo hành và sửa chữa xe ô tô điện Tesla tại trạm sạc MoMo được quy định ở điều khoản nào?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Không có dịch vụ trạm sạc xe Tesla",
        },
        {
            "id": "q_040",
            "question": "MoMo có hỗ trợ mở tài khoản ngân hàng tại Thụy Sĩ cho công dân Việt Nam không?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Không có hỗ trợ mở tài khoản ngân hàng Thụy Sĩ",
        },
        {
            "id": "q_041",
            "question": "Hướng dẫn cách kết nối ví MoMo trực tiếp với mạng vệ tinh Starlink của SpaceX?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Không có nội dung liên quan Starlink",
        },
        {
            "id": "q_042",
            "question": "Mức xử phạt hành chính đối với hành vi điều khiển xe máy vượt đèn đỏ theo Nghị định 100 là bao nhiêu?",
            "type": "unanswerable",
            "answerable": False,
            "gold_answer": "",
            "gold_doc_ids": [],
            "gold_evidence": [],
            "notes": "Nghị định giao thông ngoài phạm vi tài liệu",
        },
        {
            "id": "q_043",
            "question": "Hi MoMo, cho mình hỏi ngoài hotline thì người dùng có thể gửi khiếu nại qua tính năng nào trên ứng dụng?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Người dùng có thể sử dụng Tính năng Quản lý Dữ liệu cá nhân hoặc Tính năng Trợ giúp trên ứng dụng MoMo.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "Tính năng Quản lý Dữ liệu cá nhân hoặc Tính năng Trợ giúp trên ứng dụng MoMo")],
            "notes": "Chào hỏi và hỏi tính năng hỗ trợ khiếu nại trên app",
        },

        # --- TRAP QUESTIONS (q_044 to q_049) ---
        {
            "id": "q_044",
            "question": "Người từ đủ 12 tuổi đến dưới 15 tuổi có được tự đứng tên đăng ký mở tài khoản ví điện tử MoMo không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không được. MoMo không hướng đến người dưới 15 tuổi, và việc xử lý dữ liệu cho đối tượng này phải thông qua người đại diện theo pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không hướng đến người dưới 15 tuổi. Việc xử lý dữ liệu cá nhân của người dưới 15 tuổi, người bị hạn chế hoặc mất năng lực hành vi dân sự được thực hiện thông qua người đại diện theo pháp luật")],
            "notes": "Bẫy độ tuổi: 12 tuổi không được tự mở ví",
        },
        {
            "id": "q_045",
            "question": "MoMo có cam kết hoàn trả 100% tiền thanh toán khi người dùng hủy mua hàng trên ứng dụng của bên thứ ba không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không. M_Service không có nghĩa vụ hoàn trả bất kỳ khoản phí nào khi người dùng yêu cầu hoàn hủy sản phẩm dịch vụ của đối tác, người dùng phải liên hệ Nhà cung cấp dịch vụ để giải quyết.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "M\\_Service không có nghĩa vụ hoàn trả bất kỳ khoản phí nào khi Quý khách yêu cầu hoàn, hủy việc cung cấp sản phẩm, dịch vụ")],
            "notes": "Bẫy hoàn tiền trung gian",
        },
        {
            "id": "q_046",
            "question": "Số dư trong tài khoản ví MoMo có được phép vượt quá 500 triệu đồng nếu khách hàng đã xác thực sinh trắc học không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không được. Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm theo chính sách quản lý rủi ro của MoMo.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Số dư tối đa của Tài Khoản Ví Điện Tử là 200.000.000 đồng tại mọi thời điểm")],
            "notes": "Bẫy số dư tối đa 200 triệu",
        },
        {
            "id": "q_047",
            "question": "Có phải hạn mức thanh toán và chuyển tiền qua ví MoMo của mỗi cá nhân là không giới hạn mỗi tháng?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không phải. Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Ví cá nhân của 1 người dùng tối đa là 100.000.000 đồng/tháng.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Tổng hạn mức giao dịch (chuyển tiền và thanh toán) qua các Tài Khoản Ví Điện Tử cá nhân của 01 Người Sử Dụng tối đa là 100.000.000 đồng/tháng")],
            "notes": "Bẫy hạn mức tháng",
        },
        {
            "id": "q_048",
            "question": "Khi đã liên kết tài khoản MoMo với đối tác, người dùng có được phép hủy liên kết hay bắt buộc phải duy trì vĩnh viễn?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Người dùng có thể hủy liên kết với dịch vụ bất kỳ lúc nào ngay trên Ứng dụng MoMo hoặc tại ứng dụng/website của đối tác, không bắt buộc duy trì vĩnh viễn.",
            "gold_doc_ids": ["momo_linking"],
            "gold_evidence": [find_snippet("momo_linking", "Hủy liên kết bất kỳ lúc nào:** Quý khách có thể hủy liên kết với dịch vụ bất kỳ lúc nào ngay trên Ứng dụng MoMo")],
            "notes": "Bẫy bắt buộc duy trì liên kết đối tác",
        },
        {
            "id": "q_049",
            "question": "MoMo có được tự ý chia sẻ thông tin cá nhân của người dùng cho bất kỳ bên thứ ba nào mà không cần sự đồng ý hay căn cứ pháp lý không?",
            "type": "trap",
            "answerable": True,
            "gold_answer": "Không. MoMo không chia sẻ thông tin Người dùng cho bên thứ ba cho mục đích tiếp thị trừ khi được Người dùng đồng ý hoặc theo quy định pháp luật.",
            "gold_doc_ids": ["momo_privacy"],
            "gold_evidence": [find_snippet("momo_privacy", "MoMo không chia sẻ thông tin Người dùng cho bên thứ ba cho mục đích tiếp thị trực tiếp của họ, trừ khi được Người dùng đồng ý")],
            "notes": "Bẫy tự ý chia sẻ dữ liệu",
        },

        # --- CASUAL QUESTIONS (q_050) ---
        {
            "id": "q_050",
            "question": "Xin chào trợ lý, bạn có thể cho tôi biết dịch vụ Ví điện tử MoMo là gì không?",
            "type": "casual",
            "answerable": True,
            "gold_answer": "Ví điện tử MoMo là tài khoản điện tử sau khi người sử dụng hoàn tất quy trình đăng ký, định danh, xác thực và liên kết ngân hàng để sử dụng các dịch vụ thanh toán do MoMo cung cấp.",
            "gold_doc_ids": ["momo_terms"],
            "gold_evidence": [find_snippet("momo_terms", "Tài Khoản Ví Điện Tử MoMo (“Ví MoMo/Tài Khoản Ví Điện Tử”):** là Tài Khoản MoMo sau khi Người Sử Dụng hoàn tất quy trình đăng ký, định danh, xác thực và liên kết tài khoản ngân hàng")],
            "notes": "Câu hỏi định nghĩa kèm lời chào",
        },
    ]

    return questions


def main():
    questions = build_questions()
    output_path = DATA_DIR / "eval_set.jsonl"
    print(f"Saving {len(questions)} questions to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("Done!")


if __name__ == "__main__":
    main()
