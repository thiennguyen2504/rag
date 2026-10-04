# RAG MoMo: Hệ Thống Q&A Tiếng Việt Cho Dịch Vụ Ví Điện Tử MoMo

Dự án xây dựng pipeline RAG (Retrieval-Augmented Generation) tiếng Việt phục vụ giải đáp điều khoản, hướng dẫn và chính sách của ví điện tử MoMo, kèm hệ thống thực nghiệm so sánh hai cấu hình (Strategy A vs Strategy B).

---

## 1. Cấu Trúc Thư Mục Dự Án

```
rag-momo/
├── configs/
│   ├── models.yaml             # Cấu hình model Gemini, embedding và đơn giá token
│   ├── A.yaml                  # Cấu hình pipeline A (chunk phẳng 500 ký tự)
│   └── B.yaml                  # Cấu hình pipeline B (chunk phân cấp theo cấu trúc)
├── data/
│   ├── sources.yaml            # Danh sách nguồn tài liệu MoMo (người dùng quản lý)
│   ├── phase1_queries.txt      # 10 câu hỏi kiểm tra retrieval Phase 1 (6 có đáp án, 4 ngoài phạm vi)
│   ├── raw/                    # File HTML thô tải từ web (.gitignore)
│   ├── manual/                 # Bản sao chép thủ công (ưu tiên cao nhất nếu có)
│   ├── processed/              # Markdown sạch sau khi cào và chuẩn hóa
│   ├── corpus_manifest.json    # Snapshot metadata (URL, thời gian tải, sha256, số ký tự/từ)
│   ├── chunks_A.jsonl          # Kết quả chunking Chiến lược A
│   └── chunks_B.jsonl          # Kết quả chunking Chiến lược B
├── chroma_db/                  # Vector database Chroma persist (.gitignore)
│   └── ingest_manifest.json    # Manifest ghi nhận embedding_model và số chunk
├── prompts/
│   ├── v1_simple.txt           # Prompt cơ bản cho cấu hình A
│   ├── v2_strict.txt           # Prompt nghiêm ngặt, chống bịa đặt cho cấu hình B
│   └── judge.txt               # Prompt LLM-as-a-Judge cho Phase đánh giá
├── src/
│   ├── __init__.py
│   ├── config.py               # Quản lý cấu hình Pydantic, resolve model
│   ├── llm_utils.py            # Exponential backoff + jitter retry, compute_cost, usage
│   ├── list_models.py          # Liệt kê models Gemini qua REST API
│   ├── check_api.py            # Kiểm tra kết nối generator, judge và embedding model
│   ├── fetch_docs.py           # Thu thập web, trích xuất Trafilatura/BS4, làm sạch text
│   ├── inspect_corpus.py       # Thống kê corpus, kiểm tra chất lượng và boilerplate
│   ├── chunking.py             # Triển khai thuật toán chunking A và B
│   ├── ingest.py               # Ingest idempotent vào Chroma với batching & retry
│   ├── check_retrieval.py      # Đánh giá truy xuất top-k, tính độ lệch phân tách
│   ├── retriever.py            # Module retriever có cache Chroma và lọc threshold
│   ├── context.py              # Đóng gói và cắt tỉa ngữ cảnh (max_context_chars, dedupe)
│   ├── pipeline.py             # Pipeline LCEL hỏi đáp chuẩn hợp đồng dữ liệu + CLI
│   └── api.py                  # FastAPI server cung cấp REST API hỏi đáp
├── eval/
│   └── __init__.py
├── analysis/
│   └── phase1_retrieval_check.md # Báo cáo đánh giá retrieval Phase 1
├── tests/
│   ├── test_cleaning.py        # Kiểm thử chuẩn hóa NFC, loại boilerplate
│   ├── test_chunking.py        # Kiểm thử tính deterministic, prefix context, gộp FAQ
│   ├── test_ingest.py          # Kiểm thử tính idempotent và manifest của Chroma
│   ├── test_fetch.py           # Kiểm thử ưu tiên manual, BS4 fallback, discovery link
│   ├── test_context.py         # Kiểm thử đánh số, dedupe và giới hạn ký tự
│   ├── test_llm_utils.py       # Kiểm thử compute_cost và trích xuất usage
│   ├── test_pipeline.py        # Kiểm thử pipeline.ask với mock retriever & mock LLM
│   └── test_api.py             # Kiểm thử FastAPI endpoints qua TestClient
├── conftest.py
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

---

## 2. Hướng Dẫn Sử Dụng Phase 2 (Pipeline & FastAPI)

### 2.1. Chạy Hỏi Đáp Bằng Giao Diện Dòng Lệnh (CLI)

Chạy pipeline với Cấu hình A (`v1_simple.txt`, chunk 500 ký tự không phân cấp):
```bash
python -m src.pipeline --config A --q "Điều kiện đăng ký tài khoản MoMo là gì?"
```

Chạy pipeline với Cấu hình B (`v2_strict.txt`, chunk phân cấp có breadcrumb, trích dẫn `[1]`, `[2]`):
```bash
python -m src.pipeline --config B --q "Hạn mức giao dịch liên kết mặc định là bao nhiêu?"
```

**Thông tin hiển thị:**
- Câu hỏi & Câu trả lời được sinh từ LLM.
- Danh sách nguồn tham khảo (Score cosine similarity, Chunk ID, Tên tài liệu, Trích đoạn).
- Độ trễ từng giai đoạn: Retrieval, Generation, Total (ms).
- Số lượng token tiêu thụ: Input, Output, Total.
- Ước tính chi phí USD dựa trên bảng giá thực tế từ `configs/models.yaml`.

---

### 2.2. Khởi Chạy REST API Server (FastAPI)

Khởi chạy server uvicorn:
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Tài liệu Swagger UI tương tác có tại: `http://127.0.0.1:8000/docs`.

#### Các Endpoint Khả Dụng:

1. **`GET /health`**
   - Kiểm tra trạng thái hoạt động của server.
   - Phản hồi: `{"status": "ok"}`

2. **`GET /configs`**
   - Liệt kê danh sách các cấu hình pipeline có sẵn trong thư mục `configs/`.
   - Phản hồi: `{"configs": ["A", "B"]}`

3. **`POST /ask`**
   - Gửi câu hỏi đến pipeline RAG.
   - **Request Body**:
     ```json
     {
       "question": "Làm thế nào để liên kết tài khoản ngân hàng với MoMo?",
       "config_id": "B"
     }
     ```
   - **Response Contract**:
     ```json
     {
       "question": "Làm thế nào để liên kết tài khoản ngân hàng với MoMo?",
       "config_id": "B",
       "answer": "...",
       "sources": [
         {
           "index": 1,
           "chunk_id": "momo_linking-B-001",
           "doc_id": "momo_linking",
           "title": "Điều khoản dịch vụ liên kết tài khoản",
           "source_url": "https://momo.vn/dieu-khoan-dich-vu-lien-ket",
           "score": 0.8524,
           "text": "..."
         }
       ],
       "final_prompt": "...",
       "usage": {
         "input_tokens": 1250,
         "output_tokens": 120,
         "total_tokens": 1370
       },
       "cost_usd": 0.000129,
       "latency_ms": {
         "retrieval": 420.5,
         "generation": 2100.2,
         "total": 2520.7
       },
       "error": null
     }
     ```
   - Trả về mã lỗi `400 Bad Request` nếu `config_id` không tồn tại.

---

#### 2.3. Chạy Toàn Bộ Kiểm Thử (Unit Tests)

Chạy tất cả 41 bài kiểm thử của cả Phase 1, Phase 2 và Phase 3 (**không cần kết nối mạng**, **không cần API key**):
```bash
pytest -v
```

Các test cases bao gồm:
- **`tests/test_validate_eval_set.py`**: Kiểm thử toàn diện validator Phase 3 (evidence hợp lệ/bất hợp lệ kèm gợi ý RapidFuzz, ràng buộc unanswerable, phát hiện trùng lặp ID, thiếu doc_id, và cảnh báo chất lượng).
- **`tests/test_context.py`**: Kiểm thử đánh số 1-indexed, lọc trùng text lặp, giới hạn ký tự `max_context_chars` không cắt giữa chunk.
- **`tests/test_llm_utils.py`**: Kiểm thử tính toán chi phí token `compute_cost` và trích xuất `usage_metadata`.
- **`tests/test_pipeline.py`**: Kiểm thử luồng hoạt động của `ask()`, bắt lỗi exception không làm crash hệ thống, và cơ chế `short_circuit_on_empty_context`.
- **`tests/test_api.py`**: Kiểm thử các endpoint FastAPI `/health`, `/configs`, `/ask` và xử lý mã lỗi 400.
- **Phase 1 Tests**: Kiểm thử làm sạch NFC/boilerplate (`test_cleaning.py`), chunking A/B (`test_chunking.py`), nạp vector idempotent (`test_ingest.py`), thu thập tài liệu (`test_fetch.py`).

---

## 3. Kiến Trúc & Cấu Hình Pipeline (A vs B)

### Bảng So Sánh Cấu Hình

| Thuộc tính | Cấu hình A (`configs/A.yaml`) | Cấu hình B (`configs/B.yaml`) |
|---|---|---|
| **Mô tả** | Chunk phẳng 500 ký tự (naive), prompt đơn giản | Chunk cấu trúc có ngữ cảnh breadcrumb, prompt nghiêm ngặt |
| **Chroma Collection** | `momo_A` (202 chunks) | `momo_B` (109 chunks) |
| **Top-k** | 3 chunks | 5 chunks |
| **Similarity Threshold** | `null` | `null` (khuyến nghị 0.74 cho Phase đánh giá) |
| **Prompt Template** | `prompts/v1_simple.txt` | `prompts/v2_strict.txt` (bắt buộc trích dẫn `[1]`, `[2]`, từ chối khi thiếu tin) |
| **Temperature** | 0.1 | 0.1 |
| **Short Circuit** | `false` | `false` |
| **Cơ chế Cache** | Cache Chroma client & pipeline theo config | Cache Chroma client & pipeline theo config |

---

## 4. Hợp Đồng Dữ Liệu Kết Quả Pipeline

Mọi phản hồi từ `pipeline.ask(question, config_id)` và endpoint `POST /ask` đều tuân thủ hợp đồng:
```json
{
  "question": str,
  "config_id": str,
  "answer": str,
  "sources": [
    {
      "index": int,
      "chunk_id": str,
      "doc_id": str,
      "title": str,
      "source_url": str,
      "score": float,
      "text": str
    }
  ],
  "final_prompt": str,
  "usage": {
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int
  },
  "cost_usd": float,
  "latency_ms": {
    "retrieval": float,
    "generation": float,
    "total": float
  },
  "error": str | null
}
```

---

## 5. Hướng Dẫn Sử Dụng Phase 3 (Bộ Công Cụ Hỗ Trợ Xây Eval Set)

Phase 3 cung cấp công cụ tự động hóa hỗ trợ chuyên gia xây dựng tập đánh giá chuẩn (`eval_set.jsonl`) cho hệ thống RAG, đảm bảo tính chặt chẽ và không thể bịa đặt.

### 5.1. Cấu Trúc Tập Đánh Giá Chuẩn (`data/eval_set.jsonl`)
Mỗi câu hỏi là 1 dòng JSON tuân thủ schema:
- `id` (str): ID duy nhất (vd: `"q_001"`).
- `question` (str): Câu hỏi đánh giá.
- `type` (str): Một trong 5 loại:
  - `direct` (Mục tiêu: 12 câu): Trả lời trực tiếp từ 1 vị trí trong 1 tài liệu.
  - `multi` (Mục tiêu: 6 câu): Cần tổng hợp thông tin từ >= 2 tài liệu/phần khác nhau.
  - `unanswerable` (Mục tiêu: 8 câu): Thông tin không có trong tài liệu; hệ thống phải từ chối lịch sự.
  - `trap` (Mục tiêu: 6 câu): Chứa tiền đề sai/bẫy; hệ thống phải đính chính.
  - `casual` (Mục tiêu: 4 câu): Chào hỏi, cảm ơn; hệ thống phản hồi tự nhiên.
- `answerable` (bool): `false` với `unanswerable`, `true` với các loại còn lại.
- `gold_answer` (str): Câu trả lời chuẩn của chuyên gia.
- `gold_doc_ids` (List[str]): Danh sách `doc_id` chứa thông tin trả lời (phải là `[]` nếu `answerable=false`).
- `gold_evidence` (List[str]): Đoạn trích nguyên văn từ `data/processed/<doc>.md` (phải là `[]` nếu `answerable=false`).
- `notes` (str): Ghi chú bổ sung.

Template mẫu 5 loại câu hỏi chuẩn: [`eval/eval_set_template.jsonl`](file:///d:/VCCorp_Intern/rag-momo/eval/eval_set_template.jsonl).

---

### 5.2. Công Cụ Sinh Nháp Câu Hỏi (`eval/draft_questions.py`)

Lấy mẫu ngẫu nhiên từ `data/chunks_B.jsonl` và dùng `judge_model` (Gemini 2.5 Flash) để sinh câu hỏi nháp kèm câu trả lời và đoạn trích dẫn nguyên văn:
```bash
python -m eval.draft_questions --n 10 --out data/eval_draft.jsonl
```

Tham số tùy chọn:
- `--n` (mặc định: 10): Số lượng câu hỏi muốn sinh nháp.
- `--seed` (mặc định: 42): Seed ngẫu nhiên để tái lặp kết quả lấy mẫu.
- `--concurrency` (mặc định: 3): Số luồng đồng thời gọi API Gemini (kèm cơ chế retry backoff tự động).
- `--out` (mặc định: `data/eval_draft.jsonl`): File lưu kết quả nháp.

---

### 5.3. Công Cụ Kiểm Tra Tập Đánh Giá (`eval/validate_eval_set.py`)

Kiểm tra tính hợp lệ toàn diện của file JSONL trước khi đưa vào benchmark:
```bash
python -m eval.validate_eval_set --file data/eval_set.jsonl
```

Hoặc kiểm tra file template mẫu:
```bash
python -m eval.validate_eval_set --file eval/eval_set_template.jsonl
```

**Các cơ chế kiểm tra tự động:**
1. **Kiểm tra Schema & Ràng buộc logic**:
   - `id` duy nhất, không trùng lặp.
   - `type` thuộc 5 loại hợp lệ.
   - Với câu hỏi `answerable: false`, bắt buộc `gold_doc_ids == []` và `gold_evidence == []`.
   - Với câu hỏi `answerable: true`, bắt buộc `gold_doc_ids` không rỗng và `gold_evidence` không rỗng.
2. **Kiểm tra Tính xác thực của Evidence (Verbatim Match)**:
   - Đối chiếu từng câu trong `gold_evidence` với văn bản gốc `data/processed/<doc_id>.md`.
   - Nếu không khớp nguyên văn, sử dụng thuật toán **RapidFuzz** (sliding window) để gợi ý đoạn văn bản thực tế trong tài liệu kèm độ tương đồng (%).
3. **Cảnh báo chất lượng**:
   - Cảnh báo các câu hỏi gần trùng lặp (trùng lặp ngữ nghĩa > 85%).
   - Cảnh báo trích đoạn `gold_evidence` quá dài (> 200 ký tự).
4. **Báo cáo thống kê**:
   - In bảng phân bổ 5 loại câu hỏi so sánh với mục tiêu (12/6/8/6/4).
   - Thống kê phân bổ câu hỏi theo từng tài liệu `doc_id`.

---

## 6. Hướng Dẫn Sử Dụng Phase 4 (Chạy Đánh Giá, LLM-as-a-Judge & Tổng Hợp Metrics)

Hệ thống Phase 4 gồm 3 giai đoạn độc lập có thể chạy riêng biệt: **Generate** (`run_eval`) $\rightarrow$ **Judge** (`judge`) $\rightarrow$ **Metrics** (`metrics`).

### 6.1. Quy Trình Chạy Thực Nghiệm Đầy Đủ

#### Bước 1: Chạy Pipeline Hỏi Đáp & Ghi Log (`eval/run_eval.py`)
Chạy pipeline trên toàn bộ tập câu hỏi hoặc giới hạn `--limit`, tự động ghi log vào `logs/runs.jsonl`:
```bash
# Chạy đánh giá cấu hình A với 5 câu hỏi đầu tiên
python -m eval.run_eval --config A --limit 5 --run-id eval_A_test

# Hỗ trợ cờ --resume để bỏ qua câu hỏi đã chạy nếu bị gián đoạn:
python -m eval.run_eval --config A --resume --run-id eval_A_test
```

#### Bước 2: Chấm Điểm Bằng LLM (`eval/judge.py`)
Đọc log chạy từ `logs/runs.jsonl`, join với đáp án chuẩn trong `eval_set.jsonl`, gửi prompt chấm điểm tới `judge_model` (`gemini-2.5-flash-lite` hoặc `gemini-2.5-flash`):
```bash
python -m eval.judge --run-id eval_A_test --concurrency 2
```
*Kết quả chấm điểm được lưu tại `results/<run_id>/judgements.jsonl` (hỗ trợ tự động resume bỏ qua các câu đã chấm).*

#### Bước 3: Tính Toán Metrics & Xuất Báo Cáo (`eval/metrics.py`)
Tính toán toàn bộ chỉ số về Retrieval, Generation, Hallucination, Độ trễ, Chi phí và Trích dẫn nguồn:
```bash
python -m eval.metrics --run-id eval_A_test
```
*Kết quả xuất ra 2 định dạng:*
- `results/<run_id>/summary.json`: JSON đầy đủ cấu trúc máy đọc.
- `results/<run_id>/summary.md`: Báo cáo bảng biểu markdown trực quan (tổng thể và theo từng type).

---

### 6.2. Bộ Chỉ Số Đánh Giá (Metrics)

1. **Retrieval Metrics** (Chỉ tính trên câu hỏi `answerable`):
   - `Doc Hit@k`: Tỉ lệ câu hỏi có ít nhất 1 chunk thuộc tài liệu chuẩn (`gold_doc_ids`).
   - `Evidence Hit@k`: Tỉ lệ câu hỏi có ít nhất 1 đoạn chứng cứ (`gold_evidence`) nằm trong các chunk truy xuất (so khớp chuỗi con hoặc RapidFuzz partial ratio $\ge 90\%$).
   - `Evidence Recall`: Tỉ lệ chứng cứ chuẩn được tìm thấy trên tổng số chứng cứ yêu cầu.
   - `MRR` (Mean Reciprocal Rank): Thứ hạng nghịch đảo trung bình của chunk đầu tiên chứa chứng cứ.

2. **Generation Metrics**:
   - Tỉ lệ `Correct` (Đúng hoàn toàn), `Partial` (Đúng một phần), `Incorrect` (Sai/bịa đặt).
   - `Faithful Rate`: Tỉ lệ câu trả lời có mọi khẳng định đều được hỗ trợ bởi các đoạn văn bản trong ngữ cảnh.

3. **Hallucination & Refusal Metrics**:
   - `Unanswerable Hallucination Rate`: Tỉ lệ câu unanswerable mà hệ thống trả lời thay vì từ chối (`refused=false`).
   - `Answerable Refusal Rate` (Từ chối nhầm): Tỉ lệ câu hỏi có đáp án nhưng hệ thống từ chối trả lời.
   - `Unfaithful Rate`: Tỉ lệ câu trả lời đưa vào thông tin ngoài lề không có trong context.

4. **Operational & Citations**:
   - Độ trễ (Latency p50, p95) chi tiết cho Retrieval, Generation và Total.
   - Số lượng token tiêu thụ trung bình và chi phí USD trên từng câu hỏi / toàn bộ đợt chạy.
   - Tỉ lệ trích dẫn nguồn `[n]` và tỉ lệ trích dẫn hợp lệ ($1 \le n \le k$).

---

## 7. Hướng Dẫn Sử Dụng Phase 5 (So Sánh Hai Run, Phân Tích Lỗi & Dựng Khung Báo Cáo)

Phase 5 cung cấp bộ công cụ phân tích sâu, so sánh đối đầu giữa các cấu hình (A vs B hoặc B vs C), trích xuất ca lỗi, kiểm tra độ tin cậy của judge với người chấm tay, và tự động tạo khung báo cáo tổng hợp.

### 7.1. So Sánh Hai Đợt Chạy (`eval/compare.py`)
So sánh bảng chỉ số giữa Run A và Run B kèm chênh lệch $\Delta = B - A$, tự động xác định bên chiến thắng theo đúng chiều tối ưu (cao hơn là tốt hơn cho accuracy, thấp hơn là tốt hơn cho latency/chi phí/ảo giác) và liệt kê danh sách câu hỏi phân hóa (divergence):
```bash
python -m eval.compare --a run_live_test_A --b run_live_test_B --out results/compare_A_vs_B.md
```

### 7.2. Phân Tích và Phân Loại Lỗi (`eval/failures.py`)
Tự động gom nhóm các ca lỗi vào một trong 6 nguyên nhân chính (`retrieval_miss`, `generation_error`, `hallucination_unanswerable`, `unfaithful`, `false_refusal`, `other`) và trích xuất tối đa 5 ca điển hình kèm ngữ cảnh và phần ghi nhận phân tích tay:
```bash
python -m eval.failures --run-id run_live_test_A --out analysis/failures_run_live_test_A.md
python -m eval.failures --run-id run_live_test_B --out analysis/failures_run_live_test_B.md
```

### 7.3. Đánh Giá Đối Chiếu Người vs Máy (`eval/manual_grading.py`)
1. **Lấy mẫu phân tầng độc lập (Stratified Sampling)**:
   Xuất file CSV chứa câu hỏi, câu trả lời chuẩn và câu trả lời của mô hình (không chứa nhãn của LLM-judge để tránh thiên vị):
   ```bash
   python -m eval.manual_grading sample --run-id run_live_test_B --n 15 --out results/run_live_test_B/manual_sheet.csv
   ```
2. **Đo lường độ đồng thuận & Cohen's Kappa**:
   Sau khi chuyên viên hoàn tất chấm vào CSV, tính toán tỉ lệ đồng thuận (%) và hệ số Cohen's Kappa:
   ```bash
   python -m eval.manual_grading agree --run-id run_live_test_B --sheet results/run_live_test_B/manual_sheet.csv
   ```

### 7.4. Sinh Khung Báo Cáo Thí Nghiệm Toàn Diện (`eval/report_skeleton.py`)
Tự động đọc cấu hình, dữ liệu vector, kết quả so sánh `compare`, và xuất tài liệu tổng kết `REPORT.md`:
```bash
python -m eval.report_skeleton --a run_live_test_A --b run_live_test_B --out REPORT.md
```

---

## 8. Quy Trình Thực Nghiệm Khép Kín Từ Đầu Đến Cuối (E2E Workflow)

```bash
# 1. Thu thập & chuẩn hóa tài liệu
python -m src.fetch_docs
python -m src.inspect_corpus

# 2. Sinh chunk cho cả 2 chiến lược A và B
python -m src.chunking

# 3. Nạp vector vào Chroma DB (Idempotent)
python -m src.ingest

# 4. Kiểm tra chất lượng tập dữ liệu đánh giá
python -m eval.validate_eval_set --file data/eval_set.jsonl

# 5. Chạy pipeline sinh câu trả lời cho cấu hình A và B
python -m eval.run_eval --config A --run-id run_A_full
python -m eval.run_eval --config B --run-id run_B_full

# 6. LLM-as-a-Judge chấm điểm hai đợt chạy
python -m eval.judge --run-id run_A_full --concurrency 2
python -m eval.judge --run-id run_B_full --concurrency 2

# 7. Tổng hợp metrics & sinh báo cáo chi tiết từng run
python -m eval.metrics --run-id run_A_full
python -m eval.metrics --run-id run_B_full

# 8. So sánh trực diện A vs B
python -m eval.compare --a run_A_full --b run_B_full --out results/compare_A_vs_B.md

# 9. Phân tích nhóm lỗi của từng cấu hình
python -m eval.failures --run-id run_A_full --out analysis/failures_run_A.md
python -m eval.failures --run-id run_B_full --out analysis/failures_run_B.md

# 10. Lấy mẫu chấm tay & kiểm tra độ tin cậy của Judge
python -m eval.manual_grading sample --run-id run_B_full --n 15
python -m eval.manual_grading agree --run-id run_B_full --sheet results/run_B_full/manual_sheet.csv

# 11. Xuất khung báo cáo tổng kết hoàn chỉnh
python -m eval.report_skeleton --a run_A_full --b run_B_full --out REPORT.md
```
