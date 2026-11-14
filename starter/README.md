# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `day6_retriever.py` | Adapter dense retrieval, mở Chroma persistent và embedder từ Day 6 |
| `openrouter_controller.py` | Controller dùng OpenRouter function calling |
| `run_openrouter_agent.py` | Chạy câu hỏi thật qua OpenRouter và Chroma Day 6 |
| `run_agent_benchmark.py` | Chạy agent thật theo split và phương pháp retrieval |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `evaluate_retrieval.py` | So sánh dense, TF-IDF và hybrid RRF trên benchmark |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

### Kết nối dense retrieval Day 6

`Day6ChromaRetriever` dùng lại embedder ONNX, collection Chroma và metadata của Day 6. Chạy bằng Python environment đã cài dependencies Day 6; thư mục database mặc định là `starter\data\chroma` trong repo Day 6. Đặt đường dẫn Day 6 (không phải thư mục database) và collection muốn dùng:

Index Day 6 dùng `sentence-transformers/all-MiniLM-L6-v2` qua Chroma ONNX, vector 384 chiều; cấu hình mặc định trong source Day 6 chỉ hỗ trợ model này. Collection B đang dùng cho benchmark có 3.121 chunks; Chroma Day 6 là phiên bản 1.5.9.

```powershell
$env:DAY6_STARTER_PATH = 'C:\path\to\K5-L3-Day06\starter'
$env:DAY6_STRATEGY = 'B'
```

`DAY6_STRATEGY` nhận `A`, `B` hoặc `C`; nếu không đặt thì dùng `B`. Có thể override `CHROMA_PERSIST_DIRECTORY` theo cấu hình Day 6. Từ thư mục gốc Day 7, kiểm tra truy xuất:

```powershell
python -c "from starter.day6_retriever import Day6ChromaRetriever; r = Day6ChromaRetriever(); print([(x['id'], x['score'], x['source'], x['page']) for x in r.search('câu hỏi thử', filters=None, top_k=3)])"
```

Khi chạy, `DAY6_STARTER_PATH` phải trỏ đúng repo Day 6 có `src\store.py`, và Python phải dùng các package trong `starter\requirements.txt` của Day 6. Adapter sẽ báo lỗi nếu database/collection thiếu hoặc chunk không có metadata truy nguồn; không tự tạo index rỗng hay dùng vector giả.

### Chạy agent với OpenRouter

Controller gọi OpenRouter qua OpenAI-compatible API; model mặc định là `google/gemini-4-hydrogen`. Cài SDK và giữ API key trong `.env` cục bộ, không commit hoặc gửi key qua chat. Nếu `.env` chưa tồn tại thì tạo từ mẫu; nếu đã có thì không ghi đè, chỉ thêm/cập nhật `OPENROUTER_API_KEY` và `OPENROUTER_MODEL`:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
$day6Starter = 'C:\path\to\K5-L3-Day06\starter'
$python = Join-Path $day6Starter '.venv\Scripts\python.exe'
& $python -m pip install -r starter\requirements-model.txt
```

Đặt `OPENROUTER_MODEL=google/gemini-4-hydrogen` và giữ `DAY6_STARTER_PATH` trỏ đến thư mục `starter` Day 6. Sau đó chạy từ thư mục gốc Day 7 bằng Python đã cài dependencies Day 6. Mỗi lệnh gọi OpenRouter có timeout mặc định 90 giây; có thể theo dõi tiến trình ở terminal:

```powershell
& $python -m starter.run_openrouter_agent "Câu hỏi cần trả lời"
```

Có thể đổi model trong `.env` (`OPENROUTER_MODEL`), collection (`DAY6_STRATEGY=A`, `B` hoặc `C`) và tùy chỉnh `--top-k`, `--max-searches`, `--budget`. Lệnh in câu trả lời cùng trace retrieval/tool call ra JSON. Mỗi lần gọi OpenRouter có timeout 90 giây và thông báo tiến trình. Agent chỉ kiểm tra citation ID có trong context; cần tự đối chiếu citation với nội dung nguồn. Thực thi lệnh sẽ gửi câu hỏi và các đoạn context được truy xuất đến OpenRouter và model đã chọn.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval và không thay dense retrieval.

## 3. So sánh retrieval và đánh giá dev/test

Từ thư mục gốc Day 7, dùng Python trong môi trường Day 6 để đọc collection Chroma và chạy so sánh offline:

```powershell
$day6Starter = 'C:\path\to\K5-L3-Day06\starter'
$python = Join-Path $day6Starter '.venv\Scripts\python.exe'
& $python -m starter.evaluate_retrieval --strategy B
```

Đảm bảo `.env` Day 7 có `DAY6_STARTER_PATH` trỏ đến thư mục `starter` Day 6 như hướng dẫn ở trên.
Runner so sánh dense Day 6, TF-IDF và hybrid reciprocal-rank fusion (RRF) trên cùng collection, filter câu hỏi, top-3 và context budget 800 ký tự. Hybrid hợp nhất top-50 ứng viên với RRF k=60. Chọn phương pháp chỉ bằng dev theo thứ tự Evidence Hit@3, Hit@1, Context Evidence Rate, rồi median latency thấp hơn; sau đó đánh giá dense baseline và phương pháp đã chọn trên test. Kết quả JSON được ghi vào `starter/results/retrieval_eval/`. Đây là đánh giá retrieval offline, không gọi OpenRouter và không chấm câu trả lời/citation của agent.

## 4. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 5. Đánh giá agent thật và tổng hợp

Agent batch dùng model OpenRouter trong `.env`, chạy sau khi retrieval đã được chọn trên dev. Lệnh dev dưới đây chạy 15 câu có đáp án và 2 câu thiếu evidence qua cả ba phương pháp; sau khi đọc trace và chốt cấu hình, chạy baseline dense cùng phương pháp được chọn trên test:

```powershell
& $python -m starter.run_agent_benchmark --split dev --methods dense_day6 tfidf hybrid_rrf --strategy B
& $python -m starter.run_agent_benchmark --split test --methods dense_day6 hybrid_rrf --strategy B
```

Mặc định agent dùng top-3, tối đa 2 lượt search và context budget 800 ký tự. Nếu phương pháp chọn không phải hybrid, thay `hybrid_rrf` ở lệnh test. Batch gửi câu hỏi và các đoạn context được truy xuất đến OpenRouter, có thể phát sinh phí. Mỗi phương pháp lưu `cases.json`, `traces.json`, `summary.json` và cấu hình trong `starter/results/agent_eval/<split>/<method>/`. Summary ghi token/chi phí nếu OpenRouter trả usage; nếu không có thông tin giá, chi phí được để `null`, không giả định bằng 0. Dùng `--resume` để tiếp tục run dở mà không gọi lại các câu đã xong.

Đọc và chấm câu trả lời/citation thủ công trước khi tổng hợp metric; batch runner không tự gán nhãn đúng sai.

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Khi chấm context, thêm `context_grade` 0–2 và `context_reason`. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`; thêm `--require-context-grade` khi tổng hợp test để buộc có nhãn context cho mỗi câu có đáp án. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản và hỗ trợ filter metadata. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
