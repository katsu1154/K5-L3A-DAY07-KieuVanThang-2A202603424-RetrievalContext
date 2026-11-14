# Báo cáo Day 7 — Kiều Văn Thắng

## Dữ liệu và benchmark

- **Nguồn:** ebook Project Gutenberg #5000, mỗi chunk giữ source URL, page và metadata truy nguyên. Trước khi tái phân phối nội dung, kiểm tra điều kiện sử dụng áp dụng cho khu vực của người nộp.
- **Vector store:** collection `day6_B` của Day 6, 3.121 chunks; Chroma 1.5.9.
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` qua Chroma ONNX, vector 384 chiều; dùng lại embedder và persistent store Day 6 qua `Day6ChromaRetriever`.
- **Benchmark:** 24 câu: dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence. Evidence của 20 câu có đáp án được kiểm tra với chunk đúng source/page trong collection.
- **Agent:** `GPT-4o-mini` qua OpenRouter function calling. Gold/evidence không được truyền vào agent.
- **Tiêu chí:** Grounded answer và context chấm 0–2 theo `BENCHMARK.md`; citation được đối chiếu trực tiếp với nội dung, không chỉ kiểm tra ID; câu thiếu evidence cần abstain.

## Retrieval và thử nghiệm

Dense dùng top-3 từ collection Day 6; TF-IDF là baseline lexical; Hybrid RRF hợp nhất top-50 kết quả Dense và TF-IDF với `k=60`. Cùng benchmark filters và context budget 800 ký tự. Chọn phương pháp bằng dev theo Evidence Hit@3, Hit@1, Context Evidence Rate, sau đó median search latency. Hybrid được chọn trước khi chạy agent/test; test chỉ gồm Dense baseline và Hybrid.

| Cấu hình | Phương pháp / split | Giả thuyết hoặc thay đổi | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---:|---:|---:|---|
| Baseline | Dense Day 6, dev | Dùng nguyên collection và embedding Day 6 | 7/15 (46,7%) | 12/15 (80,0%) | 8/15 (53,3%) | Median 328,7 ms/query; API $0 |
| Cải thiện 1 | TF-IDF, dev | Khớp từ khóa bổ trợ dense | 6/15 (40,0%) | 9/15 (60,0%) | 6/15 (40,0%) | Median 11,2 ms/query; index 0,277 s; API $0 |
| Cải thiện 2 / chọn | Hybrid RRF, dev | Hợp nhất thứ hạng dense và từ khóa | 9/15 (60,0%) | 13/15 (86,7%) | 9/15 (60,0%) | Median 346,5 ms/query; API $0 |
| Đã chọn | Hybrid RRF, test | Đánh giá cấu hình đã chốt trên test | 4/5 (80,0%) | 4/5 (80,0%) | 4/5 (80,0%) | Median 313,8 ms/query; API retrieval $0 |
| Baseline | Dense Day 6, test | So sánh trên cùng test | 2/5 (40,0%) | 4/5 (80,0%) | 3/5 (60,0%) | Median 342,5 ms/query; API retrieval $0 |

Hai thử nghiệm đã thực hiện là TF-IDF lexical và Hybrid RRF. TF-IDF có latency thấp nhưng Hit@3 thấp hơn Dense. Hybrid tăng Evidence Hit@3 thêm 6,7 điểm phần trăm và Hit@1 thêm 13,3 điểm phần trăm trên dev; được chọn do chất lượng evidence, không chỉ do chi phí. Test nhỏ nên không đủ để kết luận tổng quát.

**Lỗi retrieval trên dev:** Hybrid bỏ evidence khỏi top-3 ở `q02` (khuyến nghị dùng màu sáng) và `q14` (động cơ khiến tác giả điều tra trong thư). Ở `q05`, `q06`, `q07`, `q15`, evidence có trong top-3 nhưng không được đóng gói vào context 800 ký tự. Các lỗi này gợi ý cần cải thiện query/ranking và chọn context; tăng top-k đơn thuần không khắc phục được context bị bỏ.

## Agent và đánh giá

- Cấu hình: top-3, tối đa 2 search/câu, context budget 800 ký tự; Day 6 strategy B.
- Tool use là số câu có gọi search / tổng câu. Grounded answer và context chỉ tính trên 15 câu dev hoặc 5 câu test có đáp án. Citation hợp lệ tính trên từng citation; coverage là số câu trả lời có citation / số câu đã trả lời.
- Đã chấm trace và evidence thủ công; vui lòng kiểm tra lại các nhãn chủ quan trước khi nộp.

| Cấu hình / split | Tool use | Answer 0/1/2 (mean /2) | Context 0/1/2 (mean /2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | API calls / tokens / chi phí / thời gian |
|---|---:|---|---|---:|---:|---:|---:|---|
| Dense / dev | 17/17 | 7/1/7 (1,0) | 3/4/8 (1,33) | 9/14 | 13/13 | 2/2 | 2/15 | 38 / 16.223 / $0,003207 / 61,1 s |
| TF-IDF / dev | 17/17 | 6/0/9 (1,2) | 2/3/10 (1,53) | 10/15 | 14/14 | 2/2 | 1/15 | 37 / 15.546 / $0,0031104 / 46,1 s |
| Hybrid / dev | 17/17 | 6/0/9 (1,2) | 2/4/9 (1,47) | 10/16 | 14/14 | 2/2 | 1/15 | 37 / 15.233 / $0,0030756 / 61,3 s |
| Dense / test | 7/7 | 3/0/2 (0,8) | 1/2/2 (1,2) | 2/7 | 6/6 | 1/2 | 0/5 | 16 / 6.809 / $0,0013395 / 26,4 s |
| Hybrid / test | 7/7 | 1/0/4 (1,6) | 0/1/4 (1,8) | 4/4 | 4/4 | 2/2 | 1/5 | 19 / chưa đầy đủ / tối thiểu $0,0013344 / 29,4 s |

### Đọc từng câu test

ID citation rút gọn trong bảng; ID đầy đủ, search results, context và response có trong `starter/results/agent_eval/test/<method>/traces.json`. Grade ghi theo answerable test; các câu `q23` và `q24` được chấm theo abstention.

| Query | Dense Day 6 | Hybrid RRF |
|---|---|---|
| `q16` — picture became popular | 1 search; nói copies of every description; grade 2, citation đúng (`3eb3e473`) | 1 search; cùng đáp án đúng; grade 2, citation đúng (`3eb3e473`) |
| `q17` — subject of Section XI | 1 search; trả lời kiến trúc dome/nave thay vì Sculpture; grade 0, citation sai (`7de08bb8`) | 1 search; trả lời notes on Sculpture; grade 2, citation đúng (`8cd21e9c`) |
| `q18` — material above buttresses | 1 search; iron anchors/chains; grade 2, citation đúng (`5db6d25b`) | 1 search; iron anchors/chains; grade 2, citation đúng (`5db6d25b`) |
| `q19` — what shells feed on | 1 search; nói chung chung “food”, không nêu animalcules; grade 0, citation sai (`9aa6235c`) | 1 search; animalcules that swim in the water; grade 2, citation đúng (`e9410225`) |
| `q20` — animal for flying-machine imitation | 2 searches; trả lời birds thay vì bat; grade 0, citation sai (`210d283b`) | 2 searches sau retry; abstain nhầm, context chỉ liên quan flight; grade 0 |
| `q23` — exact sketch date (không có evidence) | 2 searches; abstain đúng | 2 searches; abstain đúng |
| `q24` — sketch width (không có evidence) | 1 search; khẳng định 19 cm không có căn cứ, citation không hỗ trợ; abstain sai | 2 searches; abstain đúng |

Hybrid test `q20` có một attempt lỗi trước retry: model trả hai function calls nhưng controller yêu cầu một, tạo `RuntimeError: Expected one OpenRouter function call; received 2`. Lỗi được giữ trong `previous_attempts`; retry chạy đủ hai search nhưng agent vẫn abstain vì chạm giới hạn. Summary tính 19 API calls, nhưng không còn usage của hai API calls trong attempt lỗi; vì vậy Hybrid test tokens/tổng chi phí là `null`. `$0,0013344` là chi phí đã ghi nhận, không phải tổng đầy đủ.

Chi phí đã biết: dev 112 API calls, 47.002 tokens, `$0,009393`; test 35 API calls, tối thiểu `$0,0026739` đã ghi nhận. Tổng đã biết tối thiểu `$0,0120669`, chưa gồm chi phí không xác định của attempt lỗi Hybrid `q20`. Tổng thời gian của từng method nằm trong `summary.json`; thời gian retrieval ở bảng trên là median search/query.

## Tái lập và giới hạn

Từ thư mục gốc Day 7, dùng Python của Day 6 và `.env` cục bộ đã cấu hình `DAY6_STARTER_PATH`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`:

```powershell
& $python -m starter.evaluate_retrieval --strategy B
& $python -m starter.run_agent_benchmark --split dev --methods dense_day6 tfidf hybrid_rrf --strategy B
& $python -m starter.run_agent_benchmark --split test --methods dense_day6 hybrid_rrf --strategy B
```

Tổng hợp nhãn đã chấm:

```powershell
& $python starter\agent_metrics.py --cases starter\results\agent_eval\test\dense_day6\cases.json --traces starter\results\agent_eval\test\dense_day6\traces.json --reviews starter\results\agent_eval\test\dense_day6\reviews.json --split test --output starter\results\agent_eval\test\dense_day6\metrics.json --require-context-grade
& $python starter\agent_metrics.py --cases starter\results\agent_eval\test\hybrid_rrf\cases.json --traces starter\results\agent_eval\test\hybrid_rrf\traces.json --reviews starter\results\agent_eval\test\hybrid_rrf\reviews.json --split test --output starter\results\agent_eval\test\hybrid_rrf\metrics.json --require-context-grade
```

Benchmark/results có thể bị Git ignore; chỉ chia sẻ trace/nhãn khi được phép.
