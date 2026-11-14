# Kết quả so sánh retrieval Day 7

## Thiết lập

- Collection: Day 6 `day6_B`, 3.121 chunks.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` qua Chroma ONNX, 384 chiều; Chroma 1.5.9.
- So sánh trên dev: dense Day 6, TF-IDF, và hybrid RRF.
- Cùng áp dụng filter của benchmark; mỗi phương pháp trả top-3 và dùng context budget 800 ký tự.
- Hybrid hợp nhất top-50 kết quả dense và TF-IDF với RRF `k=60`.
- Chọn phương pháp bằng dev theo thứ tự Evidence Hit@3, Hit@1, Context Evidence Rate, rồi median latency thấp hơn. Chỉ sau khi chọn xong mới chạy dense baseline và phương pháp được chọn trên test.
- Median latency là thời gian search mỗi câu, không gồm khởi tạo; TF-IDF dựng index trong 0,277 giây. Dense dùng index Day 6 đã có sẵn.
- Đánh giá retrieval offline, không gọi OpenRouter; chi phí API là $0.

## Dev

| Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median truy vấn (ms) |
|---|---:|---:|---:|---:|
| Dense Day 6 | 7/15 (46,7%) | 12/15 (80,0%) | 8/15 (53,3%) | 328,7 |
| TF-IDF | 6/15 (40,0%) | 9/15 (60,0%) | 6/15 (40,0%) | 11,2 |
| Hybrid RRF | 9/15 (60,0%) | 13/15 (86,7%) | 9/15 (60,0%) | 346,5 |

## Hai cấu hình thử nghiệm

1. **TF-IDF lexical** — giả thuyết: khớp từ khóa và cách viết tên riêng sẽ bổ trợ dense. Kết quả Evidence Hit@3 là 9/15 (60,0%), thấp hơn dense 12/15 (80,0%); giả thuyết không được ủng hộ trên dev, dù latency truy vấn thấp nhất.
2. **Hybrid RRF** — giả thuyết: hợp nhất thứ hạng dense và TF-IDF sẽ tận dụng cả tương đồng ngữ nghĩa lẫn từ khóa. Kết quả Evidence Hit@3 là 13/15 (86,7%), cao hơn dense 12/15; Hit@1 cũng tăng từ 7/15 lên 9/15. Đây là cấu hình được chọn trên dev.

**Chọn hybrid RRF trên dev.** So với dense, hybrid tăng Evidence Hit@3 thêm 6,7 điểm phần trăm và Hit@1 thêm 13,3 điểm phần trăm; đổi lại latency truy vấn trung vị cao hơn khoảng 18 ms.

## Test

| Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median truy vấn (ms) |
|---|---:|---:|---:|---:|
| Dense Day 6 (baseline) | 2/5 (40,0%) | 4/5 (80,0%) | 3/5 (60,0%) | 342,5 |
| Hybrid RRF (đã chọn từ dev) | 4/5 (80,0%) | 4/5 (80,0%) | 4/5 (80,0%) | 313,8 |

Trên 5 câu có đáp án của test, hai phương pháp hòa về Evidence Hit@3; hybrid xếp evidence đúng ở vị trí đầu nhiều hơn và đưa evidence vào context nhiều hơn. Cỡ mẫu nhỏ nên đây là kết quả mô tả, chưa đủ để kết luận tổng quát. Năm câu có đáp án test được tạo mới từ các đoạn không dùng trong benchmark Day 6 cũ.

## Xem xét lỗi

- Hybrid bỏ lỡ evidence trong top-3 ở `q02` (lý do khuyên dùng màu sáng cho hình) và `q14` (động cơ điều tra trong lá thư). Đây là hai câu nên dùng để cải thiện truy vấn hoặc kiểm tra chiến lược truy xuất trên dev.
- Với `q05`, `q06`, `q07` và `q15`, hybrid có evidence trong top-3 nhưng evidence không nằm trong context đã đóng gói với ngân sách 800 ký tự. Vì vậy Context Evidence Rate thấp hơn Evidence Hit@3; cần xem xét thứ tự/chọn lọc context riêng, không chỉ tăng top-k.
- Hai câu không có evidence trong mỗi split được tính riêng, không đưa vào các metric hit. Các metric dưới đây chỉ đánh giá retrieval; đánh giá agent thật và nhãn câu trả lời/citation được báo riêng ở phần tiếp theo.

## Kết quả chi tiết

Runner lưu `dev_summary.json`, `dev_traces.json`, `test_summary.json` và `test_traces.json` trong `starter/results/retrieval_eval/` trên máy chạy. Thư mục này bị Git ignore mặc định; báo cáo này ghi lại metric và cách tái tạo, còn trace chứa từng câu để kiểm tra lỗi.

## Đánh giá agent thật

### Thiết lập và cách chấm

- Model: `GPT-4o-mini` qua OpenRouter; vector store Day 6 `day6_B`; dense embedding ONNX 384 chiều như phần trên.
- Mỗi câu dùng top-3, tối đa 2 lượt search, context budget 800 ký tự. Hybrid RRF hợp nhất top-50 dense và TF-IDF với `k=60`.
- Chạy cả ba phương pháp trên dev; sau khi chọn Hybrid RRF chỉ chạy Dense baseline và Hybrid RRF trên test. Mỗi split có 15/5 câu có đáp án và 2 câu thiếu evidence.
- Đã đọc trace, đối chiếu câu trả lời/citation với evidence và chấm answer/context theo rubric trong `BENCHMARK.md`. Đây là nhãn đánh giá thủ công; cần người nộp kiểm tra lại các nhận định chủ quan.
- Chi phí là số OpenRouter báo cáo. Không xem `null` là 0; lỗi thử lại có thể làm thiếu usage/chi phí.

### Kết quả dev

| Phương pháp | Tool use | Answer 0/1/2 (mean /2) | Context 0/1/2 (mean /2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | API calls / tokens / chi phí / thời gian |
|---|---:|---|---|---:|---:|---:|---:|---|
| Dense Day 6 | 17/17 | 7/1/7 (1,0) | 3/4/8 (1,33) | 9/14 | 13/13 | 2/2 | 2/15 | 38 / 16.223 / $0,003207 / 61,1 s |
| TF-IDF | 17/17 | 6/0/9 (1,2) | 2/3/10 (1,53) | 10/15 | 14/14 | 2/2 | 1/15 | 37 / 15.546 / $0,0031104 / 46,1 s |
| Hybrid RRF | 17/17 | 6/0/9 (1,2) | 2/4/9 (1,47) | 10/16 | 14/14 | 2/2 | 1/15 | 37 / 15.233 / $0,0030756 / 61,3 s |

Hybrid được chốt trước khi chạy test: đạt Hit@3/Hit@1 retrieval tốt nhất trên dev, hòa TF-IDF về answer mean và có chi phí API dev thấp nhất trong ba phương pháp. Chi phí API không phải tiêu chí duy nhất.

### Kết quả test

| Phương pháp | Tool use | Answer 0/1/2 (mean /2) | Context 0/1/2 (mean /2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | API calls / tokens / chi phí / thời gian |
|---|---:|---|---|---:|---:|---:|---:|---|
| Dense Day 6 (baseline) | 7/7 | 3/0/2 (0,8) | 1/2/2 (1,2) | 2/7 | 6/6 | 1/2 | 0/5 | 16 / 6.809 / $0,0013395 / 26,4 s |
| Hybrid RRF (đã chọn từ dev) | 7/7 | 1/0/4 (1,6) | 0/1/4 (1,8) | 4/4 | 4/4 | 2/2 | 1/5 | 19 / chưa đầy đủ / tối thiểu $0,0013344 / 29,4 s |

Mẫu test nhỏ (`N=5` câu có đáp án), nên kết quả chỉ mang tính mô tả. Hybrid tăng answer mean từ 0,8 lên 1,6 và context mean từ 1,2 lên 1,8; citation hợp lệ tăng từ 2/7 lên 4/4. Đổi lại Hybrid từ chối nhầm câu `q20` (1/5), còn Dense trả lời sai `q17`, `q19`, `q20` và khẳng định không có căn cứ ở `q24`.

### Xem xét trace và lỗi test

| Query | Dense Day 6 | Hybrid RRF |
|---|---|---|
| `q16` | Answer 2; citation hỗ trợ; 1 search | Answer 2; citation hỗ trợ; 1 search |
| `q17` | Answer 0: nói về kiến trúc thay vì điêu khắc; citation không hỗ trợ; 1 search | Answer 2: xác định Section XI nói về Sculpture; citation hỗ trợ; 1 search |
| `q18` | Answer 2: iron anchors/chains; citation hỗ trợ; 1 search | Answer 2: iron anchors/chains; citation hỗ trợ; 1 search |
| `q19` | Answer 0: chỉ nói “food”, không nêu animalcules; citation không hỗ trợ; 1 search | Answer 2: animalcules that swim in the water; citation hỗ trợ; 1 search |
| `q20` | Answer 0: nói birds thay vì bat; citation không hỗ trợ; 2 searches | Abstain sai trên câu có đáp án; context chỉ liên quan đến flight; 2 searches |
| `q23` | Abstain đúng; 2 searches | Abstain đúng; 2 searches |
| `q24` | Trả lời 19 cm không có bằng chứng; hai citation không hỗ trợ | Abstain đúng; 2 searches |

Ở lần thử đầu của Hybrid `q20`, OpenRouter trả về hai function calls trong khi controller yêu cầu đúng một, nên lượt chạy ghi nhận `RuntimeError: Expected one OpenRouter function call; received 2`. Retry hoàn thành nhưng agent dùng hết hai lượt search rồi abstain. Summary có 19 API calls và tính một failed attempt; usage/chi phí của hai API calls trong attempt lỗi không được giữ lại. Vì vậy tổng tokens và tổng chi phí Hybrid test là không đầy đủ; `$0,0013344` chỉ là phần chi phí được ghi nhận, không phải tổng cuối cùng.

### Chi phí và giới hạn

- Dev: 112 API calls, 47.002 tokens, tổng chi phí OpenRouter được báo cáo là `$0,009393`.
- Test: 35 API calls tính cả attempt lỗi của Hybrid; chi phí được ghi nhận tối thiểu `$0,0026739`, cộng một khoản không xác định của attempt lỗi.
- Tổng chi phí đã biết tối thiểu là `$0,0120669`; không thể báo tổng đầy đủ do thiếu usage của lỗi Hybrid `q20`.
- Traces, cấu hình, nhãn và metrics chi tiết nằm dưới `starter/results/agent_eval/{dev,test}/`; thư mục kết quả bị Git ignore mặc định. Chỉ đưa dữ liệu được phép chia sẻ vào repo nộp.

