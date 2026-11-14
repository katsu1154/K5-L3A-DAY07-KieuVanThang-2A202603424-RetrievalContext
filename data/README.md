# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- [benchmark.json](benchmark.json) là benchmark Day 7 gồm 15 câu có đáp án và 2 câu thiếu bằng chứng trên dev, 5 câu có đáp án và 2 câu thiếu bằng chứng trên test. Mười lăm câu dev có đáp án được chuyển từ benchmark Day 6; năm câu test có đáp án được tạo mới từ các đoạn chưa dùng trong benchmark Day 6. `gold.page` dùng số trang logic giống metadata Chroma. Đã kiểm tra cả 20 evidence quote khớp chunk đúng nguồn/trang trong collection B Day 6. Bốn câu thiếu evidence có lý do vắng mặt được ghi riêng.
- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
