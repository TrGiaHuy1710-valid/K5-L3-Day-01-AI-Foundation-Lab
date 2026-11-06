# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603924
- **Họ và tên:** Trịnh Gia Huy
- **SEED:** 6000

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: 14/15 đạt. Mục còn lại là `tính tay: f1_example`. Đáp án đúng là 8/11 ≈ **0.73**, nhưng ô tổng hợp lưu giá trị cũ 0.65 do chạy sai thứ tự ô.

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.902 | 0.919 | 0.06 | 0 |
| 2. BoW + MLP | 0.948 | 0.974 | 0.26 | 0 |
| 3a. MiniLM + kNN | 0.903 | 1.000 | 18.7 | 0 |
| 3b. MiniLM + LogReg | 0.960 | 0.974 | 7.1 | 0 |
| 3c. MiniLM + MLP head | 0.960 | 0.974 | 7.8 | 0 |
| 4a. LLM nhỏ zero-shot (gemini-3.5-flash-lite) | 0.976* | 0.976 | 803 | 0.0205 |
| 4b. LLM lớn zero-shot (gemini-3.8-flash) | 1.000* | 1.000 | 4487 | 0.0479 |
| 4c. LLM nhỏ few-shot | 0.952* | 0.952 | 707 | 0.0481 |

\*LLM chỉ chạy trên tập SUB (40 tin) để tiết kiệm chi phí, nên cột F1 test của LLM chính là F1 SUB.

## 3. Dự đoán và thực tế

1. **`llm_fewshot`.** Mình đoán few-shot làm F1 tăng và chi phí tăng khoảng 1.5 lần. Thực tế F1 **giảm** từ 0.976 xuống 0.952 và chi phí tăng khoảng 2.3 lần ($0.0205 → $0.0481). Lệch vì 4 ví dụ mẫu lấy theo thứ tự đầu tập train, không chọn lọc. Trong đó có một tin nhãn spam mà nội dung giống ham ("tattoos"), có thể làm model nhiễu. Prompt cũng dài gấp nhiều lần nên tốn token input hơn mình tính.
2. **`hard_eval`.** Mình đoán BoW yếu ở nhóm `ham_looks_spam` và `transactional_ham`, còn LLM yếu ở prompt injection. Thực tế BoW tụt mạnh (F1 còn 0.31–0.38) còn LLM nhỏ đạt **1.00** trên cả 7 nhóm, kể cả prompt injection. Mình đã đánh giá thấp khả năng đọc ngữ cảnh của LLM. Bộ 50 case cũng quá nhỏ để kết luận LLM miễn nhiễm với injection.
3. **`llm_models`.** Mình đoán model lite kém hơn model lớn 5–15% F1 và chi phí chênh khoảng 2 lần. Hướng đoán đúng nhưng độ chênh nhỏ hơn nhiều: 0.976 so với 1.000 (khoảng 2.4 điểm), trên tập chỉ 40 tin. Chi phí thực tế chênh khoảng 2.3 lần. Chênh lệch chất lượng nhỏ hơn dự đoán, còn chênh lệch độ trễ lớn hơn hẳn (p95 803 ms so với 4487 ms).

## 4. Phân tích 50 case khó

- **Tụt nhiều nhất:** BoW + MLP, từ F1 0.974 (SUB) xuống 0.378 trên 50 case khó, mất khoảng 0.60. BoW + LogReg cũng tụt mạnh (0.919 → 0.333). Embedding tụt vừa (kNN còn 0.70) và LLM nhỏ giữ 1.00. Tập test thường cùng phân phối với tập train, spam giống mẫu đã học nên điểm cao chưa chứng minh model hiểu ngữ nghĩa.
- **Case 1 (nhóm "tưởng spam nhưng không phải", #1):** "Congrats!!! You WON the bet lol, I owe you a beer. Call me later" (nhãn ham). BoW chỉ đếm token rời rạc, mà "Congrats", "WON", "Call" là từ khoá spam có trọng số cao, nên model đoán spam. Nó không nhìn được ngữ cảnh "bạn bè cá cược".
- **Case 2 (nhóm né từ khoá, #31):** "U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461" (nhãn spam). Chèn dấu chấm làm mỗi chữ thành một token lạ chưa có trong từ điển, nên BoW không thấy từ khoá nào. LLM vẫn đọc được như người nên đoán đúng.
- **Lưu ý:** hai case trên là phân tích theo cơ chế của từng model. Mình chưa in danh sách lỗi từng case của BoW. Ô soi lỗi trong notebook chỉ in model cuối (4d, sai 0/50).

## 5. Thử thách cá nhân

- **Đề theo SEED 6000:** #1 — đổi ngưỡng quyết định 0.5 → 0.2 và 0.8 cho mô hình Phần 1, báo precision/recall/F1 từng ngưỡng, và chọn ngưỡng cho ngân hàng sợ chặn nhầm OTP.
- **Thực tế mình đã làm:** so sánh MLP `hidden=4` với `hidden=512` (15 epoch). Đây **không** phải đề #1.

| Hidden | Tham số | Thời gian train | F1 test |
| --- | --- | --- | --- |
| 4 | 8,009 | 1.59s | 0.947 |
| 512 | 1,025,025 | 3.54s | 0.948 |

Mạng nhỏ có F1 gần như bằng mạng lớn dù ít hơn khoảng 128 lần tham số và train nhanh hơn. Với bài toán BoW đơn giản này, dung lượng thêm không giúp ích, nên mạng nhỏ là lựa chọn hợp lý khi cần tốc độ hoặc bộ nhớ thấp. **Cần làm lại đúng đề #1 (đổi ngưỡng) trước khi nộp.**

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin, 10 triệu SMS/ngày (khoảng 116 tin/giây).

- **Chọn:** Encoder MiniLM + LogReg (3b). F1 test 0.960, p95 7.1 ms, khoảng 3,600 tin/s, chi phí API bằng 0. Phương án này đạt cả độ trễ lẫn chất lượng.
- **Loại LLM làm đường chính:** p95 803 ms (nhỏ) và 4487 ms (lớn), đều vượt xa 50 ms. Chi phí theo bảng trên, nhân với 10 triệu tin/ngày:
  - LLM nhỏ: khoảng $205/ngày, tức khoảng $6,100/tháng.
  - LLM lớn: khoảng $479/ngày, tức khoảng $14,400/tháng.
- **Đánh đổi:** 3b chỉ đạt F1 0.55–0.60 trên 50 case khó, tệ hơn nhiều so với LLM (1.00). Mình chấp nhận điều này ở đường chính để giữ tốc độ và chi phí.
- **Thiết kế lai (router):** dùng 3b cho các tin có p ngoài khoảng [0.3, 0.7], chỉ chuyển tin lưỡng lự sang LLM (bất đồng bộ hoặc xử lý sau). Cách này vẫn đạt p95 < 50 ms cho đa số tin mà bớt được lỗi ở các case khó.

## 7. AI log và điều đã học

| Phần | Hỏi AI | AI trả | Mình sửa / kiểm chứng |
| --- | --- | --- | --- |
| 4.1–4.4 | Viết `call_openai`, prompt, `parse_label`, `llm_classify_batch` chạy được với Gemini key trong `.env` | Code 4 hàm, đặt `LLM_PROVIDER="gemini"` | Chạy thử gọi API thật |
| 4.1 | Lỗi 400 "Thinking level MINIMAL is not supported" với model lớn | Thử từng mức `reasoning_effort`; thêm `model_params` theo model (lớn dùng `none`, lite dùng `minimal`) | Thử lại trên cả 2 model, đều chạy |
| Phần 5 | Viết ngắn gọn báo cáo này từ notebook và `lab1_report.json` | Bản nháp các mục 1–7 | Đối chiếu số liệu với `lab1_report.json`; ghi rõ chỗ chưa khớp (F1 tính tay, đề thử thách) |

**Điều đã học:**
- Điểm cao trên tập test cùng phân phối với tập train chưa chứng minh model hiểu bài toán: bộ 50 case khó kéo BoW từ 0.97 xuống khoảng 0.38.
- Few-shot không tự động tốt hơn zero-shot và lại tốn token hơn nhiều. Chọn ví dụ mẫu mới là yếu tố quyết định.
- Cùng một API nhưng mỗi model nhận tham số khác nhau (`reasoning_effort`), nên phải thử từng model thay vì giả định.
