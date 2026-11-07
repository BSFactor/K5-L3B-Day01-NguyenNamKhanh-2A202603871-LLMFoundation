# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603871
- **Họ và tên:**NGUYỄN NAM KHÁNH
- **SEED** (in ở ô đầu notebook):8255

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑|
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.772      0.871        0.857   
2. BoW + MLP (deep learning)         0.988   0.913      0.949        0.919   
3a. Encoder (MiniLM) + kNN           0.876   0.924      0.899        0.919   
3b. Encoder (MiniLM) + LogReg        0.977   0.935      0.956        0.950   
3c. Encoder (MiniLM) + MLP head      0.966   0.935      0.950        0.950   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.004      43854.045       0.053   
2. BoW + MLP (deep learning)         4.913      41433.842       0.189   
3a. Encoder (MiniLM) + kNN          78.335         30.604      45.278   
3b. Encoder (MiniLM) + LogReg       78.855         34.645      28.902   
3c. Encoder (MiniLM) + MLP head     79.675         35.089      28.515   
4a. LLM nhỏ zero-shot                0.000          1.278     735.209   
4b. LLM lớn zero-shot                0.000          1.037     921.660   
4c. LLM nhỏ few-shot                 0.000          1.170     815.831   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.123         0.000  
2. BoW + MLP (deep learning)          0.267         0.000  
3a. Encoder (MiniLM) + kNN           65.049         0.000  
3b. Encoder (MiniLM) + LogReg        42.905         0.000  
3c. Encoder (MiniLM) + MLP head      38.745         0.000  
4a. LLM nhỏ zero-shot              1068.912         0.008  
4b. LLM lớn zero-shot              1223.497         0.076  
4c. LLM nhỏ few-shot               1327.957         0.018  
## 3. Dự đoán và thực tế

1-Learning rate

- Dự đoán:Với lr=0.001 đường cong sẽ converge chậm hơn, lr=50 sẽ khiến đường cong trở thành zig zag
- Thực tế : Kết quả với lr=0.001 theo hướng dự đoán. Đối với lr=50, tuy loss có nhảy vọt lên trên 2.5 ở những epoch đầu, model có thể tìm được điểm coverge rất nhanh,khiến đường cong trở nên bằng phẳng hơn

2-Model lớn vs Model nhỏ

- Dự đoán: Model Terra lớn hơn sẽ đạt kết quả F1 cao hơn nhưng do bài toán nhỏ nên có thể không quá lớn, có thể chỉ vài %. Model terra đắt gấp 10 lần luna. LLM có thể thắng mô hình 3b
- Thực tế: Cả model nhỏ và model lớn đều có F1 = 0.000, invalid = 40/40

3-Few-shot

- Dự đoán: F1 sẽ tăng do AI đã có được 1 chút hiểu biết về phân loại tin nhắn. Chi phí sẽ tăng khoảng 4 lần
- Thực tế: Few-shot vẫn có F1 = 0.000, invalid = 40/40. Chi phí được in trong notebook là $0.00070, so với $0.00030 của model nhỏ zero-shot.

Kết quả Model lớn vs Model nhỏ và Few-shot hoàn toàn không khớp dự đoán. Rất có khả năng là do mô hình ChatGPT đã sử dụng token vào việc
suy luận thay vì đưa ra ngay được câu trả lời chính xác.

## 4. Phân tích 50 case khó

Các mô hình đều giảm F1 khi chuyển sang tập khó. LogReg giảm mạnh nhất do chỉ tìm nghĩa trên mặt chữ. Kết quả trên tập test có thể lạc quan do dữ liệu có nhiều mẫu dễ nhận biết và tương đồng hơn với lúc training.

Nhóm khó nhất với BoW là natural spam do BoW có thể bỏ sót spam không chứa những từ khóa quen thuộc

Embedding gặp khó với nhóm talks_about_spam vì bản thân tin nhắn nói về spam nhưng lại hợp lệ.

LLM hiện đang chưa thể hiện độ chính xác kì vọng, có thể kết quả đang bị ảnh hưởng bởi giới hạn max_tokens. Chưa thể kết luận ngay.

Với nhóm né từ khóa, có thể chuẩn hóa văn bản như chuyển FR33 thành FREE để giúp BoW nhận diện lại các từ khóa spam

Hai case cụ thể :
- Case #13 — Spam giọng tự nhiên, nhãn thật `spam`: “Hope you're well! Quick reminder that your photos from the event are ready, view & order prints…” Bẫy: Giả làm nhắc nhở thân thiện, link bán hàng không yêu cầu.
- Case #16 — Spam giọng tự nhiên, nhãn thật `spam`: “Long time no see! I've started a little side business with skincare, you'd love it, can I send…” Bẫy: Giả làm bạn cũ, bán hàng đa cấp.

## 5. Thử thách cá nhân
🎯 Đề #6: LLM · Sửa build_classification_prompt để LLM trả JSON {"label": ..., "reason": ...}, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?

Với classification prompt:
Em chỉ thay đổi prompt đầu vào để đặc biệt yêu cầu llm đưa đầu ra json
def build_classification_prompt_json(text: str, examples: list[tuple[str, str]] | None = None) -> str:
    prompt = (
      "Classify the following SMS message as either 'spam' or 'ham'.\n"
      "Return ONLY valid JSON.\n"
      '{"label": "spam or ham", "reason": "brief explanation"}. \n'
    )

    if examples:
      for example_text, label in examples:
          prompt += f'Message: """{example_text}"""\nOutput: {{"label": "{label}", "reason": "Example classification"}}\n\n'

    prompt += f'Message: """{text}"""\nOutput:'

    return prompt

Với parser:
Em sử dụng regex để đọc phần label của response

def json_parse(response: str) -> int | None:
  label = re.search(r'"label"\s*:\s*"([^"]+)"', response)
  if label:
    label = label.group(1)
    if label == "spam":
      return 1
    elif label == "ham":
      return 0
  return None

Để lấy 10 tin SUB và giá cả model:
sub_10 = texts_te[SUB][:10]
input_price, output_price = PRICE_PER_1M[OPENAI_MINI_MODEL]

Để tính giá đối với prompt 1 từ:
Em cho chạy qua từng tin, lấy response, split ra và đếm số từ trong response và trong cả prompt input.
Từ đó sử dụng công thức để tính cost
word_input_tokens = 0
word_output_tokens = 0

for text in sub_10:
    prompt = build_classification_prompt(text)
    response, latency = call_llm(prompt, model=OPENAI_MINI_MODEL, max_tokens=128)
    word_input_tokens += len(prompt.split())
    word_output_tokens += len(response.split())

word_cost = (word_input_tokens * input_price + word_output_tokens * output_price) / 1000000

Tương tự với json:
json_input_tokens = 0
json_output_tokens = 0

for text in sub_10:
    prompt = build_classification_prompt_json(text)
    response, latency = call_llm(prompt, model=OPENAI_MINI_MODEL, max_tokens=128)
    json_input_tokens += len(prompt.split())
    json_output_tokens += len(response.split())

json_cost = (json_input_tokens * input_price + json_output_tokens * output_price) / 1000000


## 6. Phương án cho sản phẩm thật

Các cách LogReg, MLP, Encoder LogReg và Encoder MLP đều đạt yêu cầu. Ta có thể chọn MLP vì F1 test đạt 0.949, p95 khoảng 0.27 ms và tốc độ hơn 41.000 tin/giây, đủ xử lý 116 tin/giây. Encoder + LogReg có F1 cao hơn là 0.956 nhưng chỉ xử lý khoảng 35 tin/giây, cần chạy song song nhiều worker để đáp ứng lưu lượng. Với LLM nhỏ zero-shot, dựa trên bảng chi phí khoảng $0.008/1.000 tin, xử lý 10 triệu tin/ngày trong 30 ngày sẽ tốn khoảng $2.400/tháng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 1 | Bag-of-words là gì? | Giải thích cách biểu diễn văn bản thành vector dựa trên từ vựng. | Đối chiếu với hàm `vectorize()`. |
| 1 | Vì sao từ lặp nhiều lần vẫn chỉ được tính là 1? | Phân biệt BoW nhị phân và BoW đếm tần suất xuất hiện. | Kiểm tra với ví dụ `"Free FREE win!"`. |
| 2 | Vì sao code báo lỗi liên quan đến `dtype`? | Giải thích kiểu dữ liệu của NumPy và cách kiểm tra `dtype`, `shape`. | Cần đối chiếu lại với lỗi thực tế. |
| 2 | Vì sao xuất hiện lỗi đối tượng không thể duyệt (`not iterable`)? | Giải thích sự khác nhau giữa scalar và iterable. | Cần xác nhận traceback và cách sửa thực tế. |
| 3 | kNN hoạt động thế nào? | Giải thích cách tìm k hàng xóm gần nhất và bỏ phiếu phân loại. | Đối chiếu với hàm `knn_predict()`. |
| 3 | Vì sao kNN không cần train? | Giải thích kNN lưu dữ liệu huấn luyện và tính khoảng cách khi dự đoán. | Kiểm tra cách sử dụng dữ liệu train trong `knn_predict()`. |
| 3 | Vì sao cần cosine similarity thay vì tích vô hướng? | Giải thích cosine chuẩn hóa độ dài vector để so sánh hướng. | Đối chiếu với `top_k_similar()` và ví dụ `[10, 1]`. |
| 4 | Hoàn thiện hàm `call_openai()` theo template. | Viết hàm gọi API, truyền tham số và đo latency. | Kiểm tra `max_completion_tokens` và `reasoning_effort`. |
| 4 | Vì sao mock test trả về chuỗi rỗng? | Giải thích vị trí import `OpenAI` ảnh hưởng đến `patch()`. | Chuyển import vào trong hàm và chạy lại mock test. |
| 4 | Viết prompt phân loại zero-shot/few-shot. | Viết `build_classification_prompt()` hỗ trợ ví dụ mẫu. | Kiểm tra prompt được tạo ra với và không có examples. |
| 4 | Hoàn thiện hàm `llm_classify_batch()`. | Viết hàm phân loại hàng loạt, retry, đo latency và ước tính chi phí. | Chạy mock test và kiểm tra kết quả trả về. |
| 4 | Vì sao LLM đạt F1 = 0? | Giải thích `max_tokens=3` có thể khiến mô hình trả về chuỗi rỗng. | Đối chiếu với cấu hình token và kết quả thực nghiệm. |
| 5 | Cách đọc đường Pareto | Giải thích đường Pareto, điểm bị lấn át và quan hệ giữa F1 với độ trễ. | Đối chiếu F1 SUB, p95 và danh sách mô hình trên đường Pareto. |
| 5 | Tổng hợp tóm tắt hội thoại với AI để tạo log | Tổng hợp và tạo log | Đã kiểm tra lại bảng AI log

Em học được rằng điểm thấp có thể do cách gọi API và xử lý đầu ra, chưa chắc do khả năng của model. Khi chọn model cho sản phẩm thật, cần xem cả chất lượng, tốc độ và chi phí thay vì chỉ nhìn F1.