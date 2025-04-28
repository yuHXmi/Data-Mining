
# Hệ Thống Giao Dịch Định Lượng Sử Dụng Transformer

## Nhóm thực hiện
Trần Xuân Bảo, Hà Xuân Huy, Phan Hoàng Dũng

## Mô tả dự án 
Dự án này là thành quả của nhóm chúng em trong môn học Khai Phá và Phân Tích Dữ Liệu tại Trường Đại học Công nghệ, Đại học Quốc Gia Hà Nội. Chúng em xin gửi lời cảm ơn chân thành đến các thầy cô đã tận tình hướng dẫn, cung cấp những phản hồi quý giá và luôn đồng hành trong suốt quá trình thực hiện.

Mục tiêu của dự án là xây dựng một hệ thống giao dịch định lượng sử dụng mô hình Transformer để phân loại các tín hiệu giao dịch thành ba trạng thái: BUY, SELL, và HOLD. Bằng cách tích hợp các kỹ thuật tiên tiến như Time2Vec để mã hóa thời gian và các chỉ báo kỹ thuật, chúng em hướng đến việc nâng cao khả năng nắm bắt các mối quan hệ thời gian và động lực thị trường từ dữ liệu giá cổ phiếu tần suất cao. Dự án không chỉ thể hiện tiềm năng của học sâu trong thị trường tài chính mà còn là minh chứng cho nỗ lực áp dụng kiến thức lý thuyết vào thực tiễn của nhóm.

## Cấu trúc file 
- data/ # Dữ liệu lịch sử (100 hàng mẫu của dữ liệu đã xử lý)
- models/ # Checkpoints mô hình đã huấn luyện
- notebooks/ # Notebook demo, phân tích
- requirements.txt # Các thư viện cần thiết
- README.md # Mô tả dự án

## Cách xây dựng mô hình và kết quả
### Hiểu và chuẩn bị dữ liệu
Dữ liệu sử dụng là giá cổ phiếu lịch sử theo khung thời gian 1 phút (M1) từ năm 2011 đến 2020 cho huấn luyện, cùng với tập kiểm tra (1 năm) và tập thử nghiệm (3 năm). Mỗi bản ghi bao gồm các trường: Date, Time, Open, High, Low, Close, Volume, và Label (BUY, SELL, HOLD) được gán dựa trên biến động giá trong 30 phút với ngưỡng 0.0505%.
#### Các thách thức của dữ liệu:
- Dữ liệu không liên tục: Thiếu một số bản ghi theo phút.
- Thiếu dữ liệu: Lỗ hổng lớn trong khối lượng dữ liệu.
- Dữ liệu không đồng đều: Một số phiên giao dịch chỉ có vài bản ghi.
Để xử lý, nhóm đã thử nghiệm nhiều chiến lược tiền xử lý:
1. Dữ liệu thô: Sử dụng trực tiếp dữ liệu với mã hóa nhãn (SELL=0, HOLD=1, BUY=2). Kết quả kém do chuỗi không đồng nhất.
2. Lọc chuỗi: Chỉ giữ các chuỗi liên tục có độ dài tối thiểu 128, dẫn đến mất dữ liệu và giảm khả năng khái quát.
3. Nội suy tuyến tính: Điền giá trị thiếu, nhưng chuẩn hóa toàn cục gây rò rỉ dữ liệu.
#### Phương pháp cuối cùng:
- Mã hóa nhãn: Chuyển nhãn thành số nguyên (SELL=0, HOLD=1, BUY=2).
- Trích xuất đặc trưng thời gian: Tạo các đặc trưng chuẩn hóa (hour, day_of_week, index) từ timestamp.
- Tạo chuỗi: Sinh chuỗi cố định (seq_len=64) bằng generator và chuyển thành tf.data.Dataset để huấn luyện hiệu quả.
- Chuẩn hóa theo khối: Chuẩn hóa dữ liệu theo khối 64 bước thời gian để bảo toàn đặc trưng cục bộ và tăng tính ổn định.

### Kiến trúc mô hình
Nhóm đã phát triển ba mô hình dựa trên Transformer, tăng dần độ phức tạp:
1. Transformer cơ bản:
- Đầu vào: (64, 5) - OHLCV (Open, High, Low, Close, Volume).
- Kiến trúc: 4 tầng Transformer Encoder với 6-head Multi-Head Attention, mạng Feedforward (ReLU), và đầu ra phân loại (Global Average Pooling → Dense → Softmax).
2. Transformer + Time2Vec:
- Đầu vào: (64, 8) - OHLCV + 3 đặc trưng thời gian (hour, day_of_week, index).
- Thêm tầng Time2Vec để mã hóa thời gian thành vector 16 chiều, nối với đặc trưng gốc.
- Kiến trúc: Tương tự mô hình cơ bản với đầu vào mở rộng.
3. Transformer + Time2Vec + Chỉ báo kỹ thuật + Chuẩn hóa khối:
- Đầu vào: (64, 35+) - OHLCV, đặc trưng thời gian, và chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, v.v.) tính bằng thư viện ta.
- Áp dụng chuẩn hóa khối để ổn định chuỗi biến động cao.
- Kiến trúc: Tăng cường với Feedforward lớn hơn (Dense 128) và Dropout (0.1).

#### Huấn luyện và tối ưu hóa
Tất cả mô hình được huấn luyện với:
- Kích thước batch: 64
- Số epoch: 15 (với Early Stopping, patience=5)
- Tối ưu hóa: Adam (tốc độ học=0.0001)
- Hàm mất mát: Sparse Categorical Cross Entropy
- Chỉ số: Accuracy

### Đánh giá
#### Độ chính xác phân loại:
- Transformer cơ bản: 47.23%
- Transformer + Time2Vec: 48.54%
- Transformer + Time2Vec + Chỉ báo + Chuẩn hóa khối: 61.94%
#### Backtesting (Tháng 1-3/2022):
- Không có phí giao dịch:
    - Vốn ban đầu: $10,000
    - Giá trị danh mục cuối: $33,955.14
    - Lợi nhuận tổng: 239.55%
    - Tỷ lệ Sharpe: 1.04
    - Mức sụt giảm tối đa: -1.28%
    - Số giao dịch: 1822
- Có phí giao dịch 0.1%:
    - Vốn ban đầu: $10,000
    - Giá trị danh mục cuối: $7,189.17
    - Lợi nhuận tổng: -28.11%
    - Tỷ lệ Sharpe: -0.95
    - Mức sụt giảm tối đa: -28.16%
    - Số giao dịch: 636
Phí giao dịch cao làm giảm đáng kể lợi nhuận, cho thấy cần tối ưu tần suất giao dịch.

## Kết luận
Dự án đã xây dựng thành công một hệ thống giao dịch định lượng dựa trên Transformer để phân loại tín hiệu giao dịch từ dữ liệu giá cổ phiếu tần suất cao. Việc tích hợp Time2Vec, chỉ báo kỹ thuật, và chuẩn hóa khối đã cải thiện đáng kể hiệu suất, đạt độ chính xác phân loại 61.94% và lợi nhuận 239.55% trong backtesting khi không có phí giao dịch. Tuy nhiên, sự nhạy cảm với phí giao dịch nhấn mạnh tầm quan trọng của việc tối ưu tần suất giao dịch và quản lý rủi ro.

### Hướng phát triển
- Giao dịch đa tài sản: Mở rộng mô hình cho nhiều mã cổ phiếu cùng lúc.
- Phân tích đa khung thời gian: Kết hợp dữ liệu M1, M5, M15 để ra quyết định đa tầng.
- Triển khai thời gian thực: Tối ưu độ trễ và tích hợp với API giao dịch (Alpaca, Polygon.io).
- Học trực tuyến: Áp dụng cơ chế học lại định kỳ hoặc học trực tuyến để thích nghi với thay đổi thị trường.
- Quản lý rủi ro: Tích hợp stop-loss, take-profit và định cỡ vị thế để tăng độ bền vững.

Dự án này đặt nền tảng vững chắc cho các nghiên cứu tiếp theo về hệ thống giao dịch tự động, và chúng em rất hào hứng với tiềm năng ứng dụng thực tế trong thị trường tài chính.