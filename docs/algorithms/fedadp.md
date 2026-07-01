# FedAdp

File chính:

- `src/fl_baselines/algorithms/fedadp.py`

`FedAdpBuilder` tạo `FedAdpStrategy`, một Flower strategy dùng sampling giống FedAvg nhưng thay trọng số aggregation theo đóng góp của từng client. Implementation dựa trên paper *Fast-Convergent Federated Learning with Adaptive Weighting*.

Ở mỗi round, server:

1. Nhận model parameters sau local training như FedAvg.
2. Suy ra local update `delta_i = local_model_i - global_model`.
3. Tính global update bằng sample-weighted average của các local update.
4. Tính góc giữa local update và global update cho từng client.
5. Cập nhật smoothed angle theo trung bình chạy qua các lần client tham gia.
6. Áp dụng Gompertz mapping với `fedadp-alpha`, sau đó dùng sample-weighted softmax để lấy trọng số aggregation.

Client không cần logic riêng và không gửi thêm metric bắt buộc. Vì FedAdp chỉ xử lý vector parameters ở server, thuật toán không phụ thuộc vào kiến trúc model cụ thể miễn là model train/evaluate được theo contract chung.

Config chính:

- `algorithm = "fedadp"`
- `fedadp-alpha = 5.0`

Chạy nhanh:

```bash
flwr run . --run-config 'algorithm="fedadp" fedadp-alpha=5.0' --stream
```
