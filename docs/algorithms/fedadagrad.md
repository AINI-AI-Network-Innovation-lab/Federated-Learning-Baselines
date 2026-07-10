# FedAdagrad

File chính:

- `src/fl_baselines/algorithms/fedadagrad.py`

`FedAdagradBuilder` tạo Flower `FedAdagrad` strategy. Đây là biến thể FedOpt dùng Adagrad ở phía server: client vẫn train local bằng SGD như FedAvg, còn server cộng dồn pseudo-gradient per-coordinate để điều chỉnh bước cập nhật global model.

Config chính:

- `algorithm = "fedadagrad"`
- `fedadagrad-eta = 0.1`
- `fedadagrad-eta-l = 0.1`
- `fedadagrad-tau = 1e-9`

Ghi chú triển khai:

- không cần logic client riêng
- không lưu thêm client-side state
- dùng `CheckpointingStrategyMixin` để giữ checkpoint cuối và checkpoint theo round
