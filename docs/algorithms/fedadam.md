# FedAdam

File chính:

- `src/fl_baselines/algorithms/fedadam.py`

`FedAdamBuilder` tạo Flower `FedAdam` strategy. Thuật toán này thuộc họ FedOpt trong paper "Adaptive Federated Optimization": client vẫn train local bằng SGD như FedAvg, còn server xem trung bình model delta như pseudo-gradient và cập nhật global model bằng Adam-style adaptive optimizer.

Config chính:

- `algorithm = "fedadam"`
- `fedadam-eta = 0.1`
- `fedadam-eta-l = 0.1`
- `fedadam-beta-1 = 0.9`
- `fedadam-beta-2 = 0.99`
- `fedadam-tau = 1e-9`

Ghi chú triển khai:

- không cần routing client riêng trong `TorchFlowerClient`
- không lưu thêm client-side state
- dùng `CheckpointingStrategyMixin` giống các strategy server-side khác để giữ checkpoint cuối và checkpoint theo round
- `fedadam-eta-l` được truyền vào Flower strategy để nhất quán với FedOpt/FedAdam API; local optimizer trong client vẫn dùng `learning-rate`
