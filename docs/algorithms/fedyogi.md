# FedYogi

File chính:

- `src/fl_baselines/algorithms/fedyogi.py`

`FedYogiBuilder` tạo Flower `FedYogi` strategy. Đây là biến thể FedOpt dùng Yogi ở phía server: client vẫn train local bằng SGD như FedAvg, còn server cập nhật moment bậc nhất và bậc hai theo Yogi để ổn định hơn trên dữ liệu heterogeneous.

Config chính:

- `algorithm = "fedyogi"`
- `fedyogi-eta = 0.01`
- `fedyogi-eta-l = 0.0316`
- `fedyogi-beta-1 = 0.9`
- `fedyogi-beta-2 = 0.99`
- `fedyogi-tau = 0.001`

Ghi chú triển khai:

- không cần logic client riêng
- không lưu thêm client-side state
- dùng `CheckpointingStrategyMixin` để giữ checkpoint cuối và checkpoint theo round
