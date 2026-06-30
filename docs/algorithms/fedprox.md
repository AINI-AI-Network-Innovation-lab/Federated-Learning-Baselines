# FedProx

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedprox.py`
- `src/fl_baselines/training/train.py`

`FedProxBuilder` tạo Flower `FedProx` strategy. Strategy này giống FedAvg ở phía server aggregation, nhưng gửi thêm `proximal_mu` xuống client. Client training loop dùng proximal term:

```text
(mu / 2) * ||w - w_global||^2
```

Config chính:

- `algorithm = "fedprox"`
- `proximal-mu = 0.1`
