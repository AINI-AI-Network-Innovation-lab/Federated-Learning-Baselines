# FedAvgM

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedavgm.py`

`FedAvgMBuilder` tạo Flower `FedAvgM` strategy. Thuật toán này giữ local training giống FedAvg, nhưng server update dùng momentum để giảm dao động khi dữ liệu client non-IID.

Config chính:

- `algorithm = "fedavgm"`
- `server-learning-rate = 1.0`
- `server-momentum = 0.9`
