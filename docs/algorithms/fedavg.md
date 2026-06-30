# FedAvg

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedavg.py`

`FedAvgBuilder` tạo Flower `FedAvg` strategy với:

- initial parameters từ model ban đầu
- `fraction_fit`
- `fraction_evaluate`
- `min_fit_clients`
- `min_evaluate_clients`
- server-side evaluation function
- weighted metric aggregation
- checkpoint model sau mỗi round
