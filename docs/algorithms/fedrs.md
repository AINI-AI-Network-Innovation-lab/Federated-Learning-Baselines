# FedRS

File chính:

- `src/fl_baselines/algorithms/fedrs.py`
- `src/fl_baselines/training/fedrs.py`

`FedRSBuilder` giữ server aggregation kiểu FedAvg, còn local client dùng restricted softmax để giảm indirect pushing lên các lớp bị thiếu trong label distribution skew. Hệ số `alpha` điều khiển mức đóng góp của missing classes trong mẫu số softmax: `alpha = 1.0` gần với FedAvg, còn nhỏ hơn sẽ giới hạn update của missing-class proxies.

Config chính:

- `algorithm = "fedrs"`
- `fedrs-alpha = 0.5`

Ghi chú triển khai:

- client tự suy ra `observed_classes` từ local `train_loader`
- không cần state phía client hay payload đặc biệt cho server
- server vẫn dùng `CheckpointingFedAvg`
