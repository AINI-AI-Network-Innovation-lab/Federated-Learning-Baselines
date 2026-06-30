# Documentation Guide

Thư mục `docs/` chứa tài liệu chi tiết cho framework Federated Learning Baselines. Bắt đầu từ `guide.md` nếu bạn muốn đọc theo thứ tự, hoặc dùng bảng bên dưới để đi thẳng tới chủ đề cần xem.

## Reading Order

1. [overview.md](overview.md): hiểu mục tiêu project, phạm vi framework và baseline hiện tại.
2. [quickstart.md](quickstart.md): cài đặt môi trường, chạy experiment đầu tiên và override config.
3. [architecture.md](architecture.md): nắm luồng chạy Flower server/client, registry, config và artifact.
4. [datasets.md](datasets.md), [models.md](models.md), [algorithms/index.md](algorithms/index.md): xem các component hiện có.
5. [extending-baselines.md](extending-baselines.md): thêm dataset, model, algorithm hoặc baseline hoàn chỉnh.
6. [testing-and-artifacts.md](testing-and-artifacts.md): chạy test, kiểm tra thay đổi và hiểu output artifacts.

## File Map

| File | Nội dung |
| --- | --- |
| [guide.md](guide.md) | Mục lục đọc nhanh cho toàn bộ docs. |
| [overview.md](overview.md) | Tổng quan project, thiết kế chính và các baseline đang hỗ trợ. |
| [quickstart.md](quickstart.md) | Hướng dẫn setup, chạy `flwr run`, override config và chọn component. |
| [architecture.md](architecture.md) | Kiến trúc code, luồng server/client, registry, config, logging và checkpoint. |
| [datasets.md](datasets.md) | Dataset builders, nguồn dữ liệu, loader và partitioning. |
| [models.md](models.md) | Model builders, MNIST CNN và các vision backbone configurable. |
| [algorithms/index.md](algorithms/index.md) | Mục lục algorithm; mỗi algorithm có một file riêng. |
| [extending-baselines.md](extending-baselines.md) | Quy trình thêm dataset/model/algorithm mới và baseline hoàn chỉnh. |
| [testing-and-artifacts.md](testing-and-artifacts.md) | Unit tests, compile check, metrics, checkpoints và artifacts. |

## Algorithms

Mỗi algorithm có một file riêng trong `docs/algorithms/` để dễ mở rộng:

| File | Nội dung |
| --- | --- |
| [algorithms/index.md](algorithms/index.md) | Mục lục algorithm. |
| [algorithms/fedavg.md](algorithms/fedavg.md) | FedAvg strategy builder và config liên quan. |
| [algorithms/fedavgm.md](algorithms/fedavgm.md) | FedAvgM và server momentum. |
| [algorithms/fednova.md](algorithms/fednova.md) | FedNova normalized averaging và client update metrics. |
| [algorithms/fedper.md](algorithms/fedper.md) | FedPer shared base aggregation và personalized local head. |
| [algorithms/fedrep.md](algorithms/fedrep.md) | FedRep shared representation aggregation và two-phase local training. |
| [algorithms/fedprox.md](algorithms/fedprox.md) | FedProx strategy và proximal term ở client training. |
| [algorithms/scaffold.md](algorithms/scaffold.md) | SCAFFOLD control variates phía server/client. |
| [algorithms/moon.md](algorithms/moon.md) | MOON local model-contrastive training. |
