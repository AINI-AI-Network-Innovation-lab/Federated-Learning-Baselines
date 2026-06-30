# FL Baselines Using Flower

Framework thực nghiệm Federated Learning xây trên Flower và PyTorch, tập trung vào cấu trúc dễ đọc, dễ dùng và dễ mở rộng cho ba nhóm đối tượng chính: `dataset`, `model`, và `algorithm`.

Tài liệu chi tiết nằm trong `docs/`.

## Project Folders

| Path | Mô tả |
| --- | --- |
| `src/fl_baselines/` | Source code chính của framework FL, gồm app Flower, registry, dataset, model, algorithm, client, training và logging. |
| `tests/` | Unit tests cho config, registry, partitioning, model và algorithm builder. |
| `docs/` | Tài liệu chi tiết về project, kiến trúc code, cách chạy và cách mở rộng baseline. |
| `configs/` | Nơi để ghi chú hoặc preset config khi project phát triển thêm; hiện runtime config chính nằm trong `pyproject.toml`. |
| `pyproject.toml` | Metadata package, cấu hình Flower app, config mặc định và local simulation. |
| `requirements.txt` | Các dependency runtime đã khóa version cho phiên bản hiện tại. |

## Baselines

| Baseline | Year | Paper |
| --- | --- | --- |
| FedAvg | 2017 | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |
| FedAvgM | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedNova | 2020 | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) |
| FedPer | 2019 | [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) |
| FedRep | 2021 | [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/abs/2102.07078) |
| FedProx | 2020 | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| SCAFFOLD | 2020 | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) |
| MOON | 2021 | [Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257) |

## Datasets

| Dataset key | Mô tả |
| --- | --- |
| `mnist` | MNIST digits |
| `fmnist` | Fashion-MNIST |
| `emnist` | EMNIST, cấu hình split bằng `emnist-split` |
| `cifar10` | CIFAR-10 |
| `cifar100` | CIFAR-100 |
| `imagenet` | ImageNet local folder, cần chuẩn bị dữ liệu thủ công |

## Models

| Model key | Mô tả |
| --- | --- |
| `mnist_cnn` | CNN nhỏ cho MNIST |
| `lenet` | LeNet-style CNN, configurable input/output |
| `resnet9` | ResNet-9 nội bộ, configurable input/output |
| `resnet18` | TorchVision ResNet-18, configurable input/output |
| `resnet34` | TorchVision ResNet-34, configurable input/output |
| `inception` | TorchVision Inception v3, configurable input/output |
