# FL Baselines Using Flower

Framework thực nghiệm Federated Learning xây trên Flower và PyTorch, tập trung vào cấu trúc dễ đọc, dễ dùng và dễ mở rộng cho ba nhóm đối tượng chính: `dataset`, `model`, và `algorithm`.

Tài liệu chi tiết nằm trong `docs/`.

Pipeline evaluation hiện dùng hai mức rõ ràng:

- server eval trên server-side test set
- client eval trên held-out test split được tách từ chính local partition của từng client

Ở cả hai mức evaluation, framework hiện report `loss`, `accuracy`, `precision`, `recall`, và `f1`; ba metric cuối dùng macro averaging cho bài toán multi-class.

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
| FedPer | 2019 | [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) |
| FedNova | 2020 | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) |
| FedProx | 2020 | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| pFedMe | 2020 | [Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/abs/2006.08848) |
| SCAFFOLD | 2020 | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) |
| FedAdp | 2021 | [Fast-Convergent Federated Learning with Adaptive Weighting](https://arxiv.org/abs/2012.00661) |
| FedAvgM | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| Ditto | 2021 | [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221) |
| FedDyn | 2021 | [Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263) |
| FedRep | 2021 | [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/abs/2102.07078) |
| MOON | 2021 | [Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257) |
| FedDC | 2022 | [FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction](https://arxiv.org/abs/2203.11751) |
| FedDecorr | 2022 | [Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning](https://arxiv.org/abs/2210.00226) |
| FedSAM | 2022 | [Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/abs/2206.02618) |
| FedNTD | 2022 | [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://arxiv.org/abs/2106.03097) |
| FedProto | 2022 | [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243) |
| FedExP | 2023 | [FedExP: Speeding up Federated Averaging via Extrapolation](https://arxiv.org/abs/2301.09604) |
| FedSpeed | 2023 | [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://arxiv.org/abs/2302.10429) |

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
