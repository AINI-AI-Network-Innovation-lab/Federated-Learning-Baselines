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
| FedMeta | 2018 | [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876) |
| FedPer | 2019 | [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) |
| FedNova | 2020 | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) |
| FedProx | 2020 | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| pFedMe | 2020 | [Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/abs/2006.08848) |
| APFL | 2020 | [Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461) |
| SCAFFOLD | 2020 | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) |
| FedAdp | 2021 | [Fast-Convergent Federated Learning with Adaptive Weighting](https://arxiv.org/abs/2012.00661) |
| FedAvgM | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedAdagrad | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedAdam | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedYogi | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| Ditto | 2021 | [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221) |
| FedDyn | 2021 | [Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263) |
| FedAMP | 2021 | [Personalized Cross-Silo Federated Learning on Non-IID Data](https://arxiv.org/abs/2007.03797) |
| FedRep | 2021 | [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/abs/2102.07078) |
| MOON | 2021 | [Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257) |
| FedDC | 2022 | [FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction](https://arxiv.org/abs/2203.11751) |
| FedDecorr | 2022 | [Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning](https://arxiv.org/abs/2210.00226) |
| FedSAM | 2022 | [Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/abs/2206.02618) |
| FedNTD | 2022 | [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://arxiv.org/abs/2106.03097) |
| FedLC | 2022 | [Federated Learning with Label Distribution Skew via Logits Calibration](https://arxiv.org/abs/2209.00189) |
| FedRS | 2021 | [FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data](https://doi.org/10.1145/3447548.3467254) |
| FedLAMA | 2022 | [Layer-wise Adaptive Model Aggregation for Scalable Federated Learning](https://arxiv.org/abs/2110.10302) |
| FedProto | 2022 | [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243) |
| FedCurv | 2019 | [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/abs/1910.07796) |
| FedMMD | 2024 | [FedMMD: A Federated weighting algorithm considering Non-IID and Local Model Deviation](https://doi.org/10.1016/j.eswa.2023.121463) |
| FedSiKD | 2024 | [FedSiKD: Clients Similarity and Knowledge Distillation: Addressing Non-i.i.d. and Constraints in Federated Learning](https://arxiv.org/abs/2402.09095) |
| FedNP | 2023 | [FedNP: Towards Non-IID Federated Learning via Federated Neural Propagation](https://ojs.aaai.org/index.php/AAAI/article/view/26358) |
| FedDRL | 2022 | [FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning](https://arxiv.org/abs/2208.02442) |
| FedALA | 2023 | [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197) |
| FedExP | 2023 | [FedExP: Speeding up Federated Averaging via Extrapolation](https://arxiv.org/abs/2301.09604) |
| FedSpeed | 2023 | [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://arxiv.org/abs/2302.10429) |
| FedDisco | 2023 | [FedDisco: Federated Learning with Discrepancy-Aware Collaboration](https://proceedings.mlr.press/v202/ye23f.html) |
| FedGEN | 2023 | [FedGen: Generalizable Federated Learning for Sequential Data](https://arxiv.org/abs/2211.01914) |
| GAMF | 2022 | [Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning](https://proceedings.mlr.press/v162/liu22k.html) |
| FedMA | 2020 | [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440) |
| FedCDA | 2024 | [FedCDA: Federated Learning with Cross-Round Divergence-Aware Aggregation](https://openreview.net/forum?id=nbPGqeH3lt) |
| FedEnt | 2024 | [Adaptive Federated Learning via New Entropy Approach](https://arxiv.org/abs/2303.14966) |
| FedLAW | 2024 | [Revisiting Weighted Aggregation in Federated Learning with Neural Networks](https://arxiv.org/abs/2302.10911) |
| FedAAW | 2025 | [Federated Learning With Adaptive Aggregation Weights for Non-IID Data in Edge Networks](https://doi.org/10.1109/TCCN.2025.3534248) |
| FedVCK | 2025 | [FedVCK: Non-IID Robust and Communication-Efficient Federated Learning via Valuable Condensed Knowledge for Medical Image Analysis](https://arxiv.org/abs/2412.18557) |
| FedLAA | 2026 | [Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation](https://doi.org/10.1016/j.asoc.2026.115676) |

Ghi chú: registry hiện hỗ trợ thêm alias `feddrrl` trỏ tới baseline canonical `feddrl` (`FedDRL`) để tương thích cấu hình cũ.

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
