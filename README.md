# FL Baselines Using Flower

<p align="left">
  <strong>Khung thực nghiệm Federated Learning xây trên Flower và PyTorch, tập trung vào các thuật toán nền rõ ràng, dễ chạy lại và dễ mở rộng.</strong>
</p>

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Flower" src="https://img.shields.io/badge/Flower-Federated%20Learning-ffb000">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
</p>

> Bản tiếng Anh: [README.eng.md](README.eng.md)

Mục tiêu của khung này là cung cấp các thuật toán nền federated learning minh bạch, có thể tái lập và dễ mở rộng qua ba điểm mở rộng cốt lõi: `dataset`, `model`, và `algorithm`.

Tài liệu chi tiết nằm trong `docs/`.

> Đánh giá phía server dùng tập kiểm tra ở phía server, còn đánh giá phía client dùng một phần giữ lại tách từ phân hoạch cục bộ của từng client.
>
> Ở cả hai mức đánh giá, khung báo cáo `loss`, `accuracy`, `precision`, `recall`, và `f1`; ba metric cuối dùng trung bình macro cho bài toán phân loại nhiều lớp.

## Tóm Tắt Nhanh

| Bạn nhận được | Chi tiết |
| --- | --- |
| Registry mở rộng được | Thêm dataset, model, algorithm bằng builder riêng, không phải sửa luồng chính của Flower. |
| Nhiều thuật toán nền sẵn có | Hỗ trợ từ FedAvg, FedProx, SCAFFOLD đến các hướng cá nhân hóa, thích ứng, và dựa trên chưng cất tri thức. |
| Đánh giá có thể tái lập | Tách rõ đánh giá phía server và đánh giá phía client, đồng bộ metric trong toàn quy trình. |
| Tài liệu đầy đủ | Có hướng dẫn bắt đầu nhanh, kiến trúc, hướng dẫn mở rộng thuật toán nền, và mô tả từng thuật toán. |

## Bắt Đầu Nhanh

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
flwr run . --stream
```

## Cấu Trúc Dự Án

| Đường dẫn | Mô tả |
| --- | --- |
| `src/fl_baselines/` | Mã nguồn chính của khung FL, gồm app Flower, registry, dataset, model, algorithm, client, training và logging. |
| `tests/` | Unit tests cho config, registry, partitioning, model và algorithm builder. |
| `docs/` | Tài liệu chi tiết về dự án, kiến trúc mã nguồn, cách chạy và cách mở rộng thuật toán nền. |
| `configs/` | Nơi ghi chú hoặc preset config khi project phát triển thêm; hiện runtime config chính nằm trong `pyproject.toml`. |
| `pyproject.toml` | Metadata package, cấu hình Flower app, config mặc định và local simulation. |
| `requirements.txt` | Các dependency runtime đã khóa version cho phiên bản hiện tại. |

## Cách Cấu Hình

```bash
flwr run . --run-config 'algorithm="fedavg" dataset="mnist" model="mnist_cnn" partitioner="dirichlet" dirichlet-alpha=0.3' --stream
```

### FedADMM đầy đủ

Đặt `algorithm="fedadmm"` để chạy triển khai bám sát Algorithm 2 trong [FedADMM](https://arxiv.org/abs/2203.15104). Thuật toán hỗ trợ partial participation, trạng thái client/server được lưu qua các round, và các phép prox `identity`, `l1`, `box`.

```bash
flwr run . --run-config 'algorithm="fedadmm" fedadmm-penalty=1.0 fedadmm-prox="identity" fedadmm-local-steps=300 batch-size=2 learning-rate=0.01' --stream
```

State được lưu tại `outputs/fedadmm_clients/<client-id>/state.pt` và `outputs/fedadmm_server/state.pt`. Tùy chọn `fedadmm-alpha` vẫn được giữ làm alias tương thích cho penalty cũ.

## Thuật Toán Nền

| Thuật toán nền | Năm | Bài báo |
| --- | --- | --- |
| FedAvg | 2017 | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |
| FedMeta | 2018 | [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876) |
| FedCurv | 2019 | [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/abs/1910.07796) |
| FedPer | 2019 | [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) |
| APFL | 2020 | [Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461) |
| FedMA | 2020 | [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440) |
| FedNova | 2020 | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) |
| FedProx | 2020 | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) |
| pFedMe | 2020 | [Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/abs/2006.08848) |
| SCAFFOLD | 2020 | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) |
| Ditto | 2021 | [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221) |
| FedAdagrad | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedAdp | 2021 | [Fast-Convergent Federated Learning with Adaptive Weighting](https://arxiv.org/abs/2012.00661) |
| FedAdam | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedAMP | 2021 | [Personalized Cross-Silo Federated Learning on Non-IID Data](https://arxiv.org/abs/2007.03797) |
| FedAvgM | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedDyn | 2021 | [Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263) |
| FedRep | 2021 | [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/abs/2102.07078) |
| FedRS | 2021 | [FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data](https://doi.org/10.1145/3447548.3467254) |
| FedYogi | 2021 | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| MOON | 2021 | [Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257) |
| FedDC | 2022 | [FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction](https://arxiv.org/abs/2203.11751) |
| FedADMM | 2022 | [FedADMM: A Federated Primal-Dual Algorithm Allowing Partial Participation](https://arxiv.org/abs/2203.15104) |
| FedDecorr | 2022 | [Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning](https://arxiv.org/abs/2210.00226) |
| FedDRL | 2022 | [FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning](https://arxiv.org/abs/2208.02442) |
| FedLAMA | 2022 | [Layer-wise Adaptive Model Aggregation for Scalable Federated Learning](https://arxiv.org/abs/2110.10302) |
| FedLC | 2022 | [Federated Learning with Label Distribution Skew via Logits Calibration](https://arxiv.org/abs/2209.00189) |
| FedNTD | 2022 | [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://arxiv.org/abs/2106.03097) |
| FedProto | 2022 | [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243) |
| FedSAM | 2022 | [Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/abs/2206.02618) |
| GAMF | 2022 | [Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning](https://proceedings.mlr.press/v162/liu22k.html) |
| FedALA | 2023 | [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197) |
| FedDisco | 2023 | [FedDisco: Federated Learning with Discrepancy-Aware Collaboration](https://proceedings.mlr.press/v202/ye23f.html) |
| FedExP | 2023 | [FedExP: Speeding up Federated Averaging via Extrapolation](https://arxiv.org/abs/2301.09604) |
| FedGEN | 2023 | [FedGen: Generalizable Federated Learning for Sequential Data](https://arxiv.org/abs/2211.01914) |
| FedNP | 2023 | [FedNP: Towards Non-IID Federated Learning via Federated Neural Propagation](https://ojs.aaai.org/index.php/AAAI/article/view/26358) |
| FedSpeed | 2023 | [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://arxiv.org/abs/2302.10429) |
| FedCDA | 2024 | [FedCDA: Federated Learning with Cross-Round Divergence-Aware Aggregation](https://openreview.net/forum?id=nbPGqeH3lt) |
| FedEnt | 2024 | [Adaptive Federated Learning via New Entropy Approach](https://arxiv.org/abs/2303.14966) |
| FedLAW | 2024 | [Revisiting Weighted Aggregation in Federated Learning with Neural Networks](https://arxiv.org/abs/2302.10911) |
| FedMMD | 2024 | [FedMMD: A Federated weighting algorithm considering Non-IID and Local Model Deviation](https://doi.org/10.1016/j.eswa.2023.121463) |
| FedSiKD | 2024 | [FedSiKD: Clients Similarity and Knowledge Distillation: Addressing Non-i.i.d. and Constraints in Federated Learning](https://arxiv.org/abs/2402.09095) |
| FedAAW | 2025 | [Federated Learning With Adaptive Aggregation Weights for Non-IID Data in Edge Networks](https://doi.org/10.1109/TCCN.2025.3534248) |
| FedLWS | 2025 | [FedLWS: Federated Learning with Adaptive Layer-wise Weight Shrinking](https://arxiv.org/abs/2503.15111) |
| FedVCK | 2025 | [FedVCK: Non-IID Robust and Communication-Efficient Federated Learning via Valuable Condensed Knowledge for Medical Image Analysis](https://arxiv.org/abs/2412.18557) |
| FedLAA | 2026 | [Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation](https://doi.org/10.1016/j.asoc.2026.115676) |

## Tập Dữ Liệu

| Khóa tập dữ liệu | Mô tả |
| --- | --- |
| `mnist` | Chữ số MNIST |
| `fmnist` | Fashion-MNIST |
| `emnist` | EMNIST, cấu hình split bằng `emnist-split` |
| `cifar10` | CIFAR-10 |
| `cifar100` | CIFAR-100 |
| `imagenet` | Thư mục ImageNet cục bộ, cần chuẩn bị dữ liệu thủ công |

## Mô Hình

| Khóa mô hình | Mô tả |
| --- | --- |
| `mnist_cnn` | CNN nhỏ cho MNIST |
| `lenet` | CNN kiểu LeNet, có thể cấu hình đầu vào/đầu ra |
| `resnet9` | ResNet-9 nội bộ, có thể cấu hình đầu vào/đầu ra |
| `resnet18` | TorchVision ResNet-18, có thể cấu hình đầu vào/đầu ra |
| `resnet34` | TorchVision ResNet-34, có thể cấu hình đầu vào/đầu ra |
| `inception` | TorchVision Inception v3, có thể cấu hình đầu vào/đầu ra |
