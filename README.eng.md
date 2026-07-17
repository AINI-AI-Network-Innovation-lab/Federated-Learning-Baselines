# FL Baselines Using Flower

<p align="left">
  <strong>A Flower + PyTorch framework for reproducible federated learning baselines.</strong>
</p>

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Flower" src="https://img.shields.io/badge/Flower-Federated%20Learning-ffb000">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
</p>

> Vietnamese version: [README.md](README.md)

This framework focuses on one goal: providing clear, reproducible, and extensible federated learning baselines across three core extension points: `dataset`, `model`, and `algorithm`.

Detailed documentation is available in `docs/`.

> Server evaluation uses the server-side test set, while client evaluation uses a held-out split from each client’s local partition.
>
> At both evaluation levels, the framework reports `loss`, `accuracy`, `precision`, `recall`, and `f1`; the last three metrics use macro averaging for multi-class classification.

## At a Glance

| What you get | Details |
| --- | --- |
| Extensible registry | Add datasets, models, and algorithms through dedicated builders without touching Flower’s main flow. |
| Broad baseline coverage | Supports baselines from FedAvg, FedProx, and SCAFFOLD to personalized, adaptive, and distillation-based methods. |
| Reproducible evaluation | Separates server evaluation and client evaluation while keeping metrics consistent throughout the pipeline. |
| Project docs | Includes quickstart, architecture, baseline extension guidance, and per-algorithm descriptions. |

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
flwr run . --stream
```

## Project Structure

| Path | Description |
| --- | --- |
| `src/fl_baselines/` | Main framework source code, including the Flower app, registry, datasets, models, algorithms, clients, training, and logging. |
| `tests/` | Unit tests for config, registry, partitioning, model, and algorithm builders. |
| `docs/` | Detailed project documentation, code architecture, runtime instructions, and baseline extension guidance. |
| `configs/` | Space for notes or presets as the project grows; the primary runtime config currently lives in `pyproject.toml`. |
| `pyproject.toml` | Package metadata, Flower app config, default settings, and local simulation wiring. |
| `requirements.txt` | Version-pinned runtime dependencies for the current release. |

## Configuration

```bash
flwr run . --run-config 'algorithm="fedavg" dataset="mnist" model="mnist_cnn" partitioner="dirichlet" dirichlet-alpha=0.3' --stream
```

## Baselines

| Baseline | Year | Paper |
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

## Datasets

| Dataset key | Description |
| --- | --- |
| `mnist` | MNIST digits |
| `fmnist` | Fashion-MNIST |
| `emnist` | EMNIST, configured with `emnist-split` |
| `cifar10` | CIFAR-10 |
| `cifar100` | CIFAR-100 |
| `imagenet` | Local ImageNet folder; manual data preparation required |

## Models

| Model key | Description |
| --- | --- |
| `mnist_cnn` | Small CNN for MNIST |
| `lenet` | LeNet-style CNN with configurable input and output |
| `resnet9` | Internal ResNet-9 with configurable input and output |
| `resnet18` | TorchVision ResNet-18 with configurable input and output |
| `resnet34` | TorchVision ResNet-34 with configurable input and output |
| `inception` | TorchVision Inception v3 with configurable input and output |
