# Quickstart

## Cài Đặt

Cài dependencies đã khóa version:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Chạy Experiment Mặc Định

```bash
flwr run . --stream
```

## Override Config

Override experiment settings bằng Flower run config:

```bash
flwr run . --run-config 'algorithm="fedavg" dataset="mnist" model="mnist_cnn" partitioner="dirichlet" dirichlet-alpha=0.3' --stream
```

Config mặc định nằm trong `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Baseline selection
algorithm = "fedavg"

# Dataset selection
dataset = "mnist"
partitioner = "iid"
dirichlet-alpha = 0.5
seed = 42

# Model selection
model = "mnist_cnn"
input-channels = 1
input-height = 28
input-width = 28
num-classes = 10

# Federation runtime
num-server-rounds = 3
num-supernodes = 10
fraction-train = 1.0
fraction-evaluate = 1.0
client-test-fraction = 0.2

# Client local training
local-epochs = 1
batch-size = 32
learning-rate = 0.01

# FedAvgM
server-learning-rate = 1.0
server-momentum = 0.9

# FedAdp
fedadp-alpha = 5.0

# FedDyn
feddyn-alpha = 0.1

# FedDC
feddc-alpha = 0.01

# FedDecorr
feddecorr-beta = 0.1

# FedExP
fedexp-epsilon = 0.001

# FedSpeed
fedspeed-lambda = 0.1
fedspeed-alpha = 1.0
fedspeed-rho = 0.1

# FedSAM
fedsam-rho = 0.5

# FedGEN
fedgen-alpha = 1.5
fedgen-lambda = 0.1
fedgen-beta = 0.9
fedgen-delta = 0.9
fedgen-warmup-epochs = 1
fedgen-l1-weight = 0.0001

# GAMF
gamf-sigma = 2.0
gamf-initial-tau = 0.05
gamf-descent-factor = 0.9
gamf-min-tau = 0.005
gamf-max-iters = 200

# FedMA
fedma-matching-epsilon = 0.0

# FedCDA
fedcda-memory-size = 3
fedcda-num-batches = 3
fedcda-warmup-rounds = 50
fedcda-loss-weight = 1.0

# FedDRL
feddrl-actor-learning-rate = 0.0001
feddrl-critic-learning-rate = 0.001
feddrl-discount-factor = 0.99
feddrl-target-tau = 0.02
feddrl-hidden-size = 256
feddrl-replay-buffer-size = 100000
feddrl-batch-size = 32
feddrl-updates-per-round = 1
feddrl-noise-scale = 0.1
feddrl-std-scale = 0.5

# FedLAW
fedlaw-server-epochs = 1
fedlaw-server-learning-rate = 0.01
fedlaw-gamma-init = 1.0

# FedProto
fedproto-lambda = 1.0

# FedNTD
fedntd-beta = 1.0
fedntd-temperature = 1.0

# Ditto
ditto-lambda = 0.1

# pFedMe
pfedme-lambda = 15.0
pfedme-beta = 1.0
pfedme-personal-learning-rate = 0.01
pfedme-personal-steps = 5

# FedNova
fednova-server-momentum = 0.0

# FedPer
fedper-personal-layers = 1

# FedRep
fedrep-personal-layers = 1
fedrep-representation-epochs = 1

# FedALA
fedala-eta = 1.0
fedala-rand-percent = 80
fedala-layer-count = 1
fedala-threshold = 0.01
fedala-num-pre-loss = 10
fedala-start-max-steps = 100

# FedAMP
fedamp-lambda = 0.1
fedamp-alpha = 0.1
fedamp-sigma = 1.0

# FedLAA
fedlaa-beta = 5.0

# FedProx
proximal-mu = 0.1

# MOON
moon-mu = 1.0
moon-temperature = 0.5
```

## Đổi Số Client Simulation

Khi đổi số simulated clients, cần giữ hai giá trị này đồng bộ:

```toml
[tool.flwr.app.config]
num-supernodes = 10

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```

`num-supernodes` trong app config được framework dùng để chia partition và cấu hình strategy. `options.num-supernodes` trong Flower federation config điều khiển số supernode mà local simulation tạo ra.

## Evaluation Semantics

- server eval dùng server-side test set
- client eval dùng held-out test split được tách từ local partition của từng client theo `client-test-fraction`
- cả hai luồng đều report `loss`, `accuracy`, `precision`, `recall`, và `f1`
- `precision`, `recall`, và `f1` dùng macro averaging cho multi-class classification

## Chạy FedProx

```bash
flwr run . --run-config 'algorithm="fedprox" proximal-mu=0.1' --stream
```

## Chạy FedAvgM

```bash
flwr run . --run-config 'algorithm="fedavgm" server-learning-rate=1.0 server-momentum=0.9' --stream
```

## Chạy FedNova

```bash
flwr run . --run-config 'algorithm="fednova" fednova-server-momentum=0.0' --stream
```

## Chạy FedAdp

```bash
flwr run . --run-config 'algorithm="fedadp" fedadp-alpha=5.0' --stream
```

## Chạy Ditto

```bash
flwr run . --run-config 'algorithm="ditto" ditto-lambda=0.1' --stream
```

## Chạy FedDyn

```bash
flwr run . --run-config 'algorithm="feddyn" feddyn-alpha=0.1' --stream
```

## Chạy FedDC

```bash
flwr run . --run-config 'algorithm="feddc" feddc-alpha=0.01' --stream
```

## Chạy FedDRL

```bash
flwr run . --run-config 'algorithm="feddrl" feddrl-updates-per-round=1 feddrl-noise-scale=0.1' --stream
```

## Chạy FedLAW

```bash
flwr run . --run-config 'algorithm="fedlaw" fedlaw-server-epochs=3 fedlaw-server-learning-rate=0.01 fedlaw-gamma-init=1.0' --stream
```

## Chạy FedGEN

```bash
flwr run . --run-config 'algorithm="fedgen" fedgen-alpha=1.5 fedgen-lambda=0.1 fedgen-warmup-epochs=1' --stream
```

## Chạy GAMF

```bash
flwr run . --run-config 'algorithm="gamf" model="mnist_cnn" gamf-sigma=2.0 gamf-max-iters=200' --stream
```

## Chạy FedMA

```bash
flwr run . --run-config 'algorithm="fedma" model="mnist_cnn" num-server-rounds=4 fedma-matching-epsilon=0.0' --stream
```

## Chạy FedCDA

```bash
flwr run . --run-config 'algorithm="fedcda" fedcda-memory-size=3 fedcda-num-batches=3 fedcda-warmup-rounds=50' --stream
```

## Chạy FedDecorr

```bash
flwr run . --run-config 'algorithm="feddecorr" feddecorr-beta=0.1' --stream
```

## Chạy FedExP

```bash
flwr run . --run-config 'algorithm="fedexp" fedexp-epsilon=0.001' --stream
```

## Chạy FedSpeed

```bash
flwr run . --run-config 'algorithm="fedspeed" fedspeed-lambda=0.1 fedspeed-alpha=1.0 fedspeed-rho=0.1' --stream
```

## Chạy FedSAM

```bash
flwr run . --run-config 'algorithm="fedsam" fedsam-rho=0.5' --stream
```

## Chạy FedProto

```bash
flwr run . --run-config 'algorithm="fedproto" fedproto-lambda=1.0' --stream
```

## Chạy FedNTD

```bash
flwr run . --run-config 'algorithm="fedntd" fedntd-beta=1.0 fedntd-temperature=1.0' --stream
```

## Chạy pFedMe

```bash
flwr run . --run-config 'algorithm="pfedme" pfedme-lambda=15.0 pfedme-beta=1.0 pfedme-personal-learning-rate=0.01 pfedme-personal-steps=5' --stream
```

## Chạy FedPer

```bash
flwr run . --run-config 'algorithm="fedper" fedper-personal-layers=1' --stream
```

## Chạy FedRep

```bash
flwr run . --run-config 'algorithm="fedrep" fedrep-personal-layers=1 fedrep-representation-epochs=1' --stream
```

## Chạy FedALA

```bash
flwr run . --run-config 'algorithm="fedala" fedala-rand-percent=80 fedala-layer-count=1' --stream
```

## Chạy FedAMP

```bash
flwr run . --run-config 'algorithm="fedamp" fedamp-lambda=0.1 fedamp-alpha=0.1 fedamp-sigma=1.0' --stream
```

## Chạy FedLAA

```bash
flwr run . --run-config 'algorithm="fedlaa" fedlaa-beta=5.0' --stream
```

## Chạy SCAFFOLD

```bash
flwr run . --run-config 'algorithm="scaffold"' --stream
```

## Chạy MOON

```bash
flwr run . --run-config 'algorithm="moon" moon-mu=1.0 moon-temperature=0.5' --stream
```

## Chọn Model Và Shape

Các model builder đọc chung các config sau:

```toml
model = "resnet18"
input-channels = 3
input-height = 32
input-width = 32
num-classes = 10
```

Ví dụ chạy ResNet-18 cho ảnh RGB 32x32:

```bash
flwr run . --run-config 'model="resnet18" input-channels=3 input-height=32 input-width=32 num-classes=10' --stream
```

## Chọn Dataset

Các dataset key hiện có:

```text
mnist, fmnist, emnist, cifar10, cifar100, imagenet
```

Ví dụ chạy CIFAR-10 với ResNet-18:

```bash
flwr run . --run-config 'dataset="cifar10" model="resnet18" input-channels=3 input-height=32 input-width=32 num-classes=10' --stream
```

Ví dụ chạy EMNIST split `letters`:

```bash
flwr run . --run-config 'dataset="emnist" emnist-split="letters" num-classes=26' --stream
```

ImageNet cần dữ liệu đã được chuẩn bị local trong `data-dir`; framework không tự download ImageNet.
