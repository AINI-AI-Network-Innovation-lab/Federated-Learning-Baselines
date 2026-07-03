# Extending Baselines

## Cách Mở Rộng Dataset

Tạo file mới, ví dụ `src/fl_baselines/datasets/cifar10.py`:

```python
from torch.utils.data import DataLoader

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ClientDataLoaders


class Cifar10DatasetBuilder:
    name = "cifar10"

    def build_client_loaders(
        self,
        config: ExperimentConfig,
        partition_id: int,
        num_partitions: int,
    ) -> ClientDataLoaders:
        # Load train/test data, partition theo partition_id,
        # rồi trả về ClientDataLoaders(train=..., test=...)
        ...

    def build_server_loader(self, config: ExperimentConfig) -> DataLoader:
        # Trả về dataloader dùng cho centralized/server evaluation
        ...
```

Đăng ký trong `src/fl_baselines/defaults.py`:

```python
from fl_baselines.datasets.cifar10 import Cifar10DatasetBuilder

if "cifar10" not in DATASETS:
    DATASETS.register("cifar10", Cifar10DatasetBuilder())
```

Chạy:

```bash
flwr run . --run-config 'dataset="cifar10"' --stream
```

Dataset builder nên đảm bảo:

- partition deterministic theo `config.seed`
- không làm mất sample khi chia client
- `train` loader dùng shuffle nếu cần
- `test/server` loader không shuffle để metric ổn định

## Cách Mở Rộng Model

Tạo file mới, ví dụ `src/fl_baselines/models/my_model.py`:

```python
import torch
from torch import nn

from fl_baselines.core.config import ExperimentConfig


class MyModelBuilder:
    name = "my_model"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        return MyModel(...)
```

Đăng ký trong `src/fl_baselines/defaults.py`:

```python
from fl_baselines.models.my_model import MyModelBuilder

if "my_model" not in MODELS:
    MODELS.register("my_model", MyModelBuilder())
```

Chạy:

```bash
flwr run . --run-config 'model="my_model"' --stream
```

Model builder nên đảm bảo:

- mỗi lần gọi `build_model` trả về model mới, không reuse instance cũ
- output shape phù hợp với dataset/task
- model có thể serialize bằng `state_dict`
- client và server dùng cùng architecture

Các dataset vision có sẵn gồm:

- `mnist`
- `fmnist`
- `emnist`
- `cifar10`
- `cifar100`
- `imagenet`

Các model vision có sẵn đã hỗ trợ config input/output gồm:

- `lenet`
- `resnet9`
- `resnet18`
- `resnet34`
- `inception`

Ví dụ dùng ResNet-34 cho dataset RGB 64x64 với 100 class:

```bash
flwr run . --run-config 'model="resnet34" input-channels=3 input-height=64 input-width=64 num-classes=100' --stream
```

## Cách Mở Rộng Algorithm

Tạo file mới, ví dụ `src/fl_baselines/algorithms/fedprox.py`:

```python
import torch
from flwr.server.strategy import Strategy

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn


class FedProxBuilder:
    name = "fedprox"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> Strategy:
        # Tạo và trả về Flower Strategy tương ứng.
        ...
```

Đăng ký trong `src/fl_baselines/defaults.py`:

```python
from fl_baselines.algorithms.fedprox import FedProxBuilder

if "fedprox" not in ALGORITHMS:
    ALGORITHMS.register("fedprox", FedProxBuilder())
```

Chạy:

```bash
flwr run . --run-config 'algorithm="fedprox"' --stream
```

Các algorithm hiện có:

- `fedavg`
- `fedavgm`
- `fedadp`
- `ditto`
- `feddc`
- `fedent`
- `fedvck`
- `feddyn`
- `fedexp`
- `fedsam`
- `fedntd`
- `fedproto`
- `fednova`
- `pfedme`
- `fedper`
- `fedrep`
- `fedprox`
- `scaffold`
- `moon`

Algorithm builder nên đảm bảo:

- nhận `ExperimentConfig`, model ban đầu, và optional `evaluate_fn`
- không hard-code dataset/model cụ thể
- dùng metric aggregation chung nếu phù hợp
- nếu có hyperparameter mới, thêm field vào `ExperimentConfig` và default trong `pyproject.toml`
- nếu thuật toán cần state phía client, ví dụ SCAFFOLD cần client control variates, MOON cần local model round trước, pFedMe/Ditto cần persisted personalized model, hoặc FedPer/FedRep cần personal head theo client, hãy cô lập logic đó trong `clients/` và `training/` thay vì hard-code dataset/model
- nếu thuật toán cần state ở cả client và payload tensor phía server, như FedDC hoặc SCAFFOLD, hãy truyền state đó bằng parameter payload thay vì nhét vào scalar metrics
- nếu thuật toán chỉ đổi server aggregation mà vẫn giữ local training path mặc định, như FedExP, ưu tiên implement gọn trong `algorithms/` thay vì mở nhánh riêng ở `TorchFlowerClient`
- nếu thuật toán regularize tren embedding space nhu FedProto, uu tien them helper trich feature rieng thay vi thay doi `forward()` contract cua toan bo model
- nếu thuật toán regularize tren embedding space nhưng vẫn trả raw local model như FedDecorr, hãy giữ server strategy kiểu FedAvg và cô lập regularizer trong `training/`
- nếu thuật toán chỉ đổi local objective nhưng không cần client state xuyên round, như FedNTD, vẫn nên tách helper riêng trong `training/` và route bằng nhánh `algorithm` trong `TorchFlowerClient`
- nếu thuật toán chỉ đổi local optimizer nhưng vẫn aggregate như FedAvg, như FedSAM, uu tien them helper local training rieng va giu server strategy don gian
- nếu thuật toán cần server-side shared state de tinh learning rate local nhưng van aggregate nhu FedAvg, nhu FedEnt, uu tien them strategy rieng de quan ly fit-config va mean-field state thay vi sua app entrypoints
- nếu thuật toán cần client state xuyên round và client payload khác local model thô, như FedSpeed, hãy persist state theo `client_id` trong `output-dir` và trả payload mới thay vì sửa app entrypoints
- nếu thuật toán cần client condensed payload va server replay tren memory tich luy, nhu FedVCK, uu tien tach helper local/server trong `training/`, giu payload order co dinh, va de strategy tu quan ly memory cap thay vi day logic vao app entrypoints

## Cách Thêm Một Baseline Hoàn Chỉnh

Một baseline trong project này thường là tổ hợp:

```text
baseline = algorithm + dataset + model + config
```

Ví dụ thêm baseline `FedProx + CIFAR-10 + ResNet18`:

1. Thêm dataset builder `Cifar10DatasetBuilder`.
2. Thêm model builder `ResNet18Builder`.
3. Dùng algorithm builder `FedProxBuilder` hiện có hoặc thêm algorithm builder mới nếu baseline khác FedProx.
4. Đăng ký cả ba trong `register_default_components`.
5. Thêm config/hyperparameter mới vào `ExperimentConfig` nếu cần.
6. Thêm dòng baseline vào bảng trong `README.md`.
7. Thêm test cho registry/config/component mới.
8. Chạy:

```bash
python -m unittest discover -s tests -v
flwr run . --run-config 'algorithm="fedprox" dataset="cifar10" model="resnet18"' --stream
```

Nguyên tắc quan trọng: app entrypoints trong `src/fl_baselines/app/` không nên phải sửa khi thêm baseline mới. Nếu phải sửa `client_app.py` hoặc `server_app.py` cho một baseline cụ thể, abstraction có thể đang bị rò rỉ.
