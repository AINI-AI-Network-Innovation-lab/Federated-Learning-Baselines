# Architecture

## Project Layout

```text
src/fl_baselines/
├── app/          # Flower ClientApp và ServerApp entrypoints
├── core/         # config, registry, shared types
├── datasets/     # dataset builders và partitioners
├── models/       # model builders
├── algorithms/   # FL algorithm builders
├── clients/      # Flower NumPyClient wrapper cho PyTorch
├── training/     # pure PyTorch train/evaluate loops
└── logging/      # metrics aggregation và artifact saving
```

## Luồng Chạy Server

`src/fl_baselines/app/server_app.py` là entrypoint server của Flower:

1. Gọi `register_default_components()` để đăng ký dataset/model/algorithm có sẵn.
2. Parse `context.run_config` thành `ExperimentConfig`.
3. Lấy component từ registry:
   - `DATASETS.get(config.dataset)`
   - `MODELS.get(config.model)`
   - `ALGORITHMS.get(config.algorithm)`
4. Khởi tạo model ban đầu.
5. Tạo server evaluation loader từ dataset.
6. Tạo Flower strategy từ algorithm builder.
7. Trả về `ServerAppComponents(strategy=..., config=...)`.

## Luồng Chạy Client

`src/fl_baselines/app/client_app.py` là entrypoint client của Flower:

1. Gọi `register_default_components()`.
2. Parse config.
3. Xác định `partition_id` và `num_partitions`.
4. Lấy dataset/model builder từ registry.
5. Tạo dataloader cho client partition.
6. Tạo model riêng cho client.
7. Wrap bằng `TorchFlowerClient`.

## Registry

`src/fl_baselines/core/registry.py` cung cấp registry chung:

```python
DATASETS.register("mnist", MnistDatasetBuilder())
MODELS.register("mnist_cnn", MnistCnnBuilder())
ALGORITHMS.register("fedavg", FedAvgBuilder())
```

Lookup ở runtime:

```python
dataset = DATASETS.get(config.dataset)
model = MODELS.get(config.model)
algorithm = ALGORITHMS.get(config.algorithm)
```

Nếu key sai, framework sẽ fail rõ ràng, ví dụ:

```text
Unknown algorithm 'fednova'. Available: fedavg, fedavgm, fedprox, moon, scaffold
```

## Config

`src/fl_baselines/core/config.py` định nghĩa `ExperimentConfig`. Đây là object config typed dùng chung trong toàn bộ framework.

| Field | Ý nghĩa |
| --- | --- |
| `algorithm` | Tên algorithm trong registry |
| `dataset` | Tên dataset trong registry |
| `model` | Tên model trong registry |
| `num_server_rounds` | Số round FL |
| `num_supernodes` | Số client/supernode logic |
| `fraction_train` | Tỷ lệ client tham gia train mỗi round |
| `fraction_evaluate` | Tỷ lệ client tham gia evaluate mỗi round |
| `local_epochs` | Số epoch local training |
| `batch_size` | Batch size cho dataloader |
| `learning_rate` | Learning rate local optimizer |
| `proximal_mu` | Hệ số proximal term cho FedProx |
| `moon_mu` | Hệ số model-contrastive loss cho MOON |
| `moon_temperature` | Temperature cho contrastive logits của MOON |
| `server_learning_rate` | Learning rate của server optimizer cho FedAvgM |
| `server_momentum` | Momentum của server optimizer cho FedAvgM |
| `input_channels` | Số channel của input model |
| `input_height` | Chiều cao input model |
| `input_width` | Chiều rộng input model |
| `num_classes` | Số class/output logits |
| `partitioner` | `iid` hoặc `dirichlet` |
| `dirichlet_alpha` | Mức non-IID khi dùng Dirichlet |
| `seed` | Seed để partition deterministic |
| `data_dir` | Nơi lưu/tải dataset |
| `output_dir` | Nơi lưu config/model artifacts |
| `device` | `cpu`, `cuda`, hoặc device PyTorch hợp lệ |
| `emnist_split` | Split EMNIST, ví dụ `balanced`, `letters`, `digits` |
