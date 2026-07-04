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
7. Nếu strategy có hook `set_proxy_loader(...)`, server app sẽ inject luôn `server_loader` cho các baseline can proxy data tren server nhu FedLAW.
8. Trả về `ServerAppComponents(strategy=..., config=...)`.

Server evaluation luôn dùng server-side test split qua `build_server_loader(...)`.
Metric trả về từ evaluation gồm `accuracy`, `precision`, `recall`, và `f1`, trong đó `precision`/`recall`/`f1` dùng macro averaging cho multi-class classification.

## Luồng Chạy Client

`src/fl_baselines/app/client_app.py` là entrypoint client của Flower:

1. Gọi `register_default_components()`.
2. Parse config.
3. Xác định `partition_id` và `num_partitions`.
4. Lấy dataset/model builder từ registry.
5. Tạo dataloader cho client partition.
6. Tạo model riêng cho client.
7. Wrap bằng `TorchFlowerClient`.

Client-side evaluation dùng held-out client test split được tách từ chính local partition train của client theo `client-test-fraction`.
Client eval dùng cùng evaluation helper với server eval, nên cũng report `accuracy`, macro `precision`, macro `recall`, và macro `f1`.

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
Unknown algorithm 'custom'. Available: ditto, fedadp, fedavg, fedavgm, feddc, feddyn, fedexp, fedntd, fednova, fedper, fedproto, fedprox, fedrep, moon, pfedme, scaffold
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
| `client_test_fraction` | Tỷ lệ dữ liệu local của mỗi client được hold out cho client test eval |
| `local_epochs` | Số epoch local training |
| `batch_size` | Batch size cho dataloader |
| `learning_rate` | Learning rate local optimizer |
| `proximal_mu` | Hệ số proximal term cho FedProx |
| `moon_mu` | Hệ số model-contrastive loss cho MOON |
| `moon_temperature` | Temperature cho contrastive logits của MOON |
| `server_learning_rate` | Learning rate của server optimizer cho FedAvgM |
| `server_momentum` | Momentum của server optimizer cho FedAvgM |
| `fedadp_alpha` | Hằng số alpha cho Gompertz mapping trong FedAdp |
| `feddyn_alpha` | Hệ số dynamic regularization cho FedDyn |
| `feddc_alpha` | Hệ số drift penalty trong FedDC |
| `fedexp_epsilon` | Hằng số epsilon ổn định mẫu số khi FedExP tính adaptive server extrapolation step |
| `fedlaw_server_epochs` | Số epoch server-side optimization trên proxy loader cho FedLAW |
| `fedlaw_server_learning_rate` | Learning rate của optimizer học `lambda` và `gamma` trong FedLAW |
| `fedlaw_gamma_init` | Giá trị khởi tạo cho shrinking factor `gamma` của FedLAW |
| `gamf_sigma` | Hệ số sigma cho độ tương đồng second-order của GAMF |
| `gamf_initial_tau` | Nhiệt độ Sinkhorn khởi tạo của GAMF |
| `gamf_descent_factor` | Hệ số giảm nhiệt độ sau mỗi vòng GAMF |
| `gamf_min_tau` | Ngưỡng nhiệt độ tối thiểu trước khi dừng annealing của GAMF |
| `gamf_max_iters` | Số vòng lặp graph-matching tối đa của GAMF |
| `fedma_matching_epsilon` | Ngưỡng matching cost cho FedMA; bản tích hợp hiện dùng fixed-width matching và giữ field này để cấu hình/so sánh |
| `fedgen_alpha` | Hệ số scale alpha cho cập nhật feature mask của FedGEN |
| `fedgen_lambda` | Hệ số regularization penalty của FedGEN |
| `fedproto_lambda` | Hệ số regularization kéo local prototypes về global prototypes trong FedProto |
| `fedntd_beta` | Hệ số not-true distillation loss trong FedNTD |
| `fedntd_temperature` | Temperature dùng cho not-true softmax trong FedNTD |
| `ditto_lambda` | Hệ số regularization kéo personalized model về global model trong Ditto |
| `pfedme_lambda` | Hệ số proximal regularization giữa personalized model và reference model trong pFedMe |
| `pfedme_beta` | Hệ số server-side mixing của pFedMe |
| `pfedme_personal_learning_rate` | Learning rate cho inner personalized optimization của pFedMe |
| `pfedme_personal_steps` | Số bước inner personalized update trong mỗi local step của pFedMe |
| `fednova_server_momentum` | Momentum phía server cho FedNova normalized updates |
| `fedper_personal_layers` | Số module cuối có tham số được giữ local cho FedPer |
| `fedrep_personal_layers` | Số module cuối có tham số được giữ local cho FedRep |
| `fedrep_representation_epochs` | Số epoch train representation sau pha train head của FedRep |
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
