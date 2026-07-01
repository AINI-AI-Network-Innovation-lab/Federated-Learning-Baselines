# FedProto Design

## Goal

Tich hop thuat toan FedProto tu paper "FedProto: Federated Prototype Learning across Heterogeneous Clients" vao framework hien tai theo huong bam paper nhat co the, nhung van giu nguyen:

- Flower app entrypoints
- registry/config pattern cua repo
- classifier-based inference va evaluation pipeline hien tai
- kha nang so sanh performance voi cac baseline dang co

## Scope

Included:

- them baseline moi `fedproto`
- server-side prototype aggregation theo class
- client-side prototype regularization trong local training
- helper trich embedding ma khong doi public `forward()` contract cua model
- config/tests/docs cho baseline moi

Not included:

- doi tat ca model sang API `forward() -> (embedding, logits)`
- chuyen eval sang prototype-based prediction
- model-heterogeneous benchmark setup day du nhu trong paper
- luu prototype state xuyen round o client

## Paper Mapping

Paper mo ta:

- moi client tinh local prototypes bang mean embedding theo class
- server aggregate local prototypes thanh global prototypes theo class
- client toi uu:
  - classification loss
  - `+ lambda * distance(local_proto, global_proto)`
- paper co the infer bang prototype distance, va nhan manh tinh tuong thich voi model heterogeneity

Mapping sang repo:

- local model van train classifier head va van eval bang logits nhu cac baseline khac
- FedProto chi duoc dua vao nhu mot regularizer tren representation
- prototypes duoc trao doi qua server/client parameter payload thay vi scalar metrics
- local prototype cua class la mean embedding tren cac mau thuoc class do trong local train set cua round hien tai
- global prototype la sample-weighted average theo so luong mau cua tung class tu cac client duoc sample

## Chosen Architecture

### Server

Them `src/fl_baselines/algorithms/fedproto.py`:

- `FedProtoStrategy`
- `FedProtoBuilder`
- strategy ke thua `FedAvg` de reuse sampling, eval hooks, va checkpointing pattern

Server strategy se:

- giu `global_prototypes` trong memory
- gui `model_parameters + global_prototypes` cho client trong `configure_fit(...)`
- nhan `model_parameters + local_prototypes` tu clients trong `aggregate_fit(...)`
- aggregate model parameters bang FedAvg-style weighted average
- aggregate prototypes theo class bang weighted average dua tren sample count theo class

### Client

Them nhanh `algorithm == "fedproto"` trong `TorchFlowerClient.fit(...)`:

- load global model parameters
- load global prototypes tu payload cua server
- train local model bang helper moi trong `training/fedproto.py`
- tra local model parameters kem local prototypes ve server

Khong can persist state theo client vi FedProto khong can memory xuyen round ngoai model update thong thuong.

### Feature Extraction

Them helper moi `src/fl_baselines/training/features.py`:

- trich embedding penultimate representation cho cac model hien co
- khong doi `forward()` contract cua model
- support:
  - `mnist_cnn`
  - `lenet`
  - `resnet9`
  - torchvision `resnet18`/`resnet34`
  - `inception`

Muc tieu la de FedProto co the regularize tren embedding space ma khong lam rung cac baseline khac.

### Training

Them `src/fl_baselines/training/fedproto.py`:

- local training helper rieng cho FedProto
- objective moi minibatch:
  - `cross_entropy(logits, targets)`
  - `+ fedproto_lambda * prototype_regularization`
- local prototypes duoc cap nhat bang cach accumulate mean embedding theo class trong suot local training
- prototype regularization chi ap dung cho cac class co global prototype tu server
- helper tra:
  - train metrics
  - local prototypes da tinh
  - class counts tuong ung cho prototype aggregation

## State And Evaluation

- khong co client artifact state moi can persist
- evaluation giu nguyen pipeline hien tai:
  - server eval tren server-side test set
  - client eval tren held-out client test split cua tung client
- metric pipeline khong doi:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/fedproto.py`
- `src/fl_baselines/training/features.py`
- `src/fl_baselines/training/fedproto.py`
- `src/fl_baselines/clients/torch_client.py`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/extending-baselines.md`
- `docs/testing-and-artifacts.md`
- `docs/algorithms/index.md`
- `docs/algorithms/fedproto.md`

## Risks And Choices

### Paper Fidelity vs Repo Consistency

Paper nhan manh prototype-based communication va co the support model heterogeneity. User da chon giu model/eval pipeline hien tai, nen lan tich hop nay se giu dung prototype aggregation + regularization, nhung van de classifier head phuc vu suy luan va benchmark nhu cac baseline khac.

### Embedding Interface

Thay vi doi tat ca model sang tuple output, ta se them helper trich embedding theo model family. Cach nay it xam lan hon va an toan hon cho codebase hien tai.

### Prototype Payload

Proto la tensor, nen khong dua qua metrics scalar. Giong SCAFFOLD/FedDC, strategy se dong goi prototypes vao parameter payload de giao tiep server-client.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau do xoa `__pycache__`.
