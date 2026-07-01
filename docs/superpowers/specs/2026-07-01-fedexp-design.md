# FedExP Design

## Goal

Tich hop thuat toan FedExP tu paper "FedExP: Speeding up Federated Averaging via Extrapolation" vao framework hien tai theo huong bam paper o phan cot loi cua thuat toan, nhung van giu nguyen:

- Flower app entrypoints
- registry/config pattern cua repo
- local client training loop kieu FedAvg
- evaluation pipeline hien tai de so sanh performance cong bang voi cac baseline dang co

## Scope

Included:

- them baseline moi `fedexp`
- server-side adaptive extrapolation step size theo pseudo-gradient cua tung round
- config/tests/docs cho baseline moi

Not included:

- doi local objective hoac them client-side state
- average last two global iterates nhu paper de xac dinh final model
- bien the `FedExP-M` hoac `SCAFFOLD-ExP`
- thay doi evaluation semantics

## Paper Mapping

Paper mo ta FedExP nhu mot bien the cua generalized FedAvg:

- client nhan global model va train local SGD nhu FedAvg
- client gui local update/pseudo-gradient ve server
- server tinh:
  - `Delta_i = w_t - w_i`
  - `Delta_bar = average(Delta_i)`
  - `eta_g(t) = max(1, sum_i ||Delta_i||^2 / (2 * K * (||Delta_bar||^2 + epsilon)))`
- server update:
  - `w_{t+1} = w_t - eta_g(t) * Delta_bar`

Mapping sang repo:

- client van train y het FedAvg va chi tra model parameters
- server strategy suy ra `Delta_i` tu global parameters hien tai va local model parameters client gui ve
- aggregation van sample-weighted theo so luong mau cua client, de nhat quan voi FedAvg pattern trong codebase
- `K` se la so client tham gia round hien tai, phu hop voi partial participation ma paper cho phep

## Chosen Architecture

### Server

Them `src/fl_baselines/algorithms/fedexp.py`:

- `FedExPStrategy`
- `FedExPBuilder`

Strategy se ke thua `FedAvg` de reuse:

- client sampling
- evaluation hook
- fit/eval metrics aggregation
- checkpointing pattern

Trong `aggregate_fit(...)`, strategy se:

- doc current global parameters tu `self.initial_parameters` o round dau va state noi bo cac round sau
- convert local client parameters thanh `Delta_i`
- tinh `Delta_bar` bang weighted average theo `num_examples`
- tinh `eta_g(t)` tu trung binh cac binh phuong norm cua `Delta_i` va norm cua `Delta_bar`
- update global parameters bang cong thuc FedExP
- luu `eta_g(t)` vao metrics/checkpoint state neu can theo doi

### Client

Khong can training helper moi.

`TorchFlowerClient.fit(...)` se di theo nhanh mac dinh nhu `fedavg`:

- set global model parameters
- train local model bang `train_one_client(...)`
- tra local model parameters ve server

Dieu nay giu FedExP dung voi paper o cho local optimization khong doi, va giam toi da do xam lan vao codebase.

## State And Evaluation

- khong co client artifact state moi can persist
- khong doi server eval tren server-side test set
- khong doi client eval tren held-out client test split
- metric pipeline giu nguyen:
  - `loss`
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`

## Config

Them config moi:

- `fedexp-epsilon`

Y nghia:

- so duong nho cong vao mau so de tranh `eta_g(t)` blow up khi `||Delta_bar||^2` rat nho

Validation:

- `fedexp-epsilon` phai non-negative

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/fedexp.py`
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
- `docs/algorithms/fedexp.md`

## Risks And Choices

### Paper Fidelity vs Repo Consistency

Paper co de cap them average last two iterates de giam dao dong loss. De giu pipeline eval hien tai dong nhat giua cac baseline, lan tich hop nay khong thay doi cach xac dinh model duoc evaluate; no chi tich hop adaptive extrapolation step dung phan cot loi cua FedExP.

### Weighted Aggregation Detail

Paper trinh bay trung binh tren client tham gia. Trong repo hien tai, cac strategy deu aggregate theo `num_examples`, nen FedExP se di theo convention nay de so sanh performance giua baseline duoc cong bang trong cung pipeline.

### Scope Discipline

FedExP-M va SCAFFOLD-ExP duoc paper appendix de cap, nhung khong nam trong pham vi lan nay vi user muon chi tich hop algo vao pipeline hien tai.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau do xoa `__pycache__`.
