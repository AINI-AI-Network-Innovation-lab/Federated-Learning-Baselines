# FedDC Design

## Goal

Tich hop thuat toan FedDC tu paper "FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction" vao framework hien tai theo huong bam paper nhat co the, nhung van giu nguyen:

- Flower app entrypoints
- registry/config pattern cua repo
- evaluation pipeline hien tai
- kha nang chay voi partial participation

## Scope

Included:

- them baseline moi `feddc`
- custom server strategy giu server-side average update state `g`
- local training voi drift penalty va gradient correction
- persistent client state cho `h_i` va `g_i`
- corrected upload `theta_i + h_i`
- config/tests/docs cho baseline moi

Not included:

- thay doi semantics cua server eval hoac client eval
- personalized evaluation flow
- them benchmark-specific scheduler, momentum decay, hay augmentation theo setup paper

## Paper Mapping

Paper mo ta:

- moi client co local drift variable `h_i`
- local objective:
  - local empirical loss
  - `+ (alpha / 2) * ||h_i + theta_i - w||^2`
  - `+ <theta_i, g_i - g> / (eta * K)`
- sau local optimization:
  - `h_i <- h_i + (theta_i^+ - theta_i)`
- truoc upload:
  - client gui corrected parameter `theta_i^+ + h_i^+`
- server aggregate corrected parameters de cap nhat global model
- `g_i` va `g` theo doi local/global update tu round truoc

Mapping sang repo:

- `learning_rate` map sang `eta`
- `local_epochs * len(train_loader)` map sang tong so local iterations `K`
- them `feddc_alpha` cho he so drift penalty
- moi client persist:
  - `h_i`
  - `g_i`
- server strategy persist trong memory:
  - current global parameters
  - average update state `g`

## Chosen Architecture

### Server

Them `src/fl_baselines/algorithms/feddc.py`:

- `FedDCStrategy`
- `FedDCBuilder`
- strategy ke thua `FedAvg` de reuse sampling, eval hooks, va checkpointing pattern

Server strategy se:

- truyen `feddc_alpha` va `num_total_clients` vao fit config
- nhan corrected local parameters tu clients
- aggregate bang sample-weighted average theo convention hien tai cua repo
- cap nhat server state `g` tu cac local update state `g_i` duoc client gui qua metrics
- dat global model moi bang corrected average

### Client

Them nhanh `algorithm == "feddc"` trong `TorchFlowerClient.fit(...)`:

- load global parameters vao `self.model`
- load persisted client state `h_i`, `g_i`; neu chua co thi khoi tao zero tensors
- clone global model snapshot de lam moc drift/correction cho round hien tai
- train local model bang helper moi trong `training/feddc.py`
- update:
  - `h_i`
  - `g_i = theta_i^+ - theta_i`
- persist client state
- return corrected upload parameters `theta_i^+ + h_i^+`

### Training

Them `src/fl_baselines/training/feddc.py`:

- helper local training rieng cho FedDC
- objective moi minibatch:
  - `cross_entropy`
  - `+ (feddc_alpha / 2) * ||h_i + theta - w||^2`
  - `+ <theta, g_i - g> / (learning_rate * total_local_steps)`
- helper cap nhat drift state:
  - `h_i <- h_i + (theta_i^+ - theta_i)`

## State And Evaluation

- client state luu tai:
  - `outputs/feddc_clients/<client-id>/state.pt`
- state file chua:
  - `drift`
  - `local_update`
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
- `src/fl_baselines/algorithms/feddc.py`
- `src/fl_baselines/training/feddc.py`
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
- `docs/algorithms/feddc.md`

## Risks And Choices

### Paper Fidelity vs Repo Consistency

Paper viet cong thuc voi full-participation notation, nhung user muon baseline van dong bo voi pipeline hien tai. Vi vay implementation nay se giu dung local objective va drift correction, nhung adapt state handling de partial participation van an toan va deterministically reuse state theo client.

### Server Update State

Paper su dung `g_i` va `g` theo local/global update round truoc. Trong repo nay, `g_i` se duoc client tinh tu delta model cua round vua train, con `g` se duoc server cap nhat tu sample-weighted average cua cac `g_i` nhan duoc o round hien tai de phuc vu round tiep theo.

### Model Support

Implementation phai hoat dong voi cac model hien tai, mien la training/evaluation path cuoi cung tra ra logits classification. FedDC khong duoc gia dinh mot kieu kien truc dac biet nao.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau do xoa `__pycache__`.
