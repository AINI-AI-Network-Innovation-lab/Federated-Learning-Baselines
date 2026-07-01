# pFedMe Design

## Goal

Tích hợp thuật toán pFedMe từ paper "Personalized Federated Learning with Moreau Envelopes" vào framework hiện tại theo hướng bám paper nhất có thể, nhưng vẫn giữ nguyên:

- Flower app entrypoints
- registry/config pattern của repo
- evaluation pipeline hiện tại

## Scope

Included:

- thêm baseline mới `pfedme`
- custom server strategy với global update có hệ số `beta`
- client-side Moreau inner optimization cho personalized model
- local reference-model update theo công thức của paper
- lưu personalized model state theo client
- config/tests/docs cho baseline mới

Not included:

- personalized evaluation flow riêng
- thay đổi semantics của server eval hoặc client eval
- tái hiện đầy đủ phần convergence analysis của paper

## Paper Mapping

Paper mô tả:

- server gửi global model `w_t`
- client chạy `R` local rounds
- ở mỗi local round, client xấp xỉ personalized proximal solution `theta_tilde`
- client update local reference model:
  - `w_{i,r+1} = w_{i,r} - eta * lambda * (w_{i,r} - theta_tilde)`
- server aggregate các `w_{i,R}` từ client được sample:
  - `w_{t+1} = (1 - beta) * w_t + beta * avg_i(w_{i,R})`

Mapping sang repo:

- `local_epochs` map sang số local rounds `R`
- `learning_rate` map sang outer/reference update step size `eta`
- thêm `pfedme_personal_learning_rate` cho inner personalized optimizer
- thêm `pfedme_personal_steps` cho số inner proximal steps `K`
- thêm `pfedme_lambda` và `pfedme_beta`

## Chosen Architecture

### Server

Thêm `src/fl_baselines/algorithms/pfedme.py`:

- builder `PFedMeBuilder`
- strategy `PFedMeStrategy`
- strategy kế thừa `FedAvg` để reuse sampling/config/evaluate hooks
- override `aggregate_fit(...)` để:
  - aggregate weighted average các local reference model
  - trộn với global model trước đó bằng `pfedme_beta`
  - checkpoint model tương tự các baseline khác

### Client

Thêm nhánh `algorithm == "pfedme"` trong `TorchFlowerClient.fit(...)`:

- nhận global parameters
- clone một `reference_model` từ global model
- load personalized model state nếu đã tồn tại; nếu chưa có thì khởi tạo từ global model
- chạy local training bằng helper mới trong `training/pfedme.py`
- sau training:
  - copy reference model đã update về `self.model`
  - save personalized model state
  - return reference model parameters về server

### Training

Thêm `src/fl_baselines/training/pfedme.py`:

- chạy vòng lặp:
  - với mỗi local round
  - tối ưu personalized model trên objective:
    - `cross_entropy + (lambda / 2) * ||theta - reference||^2`
  - sau `K` inner steps, update reference parameters bằng:
    - `reference = reference - eta * lambda * (reference - theta)`
- trả metrics train cơ bản của local reference update path:
  - `train_loss`
  - `train_accuracy`

## State And Evaluation

- personalized state lưu ở:
  - `outputs/pfedme_clients/<client_id>/personalized.pt`
- evaluation giữ nguyên pipeline hiện tại:
  - server eval trên global/reference model
  - client eval trên model tham gia FL pipeline hiện tại
- vòng này không special-case personalized eval

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/pfedme.py`
- `src/fl_baselines/training/pfedme.py`
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
- `docs/algorithms/pfedme.md`

## Risks And Choices

### Personalized Eval

User explicitly chose to keep the current evaluation pipeline. Vì vậy implementation này sẽ bám paper ở train/aggregation, nhưng không đổi eval sang personalized model.

### Inner Solver Fidelity

Paper mô tả inner proximal subproblem được giải xấp xỉ. Trong repo này, ta sẽ dùng nhiều bước SGD trên objective proximal thay vì thêm solver mới phức tạp hơn. Đây là cách bám paper hợp lý và phù hợp pattern hiện tại của codebase.

### Model Support

Implementation phải không giả định kiến trúc đặc biệt ngoài việc model output logits cho classification, giống các baseline training loop hiện tại.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau đó xóa `__pycache__`.
