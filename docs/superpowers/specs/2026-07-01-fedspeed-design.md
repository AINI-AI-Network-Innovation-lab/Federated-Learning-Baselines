# FedSpeed Integration Design

## Goal

Tich hop FedSpeed vao codebase hien tai nhu mot baseline moi, uu tien bam paper *FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy* nhat co the, trong khi van giu nguyen Flower pipeline, evaluation semantics, registry contracts, va kha nang so sanh performance truc tiep voi cac baseline hien co.

Pham vi chi bao gom FedSpeed core algorithm. Khong mo rong sang learning-rate scheduler, FedADMM, hay cac setup benchmark ngoai pipeline hien tai cua repo.

## Fit With Current Codebase

FedSpeed khong phai baseline chi doi server aggregation. Theo paper, no can:

- local training objective rieng
- client state xuyen round `g_hat_i`
- post-processed payload gui server `x_hat_i = x_i,K - lambda * g_hat_i`

Vi vay, FedSpeed gan nhat voi nhom baseline nhu `feddyn`, `feddc`, `ditto`, `pfedme` o cho can local state duoc persist theo client. Tuy nhien, server-side aggregation cua FedSpeed van don gian hon `feddc` vi server chi average cac payload client gui len nhu FedAvg.

## Proposed Architecture

### 1. Algorithm registration

Them builder moi:

- `src/fl_baselines/algorithms/fedspeed.py`

Builder nay:

- co `name = "fedspeed"`
- dung strategy kieu `CheckpointingFedAvg`
- gui xuong client cac config:
  - `algorithm`
  - `server_round`
  - `local_epochs`
  - `learning_rate`
  - `fedspeed_lambda`
  - `fedspeed_alpha`
  - `fedspeed_rho`

Server khong can custom aggregate logic phuc tap, vi paper aggregate bang trung binh cac amended client models `x_hat_i`.

### 2. Local training module

Them file:

- `src/fl_baselines/training/fedspeed.py`

Module nay implement local loop bam paper:

1. Client nhan global model `x_t` va dat `x_{i,0}^t = x_t`.
2. O moi minibatch:
   - tinh `g1 = grad F_i(x_{i,k}^t; batch)`
   - tao extra-step point `x_dot = x_{i,k}^t + rho * g1`
   - tinh `g2 = grad F_i(x_dot; same batch)`
   - tinh quasi-gradient `g_tilde = (1 - alpha) * g1 + alpha * g2`
   - cap nhat local model theo:
     - `x_{i,k+1}^t = x_{i,k}^t - lr * (g_tilde - g_hat_prev + (1 / lambda) * (x_{i,k}^t - x_t))`
3. Sau local training:
   - cap nhat `g_hat_i^t = g_hat_i^{t-1} - (1 / lambda) * (x_{i,K}^t - x_t)`
   - tinh payload gui server:
     - `x_hat_i = x_{i,K}^t - lambda * g_hat_i^t`

Implementation se dung cung minibatch cho `g1` va `g2`, vi day la chi tiet paper neu ro trong Algorithm 1.

### 3. Client routing and persisted state

Cap nhat:

- `src/fl_baselines/clients/torch_client.py`

Them nhanh:

- `algorithm == "fedspeed"` goi `_fit_fedspeed(...)`

Nhanh nay se:

- set global parameters vao local model
- load `g_hat` state cua client tu `output-dir`
- goi `train_fedspeed_client(...)`
- save `g_hat` moi
- tra payload `x_hat` len server

Persisted state path de xuat:

- `outputs/fedspeed_clients/<client-id>/state.pt`

State chi can luu:

- `g_hat`: list tensor cung shape voi model parameters

Khong can them server-side state rieng.

### 4. Config surface

Cap nhat:

- `src/fl_baselines/core/config.py`
- `pyproject.toml`

Them hyperparameters:

- `fedspeed-lambda`
- `fedspeed-alpha`
- `fedspeed-rho`

Default de xuat:

- `fedspeed-lambda = 0.1`
- `fedspeed-alpha = 1.0`
- `fedspeed-rho = 0.1`

Ly do:

- paper dung prox weight `0.1` o setting 10% participation
- Algorithm 1 dung quasi-gradient weighted boi `alpha`; phan ablation trong paper bao cao `alpha = 1` cho `rho`
- paper report `rho0` tot quanh `0.1` trong ablation

Validation:

- `fedspeed-lambda` phai duong
- `fedspeed-alpha` phai nam trong `[0, 1]`
- `fedspeed-rho` phai khong am

## Data Flow

Moi round:

1. Server strategy `fedspeed` sample clients va gui global model + fit config.
2. Client load local state `g_hat_prev` neu co, neu chua co thi khoi tao zeros.
3. Client train local bang FedSpeed loop voi quasi-gradient perturbation va prox-correction.
4. Client cap nhat local state `g_hat`.
5. Client gui amended model `x_hat` ve server.
6. Server aggregate sample-weighted average tren cac `x_hat`.
7. Framework tiep tuc server eval va client eval theo semantics hien tai.

## Evaluation And Comparability

FedSpeed phai giu nguyen semantics evaluation hien co cua repo de ket qua so sanh duoc voi cac baseline khac khi dung cung file config:

- server eval tren server-side test set
- client eval tren held-out test split cua tung client
- metrics gom `loss`, `accuracy`, `precision`, `recall`, `f1`

User da xac nhan chi quan tam model performance, nen implementation nay khong them metric communication, fairness, hay compute-time.

## Testing Plan

Can bo sung test truoc implementation:

1. `tests/test_registry.py`
   - `fedspeed` duoc register mac dinh

2. `tests/test_config.py`
   - parse duoc `fedspeed-lambda`, `fedspeed-alpha`, `fedspeed-rho`
   - reject:
     - `lambda <= 0`
     - `alpha < 0` hoac `alpha > 1`
     - `rho < 0`

3. `tests/test_model_and_algorithm.py`
   - `FedSpeedBuilder` tao duoc strategy va fit config dung
   - support cac model hien co
   - local training cua FedSpeed cap nhat tham so model
   - `TorchFlowerClient.fit` route dung sang FedSpeed trainer
   - client persist va reuse `g_hat` state
   - payload tra ve tu client co dung so parameter arrays cua model

Muc tieu test la bao dam FedSpeed thuc su dung client state rieng, khong bi roi ve local SGD/FedProx/FedSAM mac dinh.

## Risks And Decisions

### Decision: faithful-core paper integration

Implementation se giu:

- prox-term trong local update
- prox-correction state `g_hat`
- quasi-gradient tu `g1/g2`
- amended client payload `x_hat`

Day la phan cot loi de baseline van con la FedSpeed thay vi mot bien the gan giong FedProx hoac FedSAM.

### Decision: giu server path don gian

Mac du paper phan tich duoc duoi dang prox-based method phuc tap hon, trong codebase nay server van co the giu nhu FedAvg aggregation tren payload amended tu client. Dieu nay vua bam paper, vua giu app entrypoints khong phai sua.

### Main risks

1. `g_hat` la client state xuyen round nen neu persist/load sai, baseline se bi sai hanh vi.
2. Payload client gui server khong phai raw local model, nen test route va payload shape phai ro rang.
3. Cong thuc paper dung local iteration theo tham so toan cuc `x_t`; can can than khong de `x_t` bi mutate trong local loop.

## Files Expected To Change

- `src/fl_baselines/algorithms/fedspeed.py`
- `src/fl_baselines/training/fedspeed.py`
- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `pyproject.toml`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/algorithms/fedspeed.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/quickstart.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

## Success Criteria

- Chay duoc `algorithm="fedspeed"` tren Flower pipeline hien tai.
- Cung mot config dataset/model/eval, FedSpeed co ket qua so sanh truc tiep duoc voi baseline khac.
- Local client state `g_hat` duoc persist va reuse dung theo client.
- Full test suite va compile checks pass sau khi tich hop.
