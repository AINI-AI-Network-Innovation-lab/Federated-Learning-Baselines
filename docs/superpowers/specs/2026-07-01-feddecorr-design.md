# FedDecorr Integration Design

## Goal

Tich hop FEDDECORR vao codebase hien tai theo pha dau la mot baseline moi `feddecorr`, giu nguyen Flower pipeline va evaluation semantics, dong thoi tach local decorrelation regularizer thanh phan rieng de sau nay co the mo rong thanh add-on cho cac baseline khac.

Pham vi dot nay la `FedAvg + FEDDECORR` theo tinh than paper *Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning*. Khong mo rong ngay sang `FedProx + FEDDECORR`, `FedAvgM + FEDDECORR`, hay `MOON + FEDDECORR`.

## Fit With Current Codebase

FEDDECORR khong thay doi server aggregation va khong can client state xuyen round. Theo paper, no chi them mot regularization term trong local training:

- `loss = cross_entropy + beta * L_feddecorr`

Trong do `L_feddecorr` regularize correlation matrix cua representations.

Vi vay, FEDDECORR phu hop nhat voi nhom baseline:

- server path giong `fedavg`
- local training can custom objective nhu `fedntd`
- can truy cap representation features nhu `fedproto`

Do do, integration tot nhat voi codebase hien tai la:

- builder moi `feddecorr` dung `CheckpointingFedAvg`
- local training helper rieng
- tai su dung `training/features.py` de trich representations
- mot nhanh routing nho trong `TorchFlowerClient.fit`

## Proposed Architecture

### 1. Algorithm registration

Them builder moi:

- `src/fl_baselines/algorithms/feddecorr.py`

Builder nay:

- co `name = "feddecorr"`
- dung cung sampling/checkpoint/eval hooks nhu `fedavg`
- gui xuong client:
  - `algorithm`
  - `server_round`
  - `local_epochs`
  - `learning_rate`
  - `feddecorr_beta`

Khong can custom aggregate logic vi client van tra raw local model parameters va server van sample-weighted average nhu FedAvg.

### 2. Local decorrelation loss

Them file:

- `src/fl_baselines/training/feddecorr.py`

Module nay se chua:

- `feddecorr_loss(features: torch.Tensor) -> torch.Tensor`
- `train_feddecorr_client(...)`

`feddecorr_loss` bam paper:

1. Nhan batch representations `z` shape `(N, d)`.
2. Z-score normalize theo batch:
   - `z_hat = (z - mean) / std`
3. Uoc luong correlation matrix:
   - `K = (z_hat^T z_hat) / N`
4. Tinh:
   - `L_feddecorr = mean(K^2)`

Cong thuc nay tuong duong voi `||K||_F^2 / d^2` trong paper, va la implementation-friendly version duoc neu trong Appendix G.

`train_feddecorr_client(...)` se:

- trich features bang `extract_features(model, inputs)`
- tinh `ce_loss`
- tinh `decor_loss`
- toi uu:
  - `loss = ce_loss + feddecorr_beta * decor_loss`

Ngoai ra metrics local se bao gom:

- `train_loss`
- `train_accuracy`
- `feddecorr_loss`

### 3. Feature extraction reuse

FEDDECORR can representations, nen se tai su dung:

- `src/fl_baselines/training/features.py`

Khong thay doi `forward()` contract cong khai cua cac model. Neu model tra tuple, helper hien co tiep tuc xu ly; neu model chi tra logits, helper se tiep tuc fallback theo abstraction da co trong repo.

### 4. Client routing

Cap nhat:

- `src/fl_baselines/clients/torch_client.py`

Them nhanh:

- `algorithm == "feddecorr"` goi `_fit_feddecorr(...)`

Nhanh nay:

- nhan global model nhu FedAvg
- train local bang `train_feddecorr_client(...)`
- tra raw local model parameters nhu FedAvg

Khong can persisted state tren disk.

### 5. Config surface

Cap nhat:

- `src/fl_baselines/core/config.py`
- `pyproject.toml`

Them hyperparameter:

- `feddecorr-beta`

Default:

- `feddecorr-beta = 0.1`

Ly do: paper bao cao `beta = 0.1` la gia tri on dinh va gan nhu tot nhat khi khong co prior ve dataset.

Validation:

- `feddecorr-beta` phai khong am

Cho phep `beta = 0.0` de:

- boundary test de dang
- cho phep baseline fallback ve local CE-only train khi can debug

## Data Flow

Moi round:

1. Server strategy `feddecorr` sample clients va gui global model.
2. Client set global parameters vao local model.
3. Local training:
   - forward lay logits
   - trich features
   - tinh `ce_loss + beta * feddecorr_loss`
   - SGD update
4. Client gui local model parameters ve server.
5. Server aggregate bang weighted average nhu FedAvg.
6. Framework tiep tuc server eval va client eval theo semantics hien tai.

## Evaluation And Comparability

FEDDECORR phai giu nguyen semantics evaluation hien co cua repo de so sanh truc tiep duoc voi cac baseline khac:

- server eval tren server-side test set
- client eval tren held-out client test split
- metrics eval van gom `loss`, `accuracy`, `precision`, `recall`, `f1`

User da xac nhan uu tien model performance, nen implementation nay khong them metric fairness/compute/communication vao output mac dinh.

## Testing Plan

Can bo sung test truoc implementation:

1. `tests/test_registry.py`
   - `feddecorr` duoc register mac dinh

2. `tests/test_config.py`
   - parse duoc `feddecorr-beta`
   - reject gia tri am

3. `tests/test_model_and_algorithm.py`
   - `FedDecorrBuilder` tao duoc strategy va fit config dung
   - support cac model hien co
   - `feddecorr_loss` cho tensor feature hop le va huu han
   - local training cua FEDDECORR cap nhat tham so model
   - `TorchFlowerClient.fit` route dung sang `train_feddecorr_client`
   - boundary test voi `beta = 0.0`

Muc tieu la bao dam baseline nay thuc su them decorrelation regularization tren representation thay vi tro ve train loop mac dinh.

## Risks And Decisions

### Decision: baseline rieng o pha dau, helper tai su dung cho pha sau

Thay vi them FEDDECORR nhu mot config flag cho moi baseline ngay lap tuc, pha dau se them:

- `algorithm = "feddecorr"`

nhung local loss helper duoc tach rieng de sau nay mo rong sang:

- `fedprox + feddecorr`
- `fedavgm + feddecorr`
- `moon + feddecorr`

ma khong phai viet lai decorrelation core.

### Decision: tai su dung feature extraction abstraction hien co

Khong sua model forward contracts. Dieu nay giu codebase dong nhat va tranh lam vo cac baseline da co.

### Main risks

1. Feature extraction can nhat quan giua cac model; can bo test model compatibility ro rang.
2. Z-score normalization co the gap van de `std = 0` khi batch suy bien; implementation can epsilon on dinh.
3. Neu helper feature fallback qua logits cho mot so model, decorrelation van chay duoc nhung co the kem faithful hon representation penultimate; can document ro dieu nay.

## Files Expected To Change

- `src/fl_baselines/algorithms/feddecorr.py`
- `src/fl_baselines/training/feddecorr.py`
- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `pyproject.toml`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/algorithms/feddecorr.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/quickstart.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

## Success Criteria

- Chay duoc `algorithm="feddecorr"` tren Flower pipeline hien tai.
- Cung mot config dataset/model/eval, FEDDECORR co ket qua so sanh truc tiep duoc voi baseline khac.
- Representation decorrelation regularization duoc ap dung trong local training ma khong doi evaluation pipeline.
- Full test suite va compile checks pass sau khi tich hop.
