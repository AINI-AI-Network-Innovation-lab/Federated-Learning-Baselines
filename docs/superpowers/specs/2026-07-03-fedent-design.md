# FedEnt Integration Design

## Goal

Tich hop `FedEnt` vao codebase hien tai nhu mot baseline moi `fedent`, giu nguyen Flower pipeline va evaluation semantics, dong thoi giu trung tam cua paper *Adaptive Federated Learning via New Entropy Approach* la adaptive learning rate theo entropy + mean-field.

Pham vi dot nay la:

- `FedAvg-style server aggregation + FedEnt adaptive local learning rate`
- fit voi runtime Flower hien tai
- faithful voi paper o cap do cong thuc va round-level behavior

Khong theo duoi pha dau:

- faithful tuyet doi voi mot pha precompute toan horizon `T` truoc khi train
- mo rong ngay sang `FedProx + FedEnt`, `FedAvgM + FedEnt`, hay `FedDyn + FedEnt`
- reproduce day du cac thiet lap thuc nghiem ngoai repo hien tai

## Fit With Current Codebase

Theo paper, FedEnt khong thay doi quy tac global aggregation co ban. Phan thay doi chinh la:

- server giu va cap nhat mean-field state `phi1`, `phi2`
- client tinh adaptive learning rate `eta_i(t)` moi round
- local SGD dung learning rate dong do thay vi learning rate tinh

Vi vay, FedEnt phu hop nhat voi nhom baseline:

- server path giong `fedavg`
- local behavior can custom objective/update nhu `feddyn`, `fedntd`, `fedproto`
- can round-level shared state tu server gui xuong client

Do do, integration tot nhat voi codebase hien tai la:

- builder moi `fedent` dung strategy rieng ke thua `FedAvg`
- local training helper rieng `train_fedent_client(...)`
- mot nhanh routing nho trong `TorchFlowerClient.fit`
- mean-field state duoc quan ly boi strategy thay vi app entrypoints

## Proposed Architecture

### 1. Algorithm registration

Them builder moi:

- `src/fl_baselines/algorithms/fedent.py`

Builder nay:

- co `name = "fedent"`
- tai su dung sampling/checkpoint/eval hooks nhu `fedavg`
- tao `FedEntStrategy`

`FedEntStrategy` van aggregate model parameters bang weighted average kieu `FedAvg`, nhung quan ly them:

- `phi1`: round-level mean-field estimate cho global weight
- `phi2`: round-level mean-field estimate cho weighted squared norm
- `previous_eta` theo client de ho tro decay Eq. (31) theo tinh than paper
- optional round artifacts phuc vu debug

### 2. Local FedEnt training helper

Them file:

- `src/fl_baselines/training/fedent.py`

Module nay se chua:

- helper tinh entropy-based adaptive learning rate
- helper decay learning rate theo Eq. (31)
- `train_fedent_client(...)`

Phan local train loop se:

1. Nhan `phi1`, `phi2`, `beta`, `gamma`, learning-rate base, va current model
2. Uoc luong local gradient tren model nhan tu server
3. Tinh `eta_i(t)` theo cong thuc FedEnt o dang implementation-friendly
4. Apply decay:
   - `eta_i(t) = gamma * eta_i(t-1) + (1 - gamma) * eta_i_raw(t)`
5. Dung `eta_i(t)` de train `E` local epochs bang SGD

Metrics local can bao gom:

- `train_loss`
- `train_accuracy`
- `fedent_learning_rate`
- `fedent_phi2`

### 3. Client routing

Cap nhat:

- `src/fl_baselines/clients/torch_client.py`

Them nhanh:

- `algorithm == "fedent"` goi `_fit_fedent(...)`

Nhanh nay:

- nhan global model nhu FedAvg
- doc `phi1`, `phi2`, `fedent-beta`, `fedent-gamma` tu fit config
- train local bang `train_fedent_client(...)`
- tra raw local model parameters nhu FedAvg

Ngoai model parameters, client can tra them metadata de server update mean-field state cho round sau:

- `fedent_weight_sq_norm`
- `fedent_learning_rate`

Khong can persisted state lon tren disk. `previous_eta` se duoc luu theo `client_id` trong `output-dir/fedent_clients/<client-id>/state.pt` khi `fedent-enable-decay = true`, de giu pattern repo nhat quan va ho tro round sau trong simulation.

### 4. Server-side mean-field management

`FedEntStrategy` la noi duy nhat giu logic mean-field cua baseline.

Moi round, strategy se:

1. Chuan bi `phi1(t)` va `phi2(t)` cho fit config
2. Gui `phi1`, `phi2` cung hyperparameters FedEnt xuong client
3. Nhan local parameters va `fedent_weight_sq_norm` tu clients
4. Aggregate model nhu FedAvg
5. Cap nhat:
   - `phi1(t+1)` tu weighted average cua local model round hien tai
   - `phi2(t+1)` tu weighted average cua squared norms round hien tai

Nhu vay, phan adaptation nam trong strategy/client, khong can sua `server_app.py` hay `client_app.py`.

## Mean-Field And Fixed-Point Design

### Faithful parts

Phan faithful voi paper duoc giu lai:

- van co hai dai luong `phi1(t)` va `phi2(t)`
- van tinh `eta_i(t)` dua tren `phi1(t)`, `phi2(t)` va local gradient
- van co decay `gamma` theo Eq. (31)
- van xem mean-field la shared round state do server quan ly

### Adapted parts

Paper mo ta mot pha iterative estimator calculator qua toan bo horizon `T`. Repo hien tai lai chay theo online Flower rounds, nen khong co san mot offline precompute phase tach biet. Vi vay implementation se dung:

#### 1. Warm-start online fixed-point

Dau moi round:

- `phi1(t)` duoc warm-start bang global model hien tai
- `phi2(t)` duoc warm-start bang weighted squared norm cua global model hien tai

#### 2. One-round-lag mean-field update

Sau khi thu local updates cua round `t`, strategy update:

- `phi1(t+1)` bang weighted average cua local model parameters
- `phi2(t+1)` bang weighted average cua `||w_i(t+1)||^2`

Cach nay bam dung Definition 2 cua paper o cap do quan sat round-thuc-te, khong can giai toan trajectory tu truoc.

#### 3. Optional inner refinement

Them config:

- `fedent-fixed-point-steps`

Neu gia tri > 1, strategy co the refine noi bo `phi1`, `phi2` mot vai lan nho truoc khi phat fit config round tiep theo. Muc tieu la giu tinh than Algorithm 1 ma van khop runtime online.

#### 4. Numerical safety

Cong thuc Eq. (15) nhay cam voi chia mau so va `log`, nen implementation phai co:

- epsilon cho `phi2`
- clip `p_i` vao `(eps, 1.0]`
- clamp learning rate ve mien khong am
- optional `fedent-max-learning-rate`
- fallback ve base learning rate neu cong thuc sinh `nan`/`inf`

## FedEnt Learning Rate Design In Repo

Implementation se dung ban implementation-friendly cua FedEnt:

1. Flatten model parameters thanh vector `w`
2. Flatten gradient local thanh vector `g`
3. Dung `phi1` va `phi2` o dang vector/scalar
4. Tinh:
   - weighted norm share `p_i`
   - entropy-related log factor
   - raw adaptive learning rate dua tren Eq. (15)/(22)
5. Apply decay Eq. (31)
6. Clip vao khoang an toan

Quan diem quan trong:

- paper su dung bieu dien toan hoc theo tham so model, nen viec flatten vector la faithful o cap implementation
- repo dang train bang SGD tren `nn.Module`, nen `eta_i(t)` se duoc map thanh optimizer learning rate cho local loop

## Data Flow

Moi round:

1. Server strategy `fedent` sample clients va gui global model.
2. Strategy gui kem:
   - `algorithm`
   - `server_round`
   - `local_epochs`
   - `learning_rate`
   - `fedent_beta`
   - `fedent_gamma`
   - `fedent_epsilon`
   - `fedent_max_learning_rate`
   - serialized `fedent_phi1`
   - scalar `fedent_phi2`
3. Client set global parameters vao local model.
4. Client uoc luong local gradient va tinh `eta_i(t)`.
5. Client local train `E` epochs bang adaptive learning rate.
6. Client gui local model parameters ve server kem metric/state toi thieu.
7. Server aggregate bang weighted average nhu FedAvg.
8. Strategy update mean-field state cho round sau.
9. Server eval va client eval giu semantics hien tai.

## Config Surface

Cap nhat:

- `src/fl_baselines/core/config.py`
- `pyproject.toml`

Them cac hyperparameter:

- `fedent-beta`
- `fedent-gamma`
- `fedent-epsilon`
- `fedent-fixed-point-steps`
- `fedent-max-learning-rate`
- `fedent-enable-decay`

Gia tri mac dinh de xuat:

- `fedent-beta = 0.99`
- `fedent-gamma = 0.99`
- `fedent-epsilon = 1e-8`
- `fedent-fixed-point-steps = 1`
- `fedent-max-learning-rate = 1.0`
- `fedent-enable-decay = true`

Validation:

- `fedent-beta` phai nam trong `(0, 1)`
- `fedent-gamma` phai nam trong `[0, 1)`
- `fedent-epsilon` phai duong
- `fedent-fixed-point-steps` phai duong
- `fedent-max-learning-rate` phai duong

Ly do:

- `beta` va `gamma` gan paper
- `epsilon` va clamp can thiet cho implementation on dinh
- `fixed-point-steps` cho phep ablation giua ban online-toi-gian va ban refine hon

## Evaluation And Comparability

FedEnt phai giu nguyen semantics evaluation hien co cua repo de so sanh truc tiep duoc voi cac baseline khac:

- server eval tren server-side test set
- client eval tren held-out client test split
- metrics eval van gom `loss`, `accuracy`, `precision`, `recall`, `f1`

FedEnt thay doi learning dynamic, khong thay doi evaluation contract.

## Testing Plan

Can bo sung test truoc implementation:

1. `tests/test_registry.py`
   - `fedent` duoc register mac dinh

2. `tests/test_config.py`
   - parse duoc toan bo config FedEnt
   - reject gia tri ngoai mien hop le

3. `tests/test_model_and_algorithm.py`
   - `FedEntBuilder` tao duoc strategy va fit config dung
   - fit config co du `phi1`, `phi2`, `fedent_beta`, `fedent_gamma`
   - helper tinh learning rate tra gia tri huu han
   - helper respect decay/clamp
   - local training cua FedEnt cap nhat tham so model
   - boundary test cho `phi2` rat nho
   - route dung tu `TorchFlowerClient.fit`

4. Smoke checks cho model compatibility
   - `mnist_cnn`
   - `lenet`
   - `resnet9`
   - `resnet18`
   - `resnet34`
   - `inception`

Muc tieu la bao dam baseline nay thuc su them adaptive local learning rate theo FedEnt thay vi tro ve train loop mac dinh.

## Risks And Decisions

### Decision: strategy rieng thay vi reuse thuan `FedAvgBuilder`

Ly do:

- can server-side mean-field state
- can fit config round-level phuc tap hon
- can giu baseline boundary ro rang

### Decision: online fixed-point thay vi offline horizon solver

Ly do:

- khop Flower runtime hien tai
- tranh them pha precomputation ngoai vong doi app
- van faithful voi paper o cap do round-level adaptation

### Decision: persist client eta state neu decay duoc bat

Ly do:

- Eq. (31) can `eta_i(t-1)`
- repo da co pattern persist client-specific state cho `moon`, `feddyn`, `pfedme`, `fedper`, `fedrep`

### Main risks

1. Serialized `phi1` co the lon vi bang kich thuoc model parameters.
2. Eq. (15) co the khong on dinh so voi mot so model/dataset neu khong clamp ky.
3. Dinh nghia gradient trong paper dua tren global iteration-level update, con repo train theo local minibatch epochs; can document ro implementation bridge nay.
4. Neu luu `eta` state theo client, can dam bao no khong gay xung dot voi dirty outputs hay test isolation.

## Files Expected To Change

- `src/fl_baselines/algorithms/fedent.py`
- `src/fl_baselines/training/fedent.py`
- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `pyproject.toml`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/algorithms/fedent.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

## Success Criteria

- Chay duoc `algorithm="fedent"` tren Flower pipeline hien tai.
- Strategy van aggregate kieu FedAvg nhung local learning rate thay doi theo FedEnt.
- Mean-field state `phi1`, `phi2` duoc cap nhat xuyen rounds ma khong sua app entrypoints.
- Evaluation pipeline va metrics hien tai khong bi thay doi.
- Test config/registry/builder/local-training/model-compatibility pass.
- Full test suite va compile checks pass sau khi tich hop.
