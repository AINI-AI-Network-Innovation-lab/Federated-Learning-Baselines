# FedSAM Integration Design

## Goal

Tich hop FedSAM vao codebase hien tai nhu mot baseline moi, giu nguyen federation pipeline, evaluation semantics, dataset/model registry, va cach so sanh performance dang duoc su dung cho cac baseline khac.

Pham vi chi bao gom FedSAM trong paper *Generalized Federated Learning via Sharpness Aware Minimization* (ICML 2022). Khong mo rong sang MoFedSAM o dot nay.

## Fit With Current Codebase

Codebase hien tai chia baseline thanh hai nhom:

- baseline chi can custom server strategy, vi du `fedexp`
- baseline can custom client/local training, vi du `feddyn`, `feddc`, `fedntd`, `fedproto`

FedSAM thuoc nhom thu hai. Theo paper, server van aggregate kieu FedAvg, con khac biet nam o local optimizer: client khong toi uu ERM bang SGD thuong ma dung SAM hai buoc tren cung minibatch.

Vi vay, integration phu hop nhat voi codebase hien tai la:

- strategy moi `fedsam` duoc xay dung tren `CheckpointingFedAvg`
- training module rieng cho local SAM updates
- mot nhanh routing nho trong `TorchFlowerClient.fit`

## Proposed Architecture

### 1. Algorithm registration

Them builder moi:

- `src/fl_baselines/algorithms/fedsam.py`

Builder nay:

- co `name = "fedsam"`
- dung cung sampling/checkpoint/eval hooks nhu `fedavg`
- gui `algorithm`, `server_round`, `local_epochs`, `learning_rate`, `fedsam_rho` xuong client qua `on_fit_config_fn`

Khong can custom aggregate logic vi paper FedSAM van average local model updates nhu FedAvg.

### 2. Local training module

Them file:

- `src/fl_baselines/training/fedsam.py`

Module nay se implement local SAM optimizer theo paper:

1. Tinh gradient tai tham so hien tai.
2. Tao perturbation:
   - `epsilon = rho * grad / (||grad|| + tiny_eps)`
3. Tam thoi day tham so toi diem bi perturb.
4. Tinh loss/gradient tai diem perturb do.
5. Khoi phuc tham so goc.
6. Cap nhat tham so goc bang SGD theo gradient cua loss perturb.

Chi dung thong tin local minibatch va local labels, khong them client state giua cac round.

### 3. Client routing

Cap nhat:

- `src/fl_baselines/clients/torch_client.py`

Them nhanh:

- `algorithm == "fedsam"` goi `train_fedsam_client(...)`

FedSAM se:

- nhan global parameters nhu baseline FedAvg
- train local voi SAM
- tra ve updated model parameters
- report train metrics theo convention hien tai: `train_loss`, `train_accuracy`

Khong can luu artifact/client state rieng tren disk.

### 4. Config surface

Cap nhat:

- `src/fl_baselines/core/config.py`
- `pyproject.toml`

Them hyperparameter:

- `fedsam-rho`

Default de xuat:

- `fedsam-rho = 0.5`

Ly do: paper dung `rho = 0.5` trong phan experimental setup va day la default hop ly de baseline gan paper nhat co the, trong khi van de nguoi dung override bang mot config chung.

Validation:

- `fedsam-rho` phai duong hoac it nhat khong am

Lua chon cu the:

- cho phep `rho = 0.0`

Ly do: dieu nay bien FedSAM thanh local SGD thuong, huu ich cho test boundary va khong gay mau thuan voi training loop.

## Data Flow

Moi round:

1. Server strategy `fedsam` sample clients va gui global model cung fit config.
2. Client set global model vao local model.
3. Client train bang SAM local optimizer tren train split cua chinh client.
4. Client tra local model cho server.
5. Server aggregate bang weighted average nhu FedAvg.
6. Framework tiep tuc server eval tren server test set va client eval tren held-out client test split nhu hien tai.

## Evaluation And Comparability

FedSAM phai giu nguyen semantics evaluation dang co cua repo de ket qua so sanh truc tiep duoc voi cac baseline khac khi dung cung mot file config:

- server eval tren server-side test set
- client eval tren test split duoc cat tu du lieu cua chinh client theo `client-test-fraction`
- metrics van bao gom `loss`, `accuracy`, `precision`, `recall`, `f1`

Khong them metric fairness, communication, hay compute vao baseline nay vi user chi uu tien model performance.

## Testing Plan

Can bo sung test truoc khi code implementation:

1. `tests/test_registry.py`
   - `fedsam` duoc register mac dinh

2. `tests/test_config.py`
   - parse duoc `fedsam-rho`
   - reject gia tri am

3. `tests/test_model_and_algorithm.py`
   - `FedSAMBuilder` tao duoc strategy dung kieu `CheckpointingFedAvg` / `FedAvg`
   - fit config chua `algorithm = "fedsam"` va `fedsam_rho`
   - ho tro cac model hien co
   - local training bang FedSAM thuc su cap nhat tham so model
   - boundary test voi `rho = 0.0` van train duoc

Muc tieu test la bao dam integration dong nhat voi pattern cua repo va khong lam vo kha nang so sanh baseline tren cung config.

## Risks And Decisions

### Decision: giu FedSAM o muc algorithm-only-on-client

Khong co server momentum hay global update correction o ban nay. Day la co y de bam dung baseline FedSAM trong paper, khong tron sang MoFedSAM.

### Decision: khong dua optimizer abstraction tong quat vao dot nay

Co the sau nay se co them SAM-family baseline, nhung hien tai uu tien patch nho, ro, va dong nhat voi codebase hon la refactor rong.

### Main risk

SAM can hai lan backward tren moi minibatch, de sinh ra:

- code local training dai hon SGD thuong
- can can than khi perturb/restore parameters

Giam rui ro bang cach dat training logic trong module rieng va them unit test truc tiep cho parameter update.

## Files Expected To Change

- `src/fl_baselines/algorithms/fedsam.py`
- `src/fl_baselines/training/fedsam.py`
- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `pyproject.toml`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/algorithms/fedsam.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/quickstart.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

## Success Criteria

- Chay duoc `algorithm="fedsam"` tren pipeline Flower hien tai.
- Cung mot config dataset/model/eval, FedSAM sinh ket qua so sanh truc tiep duoc voi cac baseline hien co.
- Khong thay doi semantics evaluation cua framework.
- Full test suite va compile checks pass sau khi tich hop.
