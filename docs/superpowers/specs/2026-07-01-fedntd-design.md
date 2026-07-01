# FedNTD Design

## Goal

Tich hop thuat toan FedNTD tu paper "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning" vao framework hien tai theo huong bam paper nhat co the, nhung van giu nguyen:

- Flower app entrypoints
- registry/config pattern cua repo
- evaluation pipeline hien tai

## Scope

Included:

- them baseline moi `fedntd`
- local training loss theo paper:
  - `cross_entropy + beta * not_true_distillation`
- su dung global model cua round hien tai lam teacher trong local training
- config/tests/docs cho baseline moi

Not included:

- thay doi server aggregation semantics
- persistent client state rieng
- personalized evaluation flow
- them learning-rate decay, momentum schedule, hoac benchmark-specific training tricks ngoai pipeline hien tai

## Paper Mapping

Paper mo ta:

- server broadcast global model `w_t`
- moi client khoi tao local model tu global model
- local training toi uu objective:
  - `L = LCE + beta * LNTD`
- `LNTD` la KL divergence giua student va teacher tren not-true classes voi temperature `tau`
- server aggregate local models bang weighted average kieu FedAvg

Mapping sang repo:

- server strategy tiep tuc dung kieu `FedAvg`
- `learning_rate` tiep tuc la local optimizer learning rate
- them `fedntd_beta` cho trong so loss NTD
- them `fedntd_temperature` cho `tau`
- teacher model la ban clone cua global model duoc client nhan o dau round

## Chosen Architecture

### Server

Them `src/fl_baselines/algorithms/fedntd.py`:

- builder `FedNTDBuilder`
- strategy dung `CheckpointingFedAvg`
- fit config bo sung:
  - `algorithm = "fedntd"`
  - `fedntd_beta`
  - `fedntd_temperature`

FedNTD trong paper khong doi server update rule, nen giu server side giong FedAvg la cach sach nhat va de so sanh performance nhat.

### Client

Them nhanh `algorithm == "fedntd"` trong `TorchFlowerClient.fit(...)`:

- load global parameters vao `self.model`
- clone `teacher_model` tu global model
- train `self.model` bang helper moi trong `training/fedntd.py`
- tra local model parameters da cap nhat ve server

Khong can luu client state xuyen round, vi teacher moi round luon la global model vua duoc broadcast.

### Training

Them `src/fl_baselines/training/fedntd.py`:

- helper local training rieng cho FedNTD
- objective:
  - `cross_entropy(student_logits, targets)`
  - `+ fedntd_beta * LNTD(student_logits, teacher_logits, targets, temperature)`
- `LNTD` bo true-class logit khoi softmax cua ca teacher va student theo dung Equation (10)-(11)
- teacher chay `eval()` va khong nhan gradient

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
- `src/fl_baselines/algorithms/fedntd.py`
- `src/fl_baselines/training/fedntd.py`
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
- `docs/algorithms/fedntd.md`

## Risks And Choices

### Paper Fidelity vs Repo Consistency

Paper thuc nghiem dung mot so training details nhu momentum SGD, decay learning rate, va setup benchmark rieng. User yeu cau giu nguyen pipeline hien tai, nen lan tich hop nay chi mang phan thuat toan cot loi cua FedNTD vao objective local training.

### Distillation Scope

FedNTD khong dung full KD tren tat ca classes. Phan not-true masking la diem cot loi can giu nguyen; neu thay bang KD thong thuong thi se khong con la FedNTD nua.

### Model Support

Implementation phai hoat dong voi cac model hien tai, mien la training/evaluation path cuoi cung tra ra logits classification. Neu model tra tuple, can tach logits theo convention da co cua repo.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau do xoa `__pycache__`.
