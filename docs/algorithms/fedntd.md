# FedNTD

FedNTD them local-side not-true distillation tren global FL flow hien co cua repo. Trong integration nay, server van aggregate kieu FedAvg; thay doi nam o objective local training, noi client hoc tu nhan ground-truth labels va dong thoi giu perspective cua global model tren cac not-true classes.

## Files Chinh

- `src/fl_baselines/algorithms/fedntd.py`
- `src/fl_baselines/training/fedntd.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "fedntd"`

## Config

- `algorithm = "fedntd"`
- `fedntd-beta = 1.0`
- `fedntd-temperature = 1.0`

## Hanh Vi

Moi round:

1. Server broadcast global model hien tai.
2. Client clone global snapshot lam `teacher_model`.
3. Client train local student model voi loss:
   - `cross_entropy`
   - `+ fedntd_beta * not_true_kl`
4. `not_true_kl` bo true-class logits khoi student/teacher softmax theo paper.
5. Client tra updated local model ve server de aggregate nhu FedAvg.

Teacher model chi la snapshot cua global model trong round hien tai va khong duoc persist qua round.

## Evaluation

- server eval chay tren server-side test set
- client eval chay tren held-out client test split theo pipeline hien tai
- metric giu nguyen:
  - `accuracy`
  - macro `precision`
  - macro `recall`
  - macro `f1`

## Chay Nhanh

```bash
flwr run . --run-config 'algorithm="fedntd" fedntd-beta=1.0 fedntd-temperature=1.0' --stream
```

## Ghi Chu So Sanh

FedNTD duoc tich hop de so sanh performance model trong cung pipeline data/model/round/eval cua framework hien tai. Repo nay khong co muc tieu ep cong bang theo compute/time giua cac baseline; FedNTD chi thay objective local training, con server aggregation va evaluation semantics duoc giu dong nhat voi cac baseline khac.
