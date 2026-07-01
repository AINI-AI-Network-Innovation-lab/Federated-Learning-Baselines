# FedProto

FedProto them prototype aggregation vao global FL flow hien co cua repo. Trong integration nay, client van train classifier head va van duoc evaluate bang logits nhu cac baseline khac; prototype chi duoc dung de regularize embedding va de trao doi thong tin giua server-client.

## Files Chinh

- `src/fl_baselines/algorithms/fedproto.py`
- `src/fl_baselines/training/features.py`
- `src/fl_baselines/training/fedproto.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "fedproto"`

## Config

- `algorithm = "fedproto"`
- `fedproto-lambda = 1.0`

## Hanh Vi

Moi round:

1. Server broadcast global model hien tai cung global prototypes.
2. Client train local model voi loss:
   - `cross_entropy`
   - `+ fedproto_lambda * prototype_regularization`
3. Local prototype cua moi class duoc tinh bang mean embedding theo class.
4. Client upload:
   - updated model parameters
   - prototype sums theo class
   - class counts
5. Server aggregate:
   - model parameters theo FedAvg-style weighted average
   - global prototypes theo class bang prototype sums va counts

## Feature Extraction

FedProto khong doi `forward()` contract cong khai cua model. Thay vao do, repo them helper trich embedding penultimate representation cho cac model hien co trong `src/fl_baselines/training/features.py`.

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
flwr run . --run-config 'algorithm="fedproto" fedproto-lambda=1.0' --stream
```

## Ghi Chu So Sanh

FedProto duoc tich hop theo huong giu classifier/inference pipeline hien tai cua framework, nen ket qua van co the so sanh truc tiep voi cac baseline khac trong cung setup data/model/round/eval. Repo nay khong chuyen sang prototype-only inference cho FedProto.
