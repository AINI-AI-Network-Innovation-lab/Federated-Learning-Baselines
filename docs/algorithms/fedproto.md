# FedProto

FedProto them prototype aggregation vao global FL flow hien co cua repo. Client train va giu local classifier/model rieng; server trao doi global prototypes de regularize embedding theo paper thay vi averaging local model weights.

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

1. Server broadcast model khoi tao/carrying payload cung global prototypes.
2. Client train local model voi loss:
   - `cross_entropy`
   - `+ fedproto_lambda * prototype_regularization`
3. Local prototype cua moi class duoc tinh bang mean embedding theo class.
4. Client upload:
   - local model parameters cho Flower payload/checkpoint compatibility
   - prototype sums theo class
   - class counts
5. Server aggregate:
   - global prototypes theo class bang prototype sums va counts
   - khong FedAvg local model parameters; server model payload duoc giu co dinh

FedProto local model state duoc luu duoi `output-dir/fedproto_clients/<client-id>/local_model.pt`, nen client khong bi reset ve server model o cac round sau.

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

FedProto duoc tich hop theo huong giu classifier/inference pipeline hien tai cua framework cho client eval. Repo nay khong chuyen sang prototype-only inference, nhung server-side training communication cua FedProto khong aggregate model weights.
