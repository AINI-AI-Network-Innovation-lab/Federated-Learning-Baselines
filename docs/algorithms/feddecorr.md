# FedDecorr

FedDecorr giu server-side aggregation giong FedAvg va chi thay local objective tren client thanh `cross_entropy + beta * decorrelation_loss`. Trong codebase nay, baseline nay duoc tich hop theo huong giu nguyen pipeline Flower hien co, evaluation semantics hien tai, va feature extraction abstraction da co.

## File Chính

- `src/fl_baselines/algorithms/feddecorr.py`
- `src/fl_baselines/training/feddecorr.py`
- `src/fl_baselines/training/features.py`
- `src/fl_baselines/clients/torch_client.py`

## Hành Vi Chính

- server dung `CheckpointingFedAvg`
- client trich representation bang `extract_features(...)`
- local loss la `ce_loss + feddecorr_beta * feddecorr_loss(features)`
- client van tra raw local model parameters nhu FedAvg

## Config

- `feddecorr-beta`: trong so cua decorrelation regularizer, mac dinh `0.1`

## Evaluation Và So Sánh

FedDecorr giu nguyen evaluation semantics cua framework:

- server eval tren server-side test set
- client eval tren held-out client test split
- metrics gom `loss`, `accuracy`, `precision`, `recall`, `f1`

Dieu nay giup ket qua FedDecorr co the so sanh truc tiep voi cac baseline khac khi dung cung dataset/model/config eval.
