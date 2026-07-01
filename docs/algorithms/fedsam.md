# FedSAM

FedSAM giu server-side aggregation giong FedAvg va thay local ERM/SGD bang Sharpness Aware Minimization (SAM) tren moi client. Trong codebase nay, baseline nay duoc tich hop theo huong giu nguyen pipeline Flower hien co va chi thay local training objective.

## File Chinh

- `src/fl_baselines/algorithms/fedsam.py`
- `src/fl_baselines/training/fedsam.py`
- `src/fl_baselines/clients/torch_client.py`

## Config

- `algorithm = "fedsam"`
- `fedsam-rho = 0.5`

## Hanh Vi

1. Server gui global model hien tai cho cac client duoc sample.
2. Moi client train local bang SAM hai buoc tren cung minibatch:
   - tinh gradient tai tham so hien tai
   - perturb tham so theo `rho * grad / ||grad||`
   - tinh gradient tai diem perturb
   - khoi phuc tham so goc va cap nhat bang SGD
3. Client gui local model parameters ve server.
4. Server aggregate bang sample-weighted average nhu FedAvg.

## Danh Gia

FedSAM giu nguyen evaluation semantics cua framework:

- server eval tren server-side test set
- client eval tren held-out test split cua tung client
- report `loss`, `accuracy`, `precision`, `recall`, `f1`

Dieu nay giup ket qua FedSAM co the so sanh truc tiep voi cac baseline khac khi dung cung dataset/model/config eval.

## Chay Nhanh

```bash
flwr run . --run-config 'algorithm="fedsam" fedsam-rho=0.5' --stream
```
