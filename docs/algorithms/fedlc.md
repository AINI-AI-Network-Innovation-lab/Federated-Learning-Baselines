# FedLC

FedLC them local-side logits calibration cho label distribution skew. Trong integration nay, server van aggregate kieu FedAvg; thay doi nam o local loss cua client, noi logits duoc calibrate theo local class counts truoc khi tinh cross-entropy.

## Files Chinh

- `src/fl_baselines/algorithms/fedlc.py`
- `src/fl_baselines/training/fedlc.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "fedlc"`

## Config

- `algorithm = "fedlc"`
- `fedlc-tau = 0.5`
- `fedlc-epsilon = 1e-8`

## Hanh Vi

Moi round:

1. Server broadcast global model hien tai.
2. Client tinh local class counts tu train loader cua partition hien tai.
3. Client train local model voi calibrated logits:
   - `adjusted_logits = logits - fedlc_tau * max(counts, fedlc_epsilon)^(-1/4)`
   - `loss = cross_entropy(adjusted_logits, targets)`
4. Client tra updated local model ve server de aggregate nhu FedAvg.

`fedlc-epsilon` giu missing classes on dinh ve so hoc khi local count bang zero. Missing va minority classes van nhan margin lon hon majority classes.

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
flwr run . --run-config 'algorithm="fedlc" fedlc-tau=0.5' --stream
```

## Ghi Chu So Sanh

FedLC trong repo nay tap trung vao phan algorithm: local logits calibration + FedAvg-style server aggregation. Paper cung benchmark quantity-based label skew `Q(alpha)`; repo hien co `iid` va `dirichlet`, nen can them partitioner rieng neu muon reproduce day du setup paper.
