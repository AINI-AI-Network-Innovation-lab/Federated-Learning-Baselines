# FedEnt

FedEnt them adaptive learning rate theo entropy va mean-field vao local training, trong khi van giu server aggregation kieu FedAvg.

## Files

- `src/fl_baselines/algorithms/fedent.py`
- `src/fl_baselines/training/fedent.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Flow

1. Server strategy warm-start va cap nhat `phi1`, `phi2` theo round.
2. Server gui global model cung `fedent_phi1`, `fedent_phi2` va hyperparameters xuong client.
3. Client uoc luong local gradient, tinh `eta_i(t)` theo FedEnt, roi train local bang SGD voi learning rate dong.
4. Client gui raw local model parameters ve server, kem `fedent_weight_sq_norm` de server update mean-field state cho round sau.

## Config

- `fedent-beta`: trong so entropy/adaptation
- `fedent-gamma`: decay cho learning rate giua hai rounds
- `fedent-epsilon`: epsilon on dinh so cho chia va `log`
- `fedent-fixed-point-steps`: so lan refine noi bo mean-field state
- `fedent-max-learning-rate`: clamp tren cho adaptive learning rate
- `fedent-enable-decay`: bat/tat persisted `previous_eta`

## Artifacts

- `outputs/fedent_clients/<client-id>/state.pt`: luu `previous_eta` khi decay duoc bat

## Notes

- FedEnt hien duoc tich hop theo huong faithful-while-fit-repo: giu cong thuc round-level va mean-field semantics cua paper, nhung dung online warm-start/update thay cho mot pha precompute toan horizon `T`.
- Evaluation pipeline khong doi: server eval tren server test set, client eval tren held-out local test split, report `loss`, `accuracy`, `precision`, `recall`, `f1`.
