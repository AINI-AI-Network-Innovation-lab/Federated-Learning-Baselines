# FedLAW

`fedlaw` implements *Revisiting Weighted Aggregation in Federated Learning with Neural Networks* as a server-side adaptive aggregation baseline.

## Code Paths

- `src/fl_baselines/algorithms/fedlaw.py`: `FedLAWStrategy` va `FedLAWBuilder`
- `src/fl_baselines/app/server_app.py`: inject `server_loader` vao strategy qua `set_proxy_loader(...)`

## Repo-Fit Design

1. Client local training giu nguyen flow mac dinh kieu FedAvg.
2. Server nhan raw local models va khoi tao relative weights `lambda_i` theo ti le mau local.
3. Server dung `build_server_loader(...)` lam proxy dataset de hoc `lambda` va shrinking factor `gamma`.
4. Sau khi toi uu proxy loss, server aggregate thanh `gamma * sum_i lambda_i w_i`.

## Config

- `fedlaw-server-epochs`: so epoch toi uu `lambda` va `gamma` tren proxy loader
- `fedlaw-server-learning-rate`: learning rate cho server-side optimizer
- `fedlaw-gamma-init`: gia tri khoi tao cho shrinking factor

## Example

```bash
flwr run . --run-config 'algorithm="fedlaw" fedlaw-server-epochs=3 fedlaw-server-learning-rate=0.01 fedlaw-gamma-init=1.0' --stream
```

## Notes

- Ban tich hop hien tai dung server-side evaluation loader lam proxy dataset, giup giu abstraction hien co cua repo thay vi mo rong them mot pipeline proxy-data rieng.
- `gamma` duoc hoc voi rang buoc duong, con `lambda` duoc softmax hoa de dam bao khong am va tong bang 1.
