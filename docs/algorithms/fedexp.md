# FedExP

FedExP them adaptive server extrapolation vao global FL flow hien co cua repo. Trong integration nay, client van train local SGD giong FedAvg va van tra local model parameters nhu mac dinh; phan khac biet nam o server, noi pseudo-gradient duoc suy ra tu local models de tinh server step size theo tung round.

## File Chinh

- `src/fl_baselines/algorithms/fedexp.py`

## Config

- `algorithm = "fedexp"`
- `fedexp-epsilon = 0.001`

## Hanh Vi

1. Server gui global model hien tai cho cac client duoc sample.
2. Moi client train local theo duong mac dinh cua repo, tuong tu FedAvg.
3. Client gui local model sau train ve server.
4. Server suy ra pseudo-gradient cua tung client:
   - `delta_i = global_t - local_i`
5. Server aggregate `delta_bar` theo sample weight va tinh:
   - `eta_g(t) = max(1, mean_i ||delta_i||^2 / (2 * (||delta_bar||^2 + epsilon)))`
6. Server update global model:
   - `global_{t+1} = global_t - eta_g(t) * delta_bar`

## Danh Gia

FedExP giu nguyen evaluation semantics cua framework:

- server eval tren server-side test set
- client eval tren held-out test split cua tung client
- report `loss`, `accuracy`, `precision`, `recall`, `f1`

Dieu nay giup ket qua FedExP co the so sanh truc tiep voi cac baseline khac khi dung cung dataset/model/config eval.

## Chay Nhanh

```bash
flwr run . --run-config 'algorithm="fedexp" fedexp-epsilon=0.001' --stream
```
