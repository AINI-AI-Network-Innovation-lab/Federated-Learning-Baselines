# FedLWS

`fedlws` implements *FedLWS: Federated Learning with Adaptive Layer-wise Weight Shrinking* as a server-side aggregation baseline.

## Code Paths

- `src/fl_baselines/algorithms/fedlws.py`: `FedLWSStrategy` va `FedLWSBuilder`
- `src/fl_baselines/defaults.py`: registry key `fedlws`
- `src/fl_baselines/core/config.py`: config `fedlws-beta` va `fedlws-epsilon`

## Repo-Fit Design

1. Client local training giu nguyen flow mac dinh kieu FedAvg.
2. Server nhan raw local models va aggregate theo sample-weighted FedAvg de co `w_hat`.
3. Server suy ra local updates tu `local_model_i - global_model`.
4. Voi tung trainable floating parameter layer, server tinh `tau_l` la trung binh norm cua do lech local update so voi mean update.
5. Server tinh `gamma_l = ||w_l|| / (||w_l|| + beta * tau_l * ||w_hat_l - w_l||)`.
6. Server shrink aggregated layer bang `gamma_l * w_hat_l`; buffers/non-trainable state duoc aggregate binh thuong.

FedLWS khong can client route rieng trong `torch_client.py` va khong can proxy dataset.

## Config

- `algorithm = "fedlws"`
- `fedlws-beta = 0.1`: scaling term cua cong thuc shrink; paper goi y `0.1` cho CNN va `0.01` cho ResNet.
- `fedlws-epsilon = 1e-12`: nguong on dinh mau so khi norm gan 0.

## Example

```bash
flwr run . --run-config 'algorithm="fedlws" fedlws-beta=0.1' --stream
```

## Notes

- Ban tich hop dung local model deltas lam pseudo-gradients theo pseudo-code paper, vi server da co global weights va client weights trong Flower.
- FedLWS la module sau aggregation, nen ve nguyen ly co the ket hop voi aggregation khac; registry hien tai expose no nhu mot baseline rieng tren nen FedAvg de giu runtime contract gon.
