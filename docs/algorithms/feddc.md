# FedDC

FedDC them local drift decoupling va correction vao global FL flow hien co cua repo. Trong integration nay, server van aggregate corrected local parameters theo mot strategy rieng, con moi client giu local drift `h_i` va local update state `g_i` duoi `output-dir/feddc_clients/<client-id>/state.pt`.

## Files Chinh

- `src/fl_baselines/algorithms/feddc.py`
- `src/fl_baselines/training/feddc.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "feddc"`

## Config

- `algorithm = "feddc"`
- `feddc-alpha = 0.01`

## Hanh Vi

Moi round:

1. Server broadcast global model hien tai cung server average update state `g`.
2. Client load global snapshot, drift state `h_i`, va local update state `g_i`.
3. Client train local model voi objective:
   - `cross_entropy`
   - `+ (feddc_alpha / 2) * ||h_i + theta - w||^2`
   - `+ <theta, g_i - g> / (lr * K)`
4. Client update:
   - `g_i <- theta_i^+ - theta_i`
   - `h_i <- h_i + g_i`
5. Client upload:
   - corrected parameter `theta_i^+ + h_i`
   - local update state `g_i`
6. Server average corrected parameters de lay global model moi va average `g_i` de cap nhat `g`.

## Client Artifacts

FedDC luu state theo client tai:

```text
outputs/feddc_clients/<client-id>/state.pt
```

State file chua:

- `drift`
- `local_update`

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
flwr run . --run-config 'algorithm="feddc" feddc-alpha=0.01' --stream
```

## Ghi Chu So Sanh

FedDC duoc tich hop de so sanh performance model trong cung pipeline data/model/round/eval cua framework hien tai. Repo nay khong ep cong bang theo compute/time; FedDC co them state phia server va client, nhung evaluation semantics va runtime config chung van duoc giu dong nhat voi cac baseline khac.
