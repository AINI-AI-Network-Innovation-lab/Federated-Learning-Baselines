# FedAAW

FedAAW giu local training kieu FedAvg, nhung doi server aggregation sang adaptive weighting dua tren pre-update squared gradient norm cua tung client.

## Files

- `src/fl_baselines/algorithms/fedaaw.py`
- `src/fl_baselines/training/fedaaw.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Flow

1. Server strategy gui global model cung `fedaaw` hyperparameters xuong client.
2. Moi client tinh `||grad||^2` tren global model nhan duoc, truoc khi local SGD bat dau.
3. Client train local model bang local objective mac dinh nhu FedAvg va upload raw local model parameters kem metric `fedaaw_grad_norm_sq`.
4. Server cap nhat tracker theo client, tinh adaptive aggregation weights bang softmax cua sample-share va inverse tracker, roi aggregate local models bang trong so moi.

## Config

- `fedaaw-beta`: he so dieu chinh muc anh huong cua inverse gradient tracker len aggregation score
- `fedaaw-gamma`: constant shift term trong aggregation score
- `fedaaw-epsilon`: epsilon on dinh so de tranh chia cho 0 hoac tracker qua nho

## Artifacts

- FedAAW iteration hien tai khong tao client artifact rieng; no chi them metric `fedaaw_grad_norm_sq` vao fit result moi round

## Notes

- Day la ban tich hop `FedAvg`-based cua FedAAW cho codebase hien tai; chua mo rong ngay sang `FedProx + FedAAW`, `FedNova + FedAAW`, hay cac hybrid khac.
- Gradient norm duoc tinh theo full local train loader truoc local updates de giu implementation sat y paper hon so voi mini-batch approximation.
