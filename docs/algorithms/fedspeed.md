# FedSpeed

FedSpeed them quasi-gradient local training, persisted client state `g_hat`, va amended client payload `x_hat = x_K - lambda * g_hat` vao pipeline FedAvg hien co. Trong integration nay, server van aggregate bang weighted average nhu FedAvg; phan khac biet nam o local optimizer va payload client gui len.

## File Chính

- `src/fl_baselines/algorithms/fedspeed.py`
- `src/fl_baselines/training/fedspeed.py`
- `src/fl_baselines/clients/torch_client.py`

## Hành Vi Chính

- server dung `CheckpointingFedAvg`
- client load state `g_hat` theo `client_id` tu `output-dir`
- tren moi minibatch, client tinh `g1`, tao extra-step point, tinh `g2`, roi cap nhat model bang quasi-gradient correction
- sau local training, client cap nhat `g_hat` va gui payload amended `x_hat` len server

## Config

- `fedspeed-lambda`: he so prox/correction, mac dinh `0.1`
- `fedspeed-alpha`: he so tron giua `g1` va `g2`, mac dinh `1.0`
- `fedspeed-rho`: buoc perturbation extra-step, mac dinh `0.1`

## Client State

- `fedspeed_clients/<client-id>/state.pt`: luu `g_hat` xuyen round

## Evaluation Và So Sánh

FedSpeed giu nguyen evaluation semantics cua framework:

- server eval tren server-side test set
- client eval tren held-out client test split
- metrics gom `loss`, `accuracy`, `precision`, `recall`, `f1`

Dieu nay giup ket qua FedSpeed co the so sanh truc tiep voi cac baseline khac khi dung cung dataset/model/config eval.
