# Testing And Artifacts

## Test

Test suite dùng Python `unittest`:

```bash
python -m unittest discover -s tests -v
```

Các nhóm test hiện có:

- registry: register, duplicate key, unknown key
- config: parse default và override config
- partitioning: IID/Dirichlet deterministic và không mất sample
- model: forward shape của MNIST CNN
- algorithm: FedAvg, FedAvgM, FedAdp, Ditto, FedDC, FedDecorr, FedEnt, FedVCK, FedDyn, FedExP, FedSAM, FedSpeed, FedNTD, FedProto, FedNova, pFedMe, FedPer, FedRep, FedProx, SCAFFOLD, MOON builder tạo được Flower strategy
- training: Ditto persisted personalized model, FedDC persisted drift/update state, FedDecorr representation decorrelation regularization, FedEnt adaptive learning rate va persisted `previous_eta`, FedVCK condensed payload generation, memory-capped server replay, va persisted `previous_model_state`, FedDyn persisted client state, FedExP adaptive server extrapolation step, FedSAM local SAM updates va client routing, FedSpeed persisted client state va amended payload, FedNTD local not-true distillation, FedProto embedding regularization và prototype aggregation payload, FedNova client updates, pFedMe personalized Moreau-style updates, FedPer/FedRep shared/personal split, FedProx proximal term, SCAFFOLD control delta, MOON contrastive metrics
- evaluation: server eval trên server test set; client eval trên held-out client test split; report `accuracy`, macro `precision`, macro `recall`, macro `f1`; model có thể trả logits trực tiếp hoặc tuple kiểu `(features, projection, logits)`

Khi thêm component mới, nên thêm test tối thiểu:

- dataset builder tạo được dataloader nhỏ
- model forward được một batch giả
- algorithm builder tạo được strategy
- config mới parse đúng default và override

## Artifacts

Mặc định output nằm trong `outputs/`:

- `run_config.json`: config của run
- `final_model.pt`: checkpoint cuối
- `round_<n>_model.pt`: checkpoint theo round
- `ditto_clients/<client-id>/personalized.pt`: personalized model state cho Ditto client
- `feddc_clients/<client-id>/state.pt`: drift state `h_i` và local update state `g_i` cho FedDC client
- `fedent_clients/<client-id>/state.pt`: state `previous_eta` cho FedEnt client khi decay duoc bat
- `fedvck_clients/<client-id>/state.pt`: state `previous_model_state` cho FedVCK client
- `feddyn_clients/<client-id>/state.pt`: dynamic regularization state cho FedDyn client
- `fedspeed_clients/<client-id>/state.pt`: client state `g_hat_i` cho FedSpeed
- `pfedme_clients/<client-id>/personalized.pt`: personalized model state cho pFedMe client
- `fedper_clients/<client-id>/personal.pt`: personal layer state cho FedPer client
- `fedrep_clients/<client-id>/personal.pt`: personal layer state cho FedRep client

Thư mục `outputs/` và `data/` được ignore trong `.gitignore` để không commit dữ liệu hoặc checkpoint vào repo.
