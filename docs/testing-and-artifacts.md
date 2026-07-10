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
- algorithm: FedAvg, FedAvgM, FedAdagrad, FedAdam, FedYogi, FedAdp, GAMF, FedMA, FedGEN, Ditto, FedDC, FedDecorr, FedEnt, FedLAW, FedAAW, FedVCK, FedDyn, FedExP, FedSAM, FedSpeed, FedNTD, FedLC, FedRS, FedLAMA, FedProto, FedMeta, FedCurv, FedMMD, FedNP, APFL, FedNova, pFedMe, FedPer, FedRep, FedALA, FedAMP, FedLAA, FedProx, SCAFFOLD, MOON builder tạo được Flower strategy
- training: Ditto persisted personalized model, FedDC persisted drift/update state, FedDecorr representation decorrelation regularization, GAMF second-order server-side graph matching voi anchor-based multi-client alignment, FedMA layer-wise frozen-prefix local retraining va matched averaging tren hidden units, FedGEN feature masking payload va masked evaluation flow, FedEnt adaptive learning rate va persisted `previous_eta`, FedLAW proxy-data-driven aggregation weights va shrinking factor tren server, FedAAW pre-update squared gradient norm metric va adaptive server aggregation, FedVCK condensed payload generation, memory-capped server replay, va persisted `previous_model_state`, FedDyn persisted client state, FedExP adaptive server extrapolation step, FedSAM local SAM updates va client routing, FedSpeed persisted client state va amended payload, FedNTD local not-true distillation, FedLC local logits calibration, FedRS restricted softmax with observed-class masking, FedLAMA layer-wise sync masks with persisted full local state and round-level layer selection, FedProto persisted local model state, embedding regularization, and prototype-only server aggregation, FedMeta support/query meta-gradient payloads for MAML and Meta-SGD, FedCurv persisted local curvature state plus diagonal Fisher payload aggregation, FedMMD server-side parameter-space MMD scoring plus SKNQ-style client filtering and entropy-style weighting, FedNP latent Gaussian moment-matching payloads and representation regularization, APFL persisted personalized branch plus adaptive `alpha`, FedNova client updates, pFedMe personalized Moreau-style updates, FedPer/FedRep shared/personal split, FedALA adaptive local aggregation and persisted local model state, FedAMP attentive personalized cloud models and client proximal training, FedLAA layer-wise gradient-aligned server aggregation, FedProx proximal term, SCAFFOLD control delta, MOON contrastive metrics
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
- `fedcurv_clients/<client-id>/curvature_state.pt`: local diagonal Fisher state va `F * theta` de round sau client tru phan cua chinh no khoi global `u/v`
- `apfl_clients/<client-id>/personalized.pt`: personalized local branch cua APFL client
- `apfl_clients/<client-id>/alpha.json`: he so tron `alpha` hien tai cua APFL client
- `fedlama_clients/<client-id>/state.pt`: full local model state cho FedLAMA client khi layer chưa tới kỳ sync
- `pfedme_clients/<client-id>/personalized.pt`: personalized model state cho pFedMe client
- `fedper_clients/<client-id>/personal.pt`: personal layer state cho FedPer client
- `fedrep_clients/<client-id>/personal.pt`: personal layer state cho FedRep client
- `fedala_clients/<client-id>/state.pt`: local model state và ALA weights cho FedALA client

Thư mục `outputs/` và `data/` được ignore trong `.gitignore` để không commit dữ liệu hoặc checkpoint vào repo.
