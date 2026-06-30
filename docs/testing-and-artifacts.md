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
- algorithm: FedAvg, FedAvgM, FedNova, FedPer, FedRep, FedProx, SCAFFOLD, MOON builder tạo được Flower strategy
- training: FedNova client updates, FedPer/FedRep shared/personal split, FedProx proximal term, SCAFFOLD control delta, MOON contrastive metrics
- evaluation: model trả logits trực tiếp hoặc tuple kiểu `(features, projection, logits)`

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
- `fedper_clients/<client-id>/personal.pt`: personal layer state cho FedPer client
- `fedrep_clients/<client-id>/personal.pt`: personal layer state cho FedRep client

Thư mục `outputs/` và `data/` được ignore trong `.gitignore` để không commit dữ liệu hoặc checkpoint vào repo.
