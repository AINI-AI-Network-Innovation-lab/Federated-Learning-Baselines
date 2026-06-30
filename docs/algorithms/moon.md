# MOON

File chính:

- `src/fl_baselines/algorithms/moon.py`
- `src/fl_baselines/training/moon.py`
- `src/fl_baselines/clients/torch_client.py`

`MoonBuilder` tạo Flower strategy kiểu FedAvg ở phía server. Khác biệt nằm ở local training: client dùng model hiện tại, global model của round hiện tại, và local model của round trước để tính model-contrastive loss.

Loss local:

```text
cross_entropy(logits, target) + moon_mu * contrastive_loss(current, global, previous)
```

Nếu model trả tuple kiểu MOON, ví dụ:

```text
(features, projection, logits)
```

training loop sẽ dùng `projection` làm representation và `logits` cho classification. Nếu model chỉ trả logits như các model hiện tại, framework dùng logits làm representation fallback để thuật toán vẫn chạy được.

Local model của round trước được lưu tại:

```text
<output-dir>/moon_clients/<client-id>/previous_model.pt
```

Config chính:

- `algorithm = "moon"`
- `moon-mu = 1.0`
- `moon-temperature = 0.5`
