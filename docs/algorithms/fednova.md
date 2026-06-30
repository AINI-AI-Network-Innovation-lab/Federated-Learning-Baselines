# FedNova

File chính:

- `src/fl_baselines/algorithms/fednova.py`
- `src/fl_baselines/clients/torch_client.py`

`FedNovaBuilder` tạo `FedNovaStrategy`, một Flower strategy kiểu FedAvg sampling nhưng aggregate client updates bằng normalized averaging. Client không trả model weights sau local training; thay vào đó client trả cumulative update:

```text
initial_model_state - updated_model_state
```

Server tính hệ số normalize từ `local_norm` của từng client và tỷ lệ sample của client trong round, rồi cập nhật global parameters:

```text
global_model = global_model - normalized_update
```

Implementation này tính update từ `state_dict`, nên không phụ thuộc vào kiến trúc model cụ thể. Các model hiện có chỉ cần forward trả logits cho training classification như các algorithm còn lại.

Config chính:

- `algorithm = "fednova"`
- `fednova-server-momentum = 0.0`
