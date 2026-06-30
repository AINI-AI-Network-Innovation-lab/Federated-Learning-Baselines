# SCAFFOLD

File chính:

- `src/fl_baselines/algorithms/scaffold.py`
- `src/fl_baselines/training/scaffold.py`
- `src/fl_baselines/clients/torch_client.py`

`ScaffoldBuilder` tạo `ScaffoldStrategy`, một Flower strategy tương thích FedAvg sampling nhưng có thêm server control variates. Mỗi round, server gửi:

```text
model_state + server_control_variates
```

Client dùng correction term:

```text
grad = grad + c - c_i
```

Sau local training, client cập nhật local control variates `c_i` và trả về:

```text
updated_model_state + delta_c_i
```

Server aggregate model weights theo weighted average và cập nhật server control từ trung bình `delta_c_i` của các client được chọn.

Config chính:

- `algorithm = "scaffold"`
