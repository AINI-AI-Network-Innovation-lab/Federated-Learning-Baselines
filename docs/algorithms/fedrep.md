# FedRep

File chính:

- `src/fl_baselines/algorithms/fedrep.py`
- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/algorithms/fedper.py`

`FedRepBuilder` tạo `FedRepStrategy`, một strategy chỉ aggregate shared representation/base parameters. Personal/head parameters được giữ local theo từng client, tương tự FedPer, nhưng local training chạy hai pha:

1. Train personal head với representation frozen.
2. Train shared representation với personal head frozen.

Framework chọn personal layers theo các module có tham số ở cuối model. Mặc định `fedrep-personal-layers = 1`, nên các model hiện có giữ local classifier cuối:

- `mnist_cnn`: `fc2`
- `lenet`: `fc3`
- `resnet9`: `classifier`
- `resnet18`, `resnet34`, `inception`: `fc`

Local personal state được lưu tại:

```text
<output-dir>/fedrep_clients/<client-id>/personal.pt
```

Config chính:

- `algorithm = "fedrep"`
- `fedrep-personal-layers = 1`
- `fedrep-representation-epochs = 1`
