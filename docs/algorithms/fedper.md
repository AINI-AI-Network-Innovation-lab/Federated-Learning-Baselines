# FedPer

File chính:

- `src/fl_baselines/algorithms/fedper.py`
- `src/fl_baselines/clients/torch_client.py`

`FedPerBuilder` tạo `FedPerStrategy`, một Flower strategy kiểu FedAvg sampling nhưng chỉ aggregate phần shared/base của model. Phần personal/head được lưu local theo từng client.

Framework chọn personal layers theo các module có tham số ở cuối model. Mặc định `fedper-personal-layers = 1`, nên các model hiện có sẽ giữ local classifier cuối:

- `mnist_cnn`: `fc2`
- `lenet`: `fc3`
- `resnet9`: `classifier`
- `resnet18`, `resnet34`, `inception`: `fc`

Client nhận shared parameters từ server, load personal parameters nếu đã có, train cả model, lưu lại personal parameters, rồi chỉ trả shared parameters để server aggregate.

Local personal state được lưu tại:

```text
<output-dir>/fedper_clients/<client-id>/personal.pt
```

Config chính:

- `algorithm = "fedper"`
- `fedper-personal-layers = 1`
