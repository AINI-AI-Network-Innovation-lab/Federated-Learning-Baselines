# Components

## Datasets

File chính:

- `src/fl_baselines/datasets/base.py`
- `src/fl_baselines/datasets/mnist.py`
- `src/fl_baselines/datasets/common.py`
- `src/fl_baselines/datasets/vision.py`
- `src/fl_baselines/datasets/partitioning.py`

Các dataset builder hiện có:

| Dataset key | Builder | Nguồn dữ liệu |
| --- | --- | --- |
| `mnist` | `MnistDatasetBuilder` | `torchvision.datasets.MNIST` |
| `fmnist` | `FashionMnistDatasetBuilder` | `torchvision.datasets.FashionMNIST` |
| `emnist` | `EmnistDatasetBuilder` | `torchvision.datasets.EMNIST` |
| `cifar10` | `Cifar10DatasetBuilder` | `torchvision.datasets.CIFAR10` |
| `cifar100` | `Cifar100DatasetBuilder` | `torchvision.datasets.CIFAR100` |
| `imagenet` | `ImageNetDatasetBuilder` | `torchvision.datasets.ImageNet` |

Mỗi builder tạo:

- client train loader
- client validation loader
- server evaluation loader

Partition hiện hỗ trợ:

- `iid`: chia đều ngẫu nhiên theo seed.
- `dirichlet`: chia non-IID theo label distribution và `dirichlet-alpha`.

Các dataset builder dùng chung `input-channels`, `input-height`, `input-width` để resize/convert channel theo model. Riêng `imagenet` không tự download dữ liệu; bạn cần chuẩn bị thư mục ImageNet hợp lệ trong `data-dir`.

## Model: MNIST CNN

File chính:

- `src/fl_baselines/models/base.py`
- `src/fl_baselines/models/mnist_cnn.py`

`MnistCnnBuilder` tạo một CNN nhỏ nhận input shape `(N, 1, 28, 28)` và output logits shape `(N, 10)`.

## Models: Configurable Vision Backbones

File chính:

- `src/fl_baselines/models/lenet.py`
- `src/fl_baselines/models/resnet.py`
- `src/fl_baselines/models/inception.py`

Các model này đọc chung config:

- `input-channels`
- `input-height`
- `input-width`
- `num-classes`

| Model key | Builder | Ghi chú |
| --- | --- | --- |
| `lenet` | `LeNetBuilder` | LeNet-style CNN với adaptive pooling |
| `resnet9` | `ResNet9Builder` | ResNet-9 nội bộ, nhẹ hơn ResNet torchvision |
| `resnet18` | `ResNet18Builder` | TorchVision ResNet-18, thay conv đầu theo `input-channels` |
| `resnet34` | `ResNet34Builder` | TorchVision ResNet-34, thay conv đầu theo `input-channels` |
| `inception` | `InceptionBuilder` | TorchVision Inception v3 với `aux_logits=False`; nên dùng input đủ lớn, ví dụ `75x75` trở lên |

## Algorithm: FedAvg

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedavg.py`

`FedAvgBuilder` tạo Flower `FedAvg` strategy với:

- initial parameters từ model ban đầu
- `fraction_fit`
- `fraction_evaluate`
- `min_fit_clients`
- `min_evaluate_clients`
- server-side evaluation function
- weighted metric aggregation
- checkpoint model sau mỗi round

## Algorithm: FedAvgM

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedavgm.py`

`FedAvgMBuilder` tạo Flower `FedAvgM` strategy. Thuật toán này giữ local training giống FedAvg, nhưng server update dùng momentum để giảm dao động khi dữ liệu client non-IID.

Config chính:

- `algorithm = "fedavgm"`
- `server-learning-rate = 1.0`
- `server-momentum = 0.9`

## Algorithm: FedProx

File chính:

- `src/fl_baselines/algorithms/base.py`
- `src/fl_baselines/algorithms/fedprox.py`
- `src/fl_baselines/training/train.py`

`FedProxBuilder` tạo Flower `FedProx` strategy. Strategy này giống FedAvg ở phía server aggregation, nhưng gửi thêm `proximal_mu` xuống client. Client training loop dùng proximal term:

```text
(mu / 2) * ||w - w_global||^2
```

Config chính:

- `algorithm = "fedprox"`
- `proximal-mu = 0.1`

## Algorithm: SCAFFOLD

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

## Algorithm: MOON

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
