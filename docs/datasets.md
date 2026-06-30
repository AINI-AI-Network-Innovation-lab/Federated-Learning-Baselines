# Datasets

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
