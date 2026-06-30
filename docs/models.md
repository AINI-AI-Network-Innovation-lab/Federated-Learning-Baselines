# Models

## MNIST CNN

File chính:

- `src/fl_baselines/models/base.py`
- `src/fl_baselines/models/mnist_cnn.py`

`MnistCnnBuilder` tạo một CNN nhỏ nhận input shape `(N, 1, 28, 28)` và output logits shape `(N, 10)`.

## Configurable Vision Backbones

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
