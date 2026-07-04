# FedMA

`FedMA` trong repo này bám theo ý tưởng layer-wise matched averaging của paper gốc: server match hidden units theo từng layer, broadcast layer đã match, rồi client tiếp tục train các layer phía sau trong khi đóng băng prefix đã thống nhất.

## Phạm vi tích hợp hiện tại

- hỗ trợ `mnist_cnn`
- hỗ trợ `lenet`
- chưa hỗ trợ `resnet9`, `resnet18`, `resnet34`, `inception`

Lý do giới hạn này là paper gốc nhấn mạnh CNN/LSTM theo permutation invariance của channel/neuron, còn residual path và batch normalization cần xử lý cẩn thận hơn để tránh aggregate sai.

## Cách hoạt động trong codebase này

- [src/fl_baselines/algorithms/fedma.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/algorithms/fedma.py:1) định nghĩa `FedMABuilder`, `FedMAStrategy`, layer plan cho từng model hỗ trợ, và Hungarian matching bằng `scipy.optimize.linear_sum_assignment`
- [src/fl_baselines/training/fedma.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/training/fedma.py:1) chạy local training với danh sách prefix layer bị freeze theo `fedma_stage`
- [src/fl_baselines/clients/torch_client.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/clients/torch_client.py:1) route `algorithm="fedma"` sang local trainer riêng

Mỗi round:

1. server xác định `fedma_stage`
2. client freeze toàn bộ layer trước stage đó
3. client train phần còn lại và trả model kèm label counts local
4. server match các hidden units/channel của layer hiện tại, align layer kế tiếp, rồi aggregate
5. ở layer cuối, server average classifier theo label counts từng class thay vì chỉ theo sample count tổng

## Khác biệt có chủ ý so với paper

- không dùng BBP-MAP để nở kiến trúc động
- giữ fixed-width model đúng với kiến trúc đã build từ registry
- ưu tiên integration an toàn với Flower payload hiện có hơn là tái tạo toàn bộ solver Bayesian nonparametric

Điều này khiến baseline phù hợp để benchmark trong framework hiện tại, nhưng chưa phải bản tái hiện đầy đủ phần model-growth của paper.

## Config

```toml
algorithm = "fedma"
model = "mnist_cnn"
num-server-rounds = 4
fedma-matching-epsilon = 0.0
```

`num-server-rounds` nên ít nhất bằng số layer matchable:

- `mnist_cnn`: 4 stage
- `lenet`: 5 stage

## Ví dụ chạy

```bash
flwr run . --run-config 'algorithm="fedma" model="mnist_cnn" num-server-rounds=4 fedma-matching-epsilon=0.0' --stream
```
