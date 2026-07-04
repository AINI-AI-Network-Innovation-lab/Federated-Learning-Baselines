# GAMF

`GAMF` trong paper gốc là `Graduated Assignment Model Fusion`: dùng graph matching với second-order similarity để align hidden units/channels trước khi fuse nhiều model.

## Diễn giải trong codebase này

Bản tích hợp hiện tại dùng `GAMF` như một server-side aggregation strategy cho federated learning:

- local training giữ nguyên như `FedAvg`
- server nhận toàn bộ local model của round hiện tại
- trước khi aggregate, server align từng hidden layer bằng second-order graph matching dựa trên incoming/outgoing weights
- sau khi align xong mới weighted-average theo số lượng mẫu của client

Điều này bám appendix của paper: với FL, authors thay phần `FedAvg` aggregation bằng module fusion của GAMF, còn local optimization có thể giữ nguyên hoặc kết hợp với thuật toán khác như `MOON`.

## Phạm vi hỗ trợ hiện tại

- hỗ trợ `mnist_cnn`
- hỗ trợ `lenet`
- chưa hỗ trợ `resnet9`, `resnet18`, `resnet34`, `inception`

Lý do là bản repo hiện tại chưa có một graph-construction path đủ an toàn cho residual connections và batch normalization.

## File chính

- [src/fl_baselines/algorithms/gamf.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/algorithms/gamf.py:1): builder, strategy, layer plan, Sinkhorn + Hungarian, second-order scoring
- [src/fl_baselines/defaults.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/defaults.py:1): đăng ký `gamf`
- [src/fl_baselines/core/config.py](/Users/nguyenhuyhieu/FL baseline using Flower/src/fl_baselines/core/config.py:1): config hyperparameters cho GAMF

## Khác biệt có chủ ý so với paper

- paper mô tả bản multi-graph matching đầy đủ với cycle consistency; bản repo hiện tại dùng một adaptation thực dụng hơn: anchor-based multi-client alignment quanh client tham chiếu
- giữ payload Flower chuẩn, không thêm client state mới
- ưu tiên tính ổn định và khả năng test trong framework hơn là tái hiện toàn bộ solver MGM ở quy mô lớn

Nói ngắn gọn: đây là baseline khả dụng trong repo hiện tại, trung thành với ý tưởng second-order fusion của paper, nhưng chưa phải bản reimplementation đầy đủ nhất của GAMGM nhiều-đồ-thị.

## Config

```toml
algorithm = "gamf"
model = "mnist_cnn"
gamf-sigma = 2.0
gamf-initial-tau = 0.05
gamf-descent-factor = 0.9
gamf-min-tau = 0.005
gamf-max-iters = 200
```

## Ví dụ chạy

```bash
flwr run . --run-config 'algorithm="gamf" model="mnist_cnn" gamf-sigma=2.0 gamf-max-iters=200' --stream
```
