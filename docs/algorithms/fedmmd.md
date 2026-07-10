# FedMMD

FedMMD trong repo này chỉ tích hợp phần federated aggregation của paper *FedMMD: A Federated weighting algorithm considering Non-IID and Local Model Deviation*. Repo vẫn giữ các model hiện có thay vì thêm backbone `DCMT`.

## Phạm vi tích hợp

- giữ local training path giống FedAvg
- thay server aggregation bằng một bước:
  - đo discrepancy giữa các local models
  - lọc client outlier theo SKNQ-style cutoff
  - gán entropy-style weights cho các client còn lại
  - aggregate theo trọng số kết hợp giữa sample ratio và entropy-style client weight

## Files chính

- `src/fl_baselines/algorithms/fedmmd.py`
  - `FedMMDStrategy`: tính discrepancy score, chọn client, rồi aggregate weighted average
  - `FedMMDBuilder`: đăng ký config runtime và tạo Flower strategy

## Khác paper ở đâu

Paper gốc kết hợp hai phần:

- local feature extractor `DCMT`
- server weighting `FedMMD`

Trong codebase này, chỉ phần server weighting được tích hợp. Ngoài ra, MMD được diễn giải theo kiểu implementation-oriented:

- local model parameters được flatten thành vector
- discrepancy giữa hai client được xấp xỉ bằng một parameter-space RBF score

Cách này giúp thuật toán khớp tốt với contract FedAvg hiện tại mà không cần thêm client payload mới hay một backbone riêng.

## Luồng aggregate

1. server nhận local models như FedAvg
2. flatten từng model thành vector tham số
3. tính pairwise discrepancy score cho từng client
4. dùng `fedmmd-sknq-threshold` để loại các client lệch mạnh
5. với các client được giữ lại, tính entropy-style client weights
6. trộn entropy weights với sample ratio để ra trọng số cuối cùng
7. aggregate global model bằng weighted average

## Config

| Key | Ý nghĩa |
| --- | --- |
| `fedmmd-sigma` | Độ rộng kernel RBF cho parameter-space discrepancy |
| `fedmmd-sknq-threshold` | Ngưỡng cutoff SKNQ-style trong khoảng `(0, 1)` |
| `fedmmd-min-clients` | Số client tối thiểu phải còn lại sau filtering |
| `fedmmd-entropy-eps` | Epsilon ổn định cho score normalization và weighting |

## Ghi chú

- FedMMD hiện là strategy-only baseline, nên không cần route client riêng.
- Implementation này ưu tiên bám “tinh thần” aggregation của paper trong khi vẫn giữ repo gọn và tương thích với mọi model hiện có.
