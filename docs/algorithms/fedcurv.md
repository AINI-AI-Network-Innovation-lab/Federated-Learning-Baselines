# FedCurv

FedCurv trong repo này bám theo paper *Overcoming Forgetting in Federated Learning on Non-IID Data* ở mức giao thức:

- server broadcast model hiện tại cùng hai global aggregates `u = sum diag(F_j)` và `v = sum diag(F_j) * theta_j`
- mỗi client tối ưu local cross-entropy cộng curvature penalty dựa trên phần aggregate của "các client còn lại"
- sau local training, client ước lượng diagonal Fisher, tính `diag(F_i) * theta_i`, rồi upload lại cả model lẫn hai curvature tensors đó

## Files chính

- `src/fl_baselines/algorithms/fedcurv.py`
  - `FedCurvStrategy` giữ global `u/v` và phát payload mở rộng cho client
  - server aggregate model theo trung bình đều giữa các client tham gia, còn `u/v` được cộng trực tiếp theo paper
- `src/fl_baselines/training/fedcurv.py`
  - local training loop với curvature regularization
  - diagonal Fisher estimation từ một số mini-batch local sau khi train xong round hiện tại
- `src/fl_baselines/clients/torch_client.py`
  - route `fedcurv`
  - persist local curvature state tại `outputs/fedcurv_clients/<client-id>/curvature_state.pt`

## Cách regularization được áp dụng

Paper viết loss local round `t` cho client `s` là:

- `L_s(theta) + lambda * sum_{j != s} (theta - theta_j)^T diag(F_j) (theta - theta_j)`

Trong code, biểu thức này được viết lại theo `u/v`:

- `u_other = u_global - u_self_prev`
- `v_other = v_global - v_self_prev`

và phần phụ thuộc vào `theta` được tối ưu thành:

- `sum(u_other * theta^2) - 2 * sum(v_other * theta)`

Hằng số độc lập với `theta` được bỏ đi vì không ảnh hưởng gradient.

## Config

| Key | Ý nghĩa |
| --- | --- |
| `fedcurv-lambda` | Hệ số regularization curvature-aware |
| `fedcurv-fisher-batches` | Số mini-batch dùng để ước lượng diagonal Fisher |
| `fedcurv-stability-eps` | Ngưỡng chặn dưới cho Fisher diagonal |

## Ghi chú implementation

- Client phải lưu local Fisher round trước để trừ phần của chính nó ra khỏi global `u/v` ở round kế tiếp; đây là điểm quan trọng để bám đúng paper.
- Diagonal Fisher hiện được xấp xỉ bằng trung bình bình phương gradient của cross-entropy trên `fedcurv-fisher-batches` mini-batch local cuối round.
- Vì codebase này cho phép kích thước client data khác nhau, metrics vẫn aggregate theo sample count, còn model FedCurv được average đều giữa các client tham gia để gần hơn với mô tả trong paper.
