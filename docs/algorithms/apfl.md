# APFL

APFL trong repo này tích hợp personalized federated learning theo paper *Adaptive Personalized Federated Learning* với hai nhánh model ở client:

- nhánh `global` được train và gửi về server để aggregate kiểu FedAvg
- nhánh `personalized` được giữ local và trộn với global branch theo hệ số `alpha`

## Files chính

- `src/fl_baselines/algorithms/apfl.py`
  - strategy builder cho flow server-side kiểu FedAvg
- `src/fl_baselines/training/apfl.py`
  - local training loop cập nhật global branch, personalized branch, và tùy chọn cập nhật `alpha`
- `src/fl_baselines/clients/torch_client.py`
  - route `apfl`
  - load/save personalized state và `alpha`
  - evaluation bằng mixed model

## Cách hoạt động

Mỗi client giữ:

- `w`: bản local của global model
- `v`: personalized local model
- `alpha`: hệ số trộn

Personalized model khi train/eval là:

- `alpha * v + (1 - alpha) * w`

Trong integration này:

1. client nhận global parameters từ server
2. update `w` bằng local SGD bình thường
3. update `v` trên loss của mixed model
4. nếu `apfl-adaptive-alpha` bật, cập nhật `alpha` bằng một gradient step
5. chỉ nhánh `w` được gửi lại cho server

## Persisted state

APFL lưu state tại:

- `outputs/apfl_clients/<client-id>/personalized.pt`
- `outputs/apfl_clients/<client-id>/alpha.json`

State này giúp personalized branch và `alpha` không bị reset giữa các round.

## Config

| Key | Ý nghĩa |
| --- | --- |
| `apfl-alpha` | Hệ số trộn khởi tạo trong khoảng `(0, 1)` |
| `apfl-personal-learning-rate` | Learning rate cho personalized branch |
| `apfl-adaptive-alpha` | Có cập nhật `alpha` trong local training hay không |
| `apfl-alpha-learning-rate` | Learning rate cho cập nhật `alpha` |

## Ghi chú implementation

- Server side vẫn là aggregation kiểu FedAvg vì paper chỉ personal hóa ở client.
- Evaluation phía client dùng mixed personalized model, nên phản ánh đúng tinh thần APFL hơn việc chỉ đo global branch.
