# FedDyn

FedDyn triển khai federated learning với dynamic regularization. Trong repo này, server giữ auxiliary correction state riêng, còn mỗi client lưu local linear-state tensor dưới `output-dir/feddyn_clients/<client-id>/state.pt`.

## Files

- `src/fl_baselines/algorithms/feddyn.py`
- `src/fl_baselines/training/feddyn.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "feddyn"`
- `feddyn-alpha = 0.1`
- `local-epochs`
- `learning-rate`

## Notes

- Client local optimization dùng SGD trên FedDyn objective.
- Client không được chọn ở round hiện tại sẽ giữ nguyên persisted state.
- Server update global model bằng weighted average của selected client models cộng correction từ auxiliary state.
