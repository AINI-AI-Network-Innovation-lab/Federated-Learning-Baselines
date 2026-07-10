# FedLAMA

FedLAMA trong repo này bám theo paper *Layer-wise Adaptive Model Aggregation for Scalable Federated Learning* nhưng được thích nghi với contract round-based của Flower. Ý tưởng chính vẫn giữ nguyên: mỗi tensor trong `state_dict()` được xem như một layer unit, server theo dõi độ lệch giữa model local và global, rồi điều chỉnh chu kỳ sync theo từng layer.

## Files

- `src/fl_baselines/algorithms/fedlama.py`
- `src/fl_baselines/training/fedlama.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "fedlama"`
- `fedlama-base-interval`
- `fedlama-interval-factor`
- `local_epochs`
- `learning_rate`

## Behavior

- Server khởi tạo tất cả layer với base interval `tau'`.
- Sau mỗi round, server nhận model local và một vector discrepancy theo layer từ client.
- Strategy tính layer discrepancy theo kiểu paper, rồi tăng interval cho các layer có mức discrepancy thấp hơn ngưỡng ưu tiên tương đối.
- Client lưu full local state trong `output-dir/fedlama_clients/<client-id>/state.pt`, nên các layer chưa tới kỳ sync sẽ không bị reset về global model ở round sau.
- Trong adaptation này, Flower round được dùng như một đơn vị sync thực dụng; nếu không có layer nào tới hạn sync trong một round, strategy vẫn chọn một layer fallback để giữ protocol ổn định.

## Notes

- `fedlama-base-interval = 1` và `fedlama-interval-factor = 2.0` là cấu hình mặc định trong repo.
- Layer-wise discrepancy được tính trên toàn bộ tensor của model, nên baseline này chạy được với các model hiện có trong repo.
- Đây là bản tích hợp thực dụng cho codebase hiện tại, không phải mô phỏng step-level scheduler của paper nguyên bản theo từng mini-batch.
