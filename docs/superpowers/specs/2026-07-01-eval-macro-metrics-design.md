# Evaluation Macro Metrics Design

## Goal

Bổ sung `precision`, `recall`, và `f1` vào evaluation metrics của framework, áp dụng thống nhất cho cả:

- server-side evaluation trên server test set
- client-side evaluation trên held-out client test split

## Scope

Included:

- mở rộng `evaluate_model(...)` để trả thêm `precision`, `recall`, `f1`
- dùng macro-average cho bài toán multi-class
- giữ nguyên API hiện tại: `(loss, metrics_dict)`
- không thêm dependency mới
- cập nhật test và docs liên quan đến evaluation semantics

Not included:

- thay đổi training metrics
- thêm confusion matrix hoặc per-class metrics
- thêm config để chọn averaging mode

## Chosen Approach

Tính macro `precision`, `recall`, `f1` trực tiếp bằng PyTorch trong evaluation loop:

1. Thu thập `predictions` và `targets` theo từng batch.
2. Sau khi duyệt xong dataloader, tính thống kê theo từng lớp:
   - `tp`
   - `predicted_positive`
   - `actual_positive`
3. Với từng lớp:
   - `precision_c = tp / predicted_positive` nếu mẫu số > 0, ngược lại `0.0`
   - `recall_c = tp / actual_positive` nếu mẫu số > 0, ngược lại `0.0`
   - `f1_c` từ `precision_c` và `recall_c`, nếu tổng bằng 0 thì `0.0`
4. Lấy trung bình đều qua toàn bộ các lớp xuất hiện trong logits output dimension.

## Why Macro

User đã chọn macro averaging. Đây là lựa chọn phù hợp cho bối cảnh FL non-IID vì nó tránh để lớp lớn lấn át hoàn toàn chất lượng của lớp nhỏ.

## Architecture Impact

- `src/fl_baselines/training/evaluate.py` là điểm thay đổi chính.
- `weighted_average(...)` không cần đổi vì đã hỗ trợ aggregate scalar metrics theo số mẫu.
- Mọi strategy hiện tại sẽ tự động nhận metric mới ở evaluate path vì đều dùng `evaluate_model(...)`.

## Edge Cases

- Nếu `total_examples == 0`, trả tất cả metric classification về `0.0`.
- Nếu một lớp không có predicted positive hoặc actual positive, metric của lớp đó là `0.0`.
- Model trả tuple/list vẫn tiếp tục dùng phần logits cuối như hiện tại.

## Files To Change

- `docs/superpowers/specs/2026-07-01-eval-macro-metrics-design.md`
- `docs/superpowers/plans/2026-07-01-eval-macro-metrics.md`
- `src/fl_baselines/training/evaluate.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/testing-and-artifacts.md`

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Sau đó xóa `__pycache__`.
