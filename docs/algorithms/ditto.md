# Ditto

Ditto thêm personalized local model ở client trên nền global FL flow hiện có của repo. Trong integration này, client vẫn gửi global-model update về server, còn personalized model được giữ local tại `output-dir/ditto_clients/<client-id>/personalized.pt`.

## Files

- `src/fl_baselines/algorithms/ditto.py`
- `src/fl_baselines/training/ditto.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "ditto"`
- `ditto-lambda = 0.1`
- `local-epochs`
- `learning-rate`

## Notes

- Global aggregation vẫn đi theo standard global strategy path.
- Personalized models không được aggregate.
- Personalized evaluation được hoãn ở integration pass này.
