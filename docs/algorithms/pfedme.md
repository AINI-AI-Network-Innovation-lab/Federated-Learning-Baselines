# pFedMe

`pFedMe` trong repo này bám paper "Personalized Federated Learning with Moreau Envelopes" theo hai ý chính:

- server giữ global/reference model và update với hệ số mixing `pfedme_beta`
- mỗi client tối ưu một personalized model local bằng objective proximal kiểu Moreau, rồi dùng personalized model đó để update reference model gửi về server

## Files Chính

- `src/fl_baselines/algorithms/pfedme.py`
  - `PFedMeBuilder` tạo strategy và đóng gói fit config
  - `PFedMeStrategy` override `aggregate_fit(...)` để thực hiện:
    - weighted average các local reference model
    - global update dạng `(1 - beta) * w_t + beta * avg(w_i,R)`
- `src/fl_baselines/training/pfedme.py`
  - chứa local training loop cho reference model và personalized model
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "pfedme"`
  - load/save personalized state theo client

## Config

`pFedMe` dùng các config chung của runtime như:

- `dataset`
- `model`
- `num-server-rounds`
- `num-supernodes`
- `local-epochs`
- `learning-rate`

và thêm các hyperparameter riêng:

- `pfedme-lambda`
  - cường độ proximal regularization giữa personalized model và reference model
- `pfedme-beta`
  - hệ số server-side mixing giữa global model cũ và average local reference model
- `pfedme-personal-learning-rate`
  - learning rate cho inner personalized optimization
- `pfedme-personal-steps`
  - số bước inner update của personalized model trong mỗi local step

## Client State

Mỗi client lưu personalized model ở:

```text
outputs/pfedme_clients/<client-id>/personalized.pt
```

State này được tái sử dụng ở các round sau để personalized model không bị reset hoàn toàn giữa các round.

## Evaluation

Repo giữ nguyên evaluation pipeline hiện tại:

- server eval chạy trên server-side test set với global/reference model
- client eval chạy trên held-out client test split theo pipeline hiện tại

Nói cách khác, vòng tích hợp này không thêm personalized evaluation riêng cho `pFedMe`.

## Ghi Chú So Sánh Baseline

Repo này dùng cùng data/model/round/eval setup để so performance model giữa các baseline. `pFedMe` có thêm inner personalized optimization nên không nhằm ép công bằng theo compute/time; nó được tích hợp để benchmark chất lượng model trong cùng pipeline chạy của framework.
