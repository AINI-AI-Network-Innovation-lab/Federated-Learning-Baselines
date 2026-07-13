# FedDRL

`FedDRL` implement adaptive server aggregation theo paper "FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning".

## Files

- `src/fl_baselines/algorithms/feddrl.py`: strategy `FedDRLStrategy`, replay buffer, actor-critic networks, reward update, va aggregation weights.
- `src/fl_baselines/training/feddrl.py`: helper train client, tinh `pre_train_loss`, `post_train_loss`, va reward utility.
- `src/fl_baselines/clients/torch_client.py`: route `algorithm="feddrl"` de client gui them metrics can thiet cho state cua server.

## Server State Va Action

Moi round, client gui ve:

- `feddrl_pre_train_loss`: loss cua global model tren local train loader truoc khi local update
- `feddrl_post_train_loss`: loss sau local train
- `num_examples`: so mau local, duoc chuan hoa thanh sample ratio trong state

Server xay state theo tung client: `[pre_train_loss, post_train_loss, sample_ratio]`.
Actor network sinh raw action score cho moi client, sau do `softmax` de ra impact factor va aggregate model.

## Reward

Reward duoc thiet ke theo dung tinh than paper:

- giam average pre-train loss tren tat ca client
- giam chenh lech `max(pre_loss) - min(pre_loss)` de tranh bias ve mot nhom client

Trong code, reward duoc dua ve dang toi da hoa:

```text
reward = -(mean_pre_loss + bias_gap)
```

## Hyperparameters

Them trong `ExperimentConfig` va `pyproject.toml`:

- `feddrl-actor-learning-rate`
- `feddrl-critic-learning-rate`
- `feddrl-discount-factor`
- `feddrl-target-tau`
- `feddrl-hidden-size`
- `feddrl-replay-buffer-size`
- `feddrl-batch-size`
- `feddrl-updates-per-round`
- `feddrl-noise-scale`
- `feddrl-std-scale`

## Notes

- Implementation hien tai chon huong practical-faithful: actor-critic online trong strategy server.
- Paper co mo ta them two-stage worker/offline RL training; phan do chua duoc dua vao repo nay de giu baseline gon va phu hop runtime Flower hien tai.
- Ten algorithm chuan trong registry, docs, va fit config la `feddrl`.
