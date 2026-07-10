# FedMeta

FedMeta integrates federated meta-learning with MAML and Meta-SGD style updates. The server maintains algorithm parameters: model initialization `theta` for MAML, or `theta` plus learned inner step tensors `alpha` for Meta-SGD. Clients split local batches into support/query sets, adapt on support data, compute query loss, and return meta-gradients to the server.

## Files Chinh

- `src/fl_baselines/algorithms/fedmeta.py`
- `src/fl_baselines/training/fedmeta.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "fedmeta"`

## Config

- `algorithm = "fedmeta"`
- `fedmeta-method = "maml"` or `"meta-sgd"`
- `fedmeta-inner-learning-rate = 0.01`
- `fedmeta-outer-learning-rate = 0.001`
- `fedmeta-support-fraction = 0.5`
- `fedmeta-inner-steps = 1`
- `fedmeta-first-order = true`
- `fedmeta-alpha-init = 0.01`

## Hanh Vi

Moi round:

1. Server sends algorithm parameters to selected clients.
2. Client splits each local batch into support and query parts.
3. Client performs one or more inner updates on support loss.
4. Client computes query loss with adapted parameters and returns meta-gradient payload.
5. Server weighted-averages meta-gradients and applies an outer update.

For `"maml"`, the payload is gradients for `theta`. For `"meta-sgd"`, the payload includes gradients for both `theta` and `alpha`.

## Chay Nhanh

```bash
flwr run . --run-config 'algorithm="fedmeta" fedmeta-method="maml" fedmeta-inner-learning-rate=0.01 fedmeta-outer-learning-rate=0.001' --stream
```
