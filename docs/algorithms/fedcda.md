# FedCDA

`fedcda` implements *FedCDA: Federated Learning with Cross-Round Divergence-Aware Aggregation* as a server-side baseline that reuses the default local training path.

## Files

- `src/fl_baselines/algorithms/fedcda.py`: `FedCDAStrategy` and `FedCDABuilder`.
- `src/fl_baselines/core/config.py`: runtime config fields and validation.
- `src/fl_baselines/defaults.py`: default registry wiring.

## Behavior

FedCDA keeps a small history of recent local models for each client and chooses which cached model to aggregate in each round.

1. Each participating client still trains exactly like the default FedAvg-style local loop and returns the normal `train_loss`.
2. The server caches the latest local model for that client into a per-client history of size `fedcda-memory-size`.
3. During warmup rounds, the strategy behaves like current-round aggregation over the latest available model for each selected client.
4. After warmup, the server performs batch greedy cross-round selection over participating clients and keeps previously selected models for non-participating clients.
5. The final global model is the uniform average of the selected cached models.

This implementation is intentionally practical-faithful:

- it keeps the cross-round memory and divergence-aware selection idea from the paper
- it reuses `train_loss` already produced by the framework instead of adding a new client payload
- it aggregates over the set of clients with cached models currently known to the server

## Config

- `fedcda-memory-size`: number of cached local models kept per client; default `3`
- `fedcda-num-batches`: number of greedy selection batches; default `3`
- `fedcda-warmup-rounds`: number of rounds to stay in warmup mode before cross-round selection; default `50`
- `fedcda-loss-weight`: coefficient for the divergence-aware selection objective; default `1.0`

## Run

```bash
flwr run . --run-config 'algorithm="fedcda" fedcda-memory-size=3 fedcda-num-batches=3 fedcda-warmup-rounds=50' --stream
```

## Notes

- FedCDA does not require a custom client branch in `TorchFlowerClient`.
- Strategy metrics expose:
  - `fedcda_cache_size`
  - `fedcda_selected_client_count`
  - `fedcda_used_cross_round`
