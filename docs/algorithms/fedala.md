# FedALA

FedALA adds Adaptive Local Aggregation before ordinary local training. Each client keeps its previous local model and ALA weights, blends the downloaded global model with the local model on selected higher layers, then trains and uploads the resulting local model through the FedAvg-style server path.

## Files

- `src/fl_baselines/algorithms/fedala.py`
- `src/fl_baselines/training/fedala.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "fedala"`
- `fedala-eta = 1.0`
- `fedala-rand-percent = 80`
- `fedala-layer-count = 1`
- `fedala-threshold = 0.01`
- `fedala-num-pre-loss = 10`
- `fedala-start-max-steps = 100`
- `local-epochs`
- `learning-rate`

## Client State

FedALA persists client state at `output-dir/fedala_clients/<client-id>/state.pt`. The state contains the latest local model, learned ALA weights, and whether the initial weight-learning phase is still active.

## Notes

- Server aggregation remains FedAvg-compatible.
- `fedala-layer-count` selects the last parameterized modules for ALA; lower modules are copied from the global model during local initialization.
- ALA weights are clipped into `[0, 1]` after each update.
- `fedala-start-max-steps` bounds the paper's initial convergence phase so smoke runs cannot hang indefinitely.
