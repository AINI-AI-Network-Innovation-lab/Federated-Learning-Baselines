# FedLAA

FedLAA implements the server-side layer-wise adaptive aggregation proposed in "Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation". Clients still train locally like `FedAvg`; the server changes how returned model updates are combined.

## Files

- `src/fl_baselines/algorithms/fedlaa.py`

## Runtime Config

- `algorithm = "fedlaa"`
- `fedlaa-beta = 5.0`
- `local-epochs`
- `learning-rate`

## Behavior

- The strategy reconstructs approximate gradients from returned parameter deltas using `-delta / (learning_rate * local_epochs)`.
- For each parameter tensor in the model `state_dict()`, the server computes a data-size-weighted consensus gradient.
- Each client-tensor pair receives an angle to that consensus, and the strategy keeps a smoothed angle history per client and tensor across rounds.
- Smoothed angles are mapped through the paper's Gompertz contribution function, then normalized with sample-count weighting to produce tensor-specific aggregation weights.
- The next global model is built tensor-by-tensor instead of with one scalar weight for the whole client model.

## Notes

- In this repository, each tensor in the PyTorch `state_dict()` acts as the layer unit for FedLAA weighting.
- Client local training remains on the standard `TorchFlowerClient.fit()` path because the paper's novelty is entirely at the server aggregation stage.
- If a local or consensus gradient has zero norm, the implementation treats the angle as `pi / 2` and falls back safely instead of producing unstable weights.
