# FedLAA Design

## Goal

Integrate FedLAA from "Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation" into the Flower baseline repository as a new algorithm option that preserves existing client training behavior and adds server-side layer-wise adaptive aggregation.

## Constraints

- Follow existing algorithm-builder and strategy registration patterns in the repository.
- Keep client-side behavior identical to standard local training unless the paper requires otherwise.
- Avoid changing unrelated existing FedALA and FedAMP work already present in the tree.
- Support current model builders that expose standard PyTorch `state_dict()` parameter ordering.

## Chosen Approach

Implement FedLAA as a dedicated `FedAvg`-derived strategy. The strategy will:

- reconstruct per-client layer-wise gradients from returned model updates;
- compute a data-size-weighted consensus gradient for each layer;
- compute per-client, per-layer angular similarity against that consensus;
- maintain smoothed layer angles across rounds per client;
- map smoothed angles through the Gompertz contribution function from the paper;
- normalize weights per layer using sample counts;
- aggregate each layer independently with those weights.

Clients will continue to use the generic `TorchFlowerClient.fit()` training path used by `FedAvg`, because the paper's novelty is on the server side.

## File Boundaries

- `src/fl_baselines/algorithms/fedlaa.py`
  New strategy and builder implementation.
- `src/fl_baselines/core/config.py`
  Add `fedlaa_beta` config parsing and validation.
- `src/fl_baselines/defaults.py`
  Register the new algorithm builder.
- `pyproject.toml`
  Expose the new run-config option.
- `tests/test_config.py`
  Config parsing and validation coverage.
- `tests/test_registry.py`
  Registry coverage for the new algorithm.
- `tests/test_model_and_algorithm.py`
  Strategy behavior, builder wiring, and model compatibility coverage.
- `docs/algorithms/fedlaa.md` and overview docs
  User-facing documentation.

## Data Flow

1. Server sends the current global parameters and standard fit config.
2. Clients perform normal local training and return full model parameters.
3. FedLAA computes layer-wise updates by subtracting previous global parameters.
4. FedLAA reconstructs approximate gradients using `-delta / (learning_rate * local_epochs)`.
5. For each layer, FedLAA computes the data-proportional consensus gradient.
6. For each client-layer pair, FedLAA computes the angle to the consensus gradient and updates the running smoothed angle.
7. FedLAA converts smoothed angles to contribution scores with the paper's Gompertz mapping, then normalizes scores with sample-count weighting.
8. FedLAA aggregates each layer independently into the next global model.

## Edge Handling

- If a layer consensus gradient or local gradient has zero norm, fall back to `pi/2` for that angle so the layer is treated as neutral rather than exploding numerically.
- If per-layer normalized weights become non-finite or sum to zero, fall back to sample-proportional weights for that layer.
- Keep client keys consistent with existing strategies by preferring `client_proxy.cid`.

## Testing

- Verify config parsing and validation for `fedlaa-beta`.
- Verify registry exposure.
- Verify builder creates a `FedLAA` strategy and passes the algorithm fit config.
- Verify aggregation can favor a layer from one client and another layer from another client when their directions differ.
- Verify strategy remains compatible with the current model zoo through builder smoke tests.

