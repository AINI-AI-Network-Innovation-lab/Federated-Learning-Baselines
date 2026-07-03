# FedDisco

`feddisco` implements *FedDisco: Federated Learning with Discrepancy-Aware Collaboration* as a standalone FedAvg-style baseline.

Paper: <https://proceedings.mlr.press/v202/ye23f.html>

## Key Idea

FedDisco changes server aggregation weights for non-IID label distributions. Each client computes a scalar discrepancy between its local label distribution and a uniform target distribution, then returns that scalar in fit metrics.

For participating clients, the server computes:

- `n_k = num_examples_k / sum_j num_examples_j`
- `score_k = n_k - a * d_k + b`
- `p_k = ReLU(score_k) / sum_j ReLU(score_j)`

where `d_k` is the client discrepancy, `a` is `feddisco-discrepancy-weight`, and `b` is `feddisco-bias`.

If all ReLU scores are zero or invalid, the implementation falls back to normal sample-size weighting.

## Files

- `src/fl_baselines/algorithms/feddisco.py`: `FedDiscoStrategy` and `FedDiscoBuilder`.
- `src/fl_baselines/training/feddisco.py`: local label distribution and discrepancy helpers.
- `src/fl_baselines/clients/torch_client.py`: routes `algorithm="feddisco"` to the dedicated local helper.
- `src/fl_baselines/core/config.py`: FedDisco hyperparameters and validation.

## Config

```bash
flwr run . --run-config 'algorithm="feddisco" feddisco-discrepancy-weight=0.5 feddisco-bias=0.1 feddisco-metric="kl"' --stream
```

Supported config keys:

- `feddisco-discrepancy-weight`: non-negative paper hyperparameter `a`; default `0.5`.
- `feddisco-bias`: non-negative paper hyperparameter `b`; default `0.1`.
- `feddisco-metric`: one of `kl`, `l1`, `l2`, or `cosine`; default `kl`.
- `feddisco-epsilon`: positive numerical guard; default `1e-8`.

## Notes

This integration uses the paper's uniform target distribution in the first iteration. It does not implement secure aggregation for a private global target distribution.
