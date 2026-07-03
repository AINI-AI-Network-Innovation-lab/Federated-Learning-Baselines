# FedDisco Design

## Goal

Integrate `FedDisco` into the current Flower baseline repository as a new algorithm option `feddisco`, following the paper *FedDisco: Federated Learning with Discrepancy-Aware Collaboration*.

This first iteration targets the paper's core standalone behavior on top of `FedAvg`:

- clients estimate the discrepancy between their local label distribution and a target distribution
- clients upload one scalar discrepancy metric with normal fit results
- the server weights selected clients by both sample share and discrepancy

Out of scope for this iteration:

- combining FedDisco with `fedprox`, `moon`, or other baselines
- secure aggregation for collecting a private global label distribution
- exact reproduction of the paper's experiment stack
- new datasets, models, or app entrypoints outside the current repo architecture

## Repo Fit

The current codebase already supports algorithm-specific strategies and client-side training branches:

- `src/fl_baselines/algorithms/` contains strategy builders
- `src/fl_baselines/clients/torch_client.py` routes algorithms with custom local behavior
- `src/fl_baselines/training/` holds isolated local training helpers
- `src/fl_baselines/core/config.py` and `pyproject.toml` define runtime config
- tests cover config parsing, registry wiring, strategy behavior, and client routing

FedDisco fits this structure well because it only needs:

- one local label-distribution scan
- one uploaded scalar discrepancy value
- one custom server aggregation rule

The first repo-fit implementation should be a standalone `feddisco` baseline that inherits from `FedAvg` and only overrides aggregation behavior.

## Method Summary

FedDisco addresses category distribution heterogeneity. For each selected client `k`, the method compares its local category distribution `D_k` against a target category distribution `T`.

The paper primarily uses a uniform target distribution:

- `T_c = 1 / C`

where `C` is the number of classes. Appendix material discusses privacy-preserving global-distribution estimation for imbalanced settings, but this first integration should use the uniform target because it is simple, local, and matches the paper's main fairness-oriented setup.

For selected clients in a round, the server computes:

- `n_k = |D_k| / sum_j |D_j|`
- `score_k = n_k - a * d_k + b`
- `p_k = ReLU(score_k) / sum_j ReLU(score_j)`

where:

- `d_k` is the discrepancy between local distribution `D_k` and target distribution `T`
- `a` controls discrepancy strength
- `b` is a positive bias term
- `p_k` is the aggregation weight

The paper reports `a` in roughly `[0.4, 0.6]` and `b = 0.1` as a safe practical range. The first integration should default to:

- `feddisco_discrepancy_weight = 0.5`
- `feddisco_bias = 0.1`

## Proposed Architecture

### 1. New Algorithm Module

Add:

- `src/fl_baselines/algorithms/feddisco.py`

This module will define:

- `FedDiscoStrategy`
- `FedDiscoBuilder`

Responsibilities:

- inherit from `FedAvg`
- maintain per-client discrepancy cache
- compute discrepancy-aware aggregation weights
- aggregate model parameters using FedDisco weights
- expose per-round config to clients
- save round checkpoints using the existing checkpoint pattern

`FedDiscoStrategy` should follow the style of existing custom strategies such as `fedadp`, `fedent`, `fedvck`, and `fedaaw`.

### 2. Client-Side Helper

Add:

- `src/fl_baselines/training/feddisco.py`

This helper will keep local optimization aligned with `FedAvg` while adding discrepancy computation.

Responsibilities:

- compute the local label distribution from the training loader
- compare it against a uniform target distribution
- support multiple discrepancy metrics
- call the existing normal local training path
- return standard training metrics plus `feddisco_discrepancy`

The helper should compute distribution statistics by iterating over loader labels rather than relying on dataset-specific attributes. This keeps it compatible with the repo's current dataset wrappers.

### 3. Client Routing

Update:

- `src/fl_baselines/clients/torch_client.py`

Add a dedicated branch:

- `algorithm == "feddisco"` -> `_fit_feddisco(...)`

This branch will:

- load global parameters into the local model
- read `local_epochs`, `learning_rate`, `num_classes`, and FedDisco config
- call `train_feddisco_client(...)`
- return model parameters as usual
- return `feddisco_discrepancy` through metrics

No persistent client-side state is required.

## Discrepancy Metrics

The paper discusses several discrepancy choices, with KL-Divergence commonly used in experiments and L1, L2, and cosine also reported as robust. The repo should expose:

- `kl`
- `l1`
- `l2`
- `cosine`

Default:

- `feddisco_metric = "kl"`

Implementation details:

- build local distribution `D_k` from class counts
- build uniform target `T`
- use `epsilon` for numerical safety
- return a finite scalar

Metric formulas:

- KL: `sum_c D_c * log((D_c + epsilon) / (T_c + epsilon))`
- L1: `sum_c abs(D_c - T_c)`
- L2: `sqrt(sum_c (D_c - T_c)^2)`
- cosine distance: `1 - cosine_similarity(D_c, T_c)`

If a client has no local examples, the helper should return discrepancy `0.0` and normal zero-like training metrics following existing repo conventions.

## Aggregation-State Design

`FedDiscoStrategy` needs a stable per-client key. The repo already uses `client_proxy.cid` in custom strategies, so FedDisco should use the same convention.

Server state to keep:

- `client_discrepancies`: `dict[str, float]`
- `last_aggregation_weights`: optional debug state for tests and inspection

The paper notes discrepancy communication is needed only once because label distributions are static. In Flower's fit flow, the first implementation can let clients include the scalar in fit metrics each round. The server will cache the latest value per client and reuse it if a future result omits the metric.

This is a small repo-fit adaptation:

- it avoids adding a separate pre-training discovery protocol
- it keeps partial participation simple
- it still transmits only one scalar per participating client

## Aggregation Rule

During `aggregate_fit`, the strategy should:

1. collect each client's uploaded model parameters
2. read `num_examples`
3. read `feddisco_discrepancy` from metrics
4. update the discrepancy cache for that client
5. compute sample-size shares among participating clients
6. compute `score_k = n_k - a * d_k + b`
7. apply ReLU to each score
8. normalize positive scores into aggregation weights
9. aggregate client parameters with these weights

Fallback behavior:

- if discrepancy is missing or non-finite, use cached discrepancy if available
- if no valid discrepancy exists for a client, use `0.0`
- if all ReLU scores are zero or invalid, fall back to sample-size weighting

All server-side weight computation should use `float64` for numerical stability.

## Config Design

Add new config fields in `ExperimentConfig` and `pyproject.toml`:

- `feddisco_discrepancy_weight`
- `feddisco_bias`
- `feddisco_metric`
- `feddisco_epsilon`

Purpose:

- `feddisco-discrepancy-weight`: paper hyperparameter `a`
- `feddisco-bias`: paper hyperparameter `b`
- `feddisco-metric`: discrepancy metric name
- `feddisco-epsilon`: implementation-only numerical guard

Defaults:

- `feddisco-discrepancy-weight = 0.5`
- `feddisco-bias = 0.1`
- `feddisco-metric = "kl"`
- `feddisco-epsilon = 1e-8`

Validation:

- `feddisco-discrepancy-weight` must be non-negative
- `feddisco-bias` must be non-negative
- `feddisco-metric` must be one of `kl`, `l1`, `l2`, or `cosine`
- `feddisco-epsilon` must be positive

## Testing Strategy

Add or extend tests to cover:

- config defaults and validation
- registry includes `feddisco`
- builder creates a FedDisco strategy
- fit config carries FedDisco-specific values
- local discrepancy helper returns lower discrepancy for uniform labels than skewed labels
- helper supports `kl`, `l1`, `l2`, and `cosine`
- client routing calls the FedDisco training helper
- strategy assigns higher weight to a lower-discrepancy client when sample sizes are equal
- strategy falls back to sample weighting when all ReLU scores are zero
- model and algorithm smoke tests include `feddisco`

## Documentation

Update `README.md` to include:

- `feddisco` in the algorithm list
- paper link: `https://proceedings.mlr.press/v202/ye23f.html`
- CLI/config example for selecting FedDisco
- short description of discrepancy-aware aggregation

## Risks And Trade-Offs

- Uniform target distribution is faithful to the paper's primary setting, but it may be suboptimal for naturally imbalanced datasets.
- Sending only scalar discrepancy avoids exposing raw class histograms, but it still reveals a small amount of distribution information.
- Computing label distribution by scanning the training loader adds a small local overhead.
- FedDisco weights can become degenerate when all scores are clipped to zero, so fallback sample weighting is necessary for robustness.

## Acceptance Criteria

The integration is complete when:

- `feddisco` is selectable from config and registry
- clients return `feddisco_discrepancy`
- server aggregation follows the FedDisco ReLU-normalized weighting rule
- README documents the new baseline and paper link
- unit tests pass for config, registry, helper, client routing, and strategy behavior
- full existing test discovery passes
