# FedAAW Design

## Goal

Integrate `FedAAW` into the current Flower baseline repository as a new algorithm option `fedaaw`, following the paper *Federated Learning With Adaptive Aggregation Weights for Non-IID Data in Edge Networks*.

This first iteration targets the paper's core behavior on top of `FedAvg`:

- clients compute the squared full-batch gradient norm before local updates
- the server tracks a per-client running statistic across rounds
- aggregation weights are adapted each round from data-size share and tracked gradient information

Out of scope for this iteration:

- immediate integration into `fedprox`, `fednova`, `moon`, `fedntd`, or other baselines
- exact reproduction of the paper's experiment stack
- introducing new datasets, models, or training apps outside the current repo architecture

## Repo Fit

The current codebase already supports algorithm-specific server strategies and client-side training branches:

- `src/fl_baselines/algorithms/` contains strategy builders
- `src/fl_baselines/clients/torch_client.py` routes algorithms that need custom local behavior
- `src/fl_baselines/training/` holds isolated local training helpers
- `src/fl_baselines/core/config.py` and `pyproject.toml` define runtime config
- tests already cover config parsing, registry wiring, strategy behavior, and client routing

FedAAW fits this structure well because it only needs:

- one extra scalar uploaded from each client per round
- one custom server aggregation rule
- no change to the local optimization objective itself

This makes `fedaaw` a good standalone baseline before any future extension to `FedProx + FedAAW` or similar hybrids.

## Method Summary

From the paper, each selected client `k` in round `t`:

1. receives the global model `w_t`
2. computes `||∇F_k(w_t,0)||^2` before local training
3. performs normal local SGD
4. uploads its updated model and the scalar gradient-norm value

The server maintains a tracker:

- `R_k^0 = g_k^0`
- `R_k^t = (t * R_k^(t-1) + g_k^t) / (t + 1)`

where `g_k^t = ||∇F_k(w_t,0)||^2`.

For the selected client set `U_t`, the server computes:

- `q_k^t = |D_k| / sum_{j in U_t} |D_j| + beta / R_k^t - gamma`
- `p_k^t = softmax(q_k^t)`

and aggregates local models using `p_k^t` instead of plain sample-size weights.

The paper notes that `gamma` has minor practical impact after normalization, while `beta` controls how strongly small gradient norms increase aggregation weight.

## Proposed Architecture

### 1. New Algorithm Module

Add:

- `src/fl_baselines/algorithms/fedaaw.py`

This module will define:

- `FedAAWStrategy`
- `FedAAWBuilder`

Responsibilities:

- keep a copy of current global parameters
- maintain per-client FedAAW tracker state
- compute adaptive aggregation weights from client metrics
- aggregate client models with those weights
- expose per-round config to clients
- save round checkpoints using the repo's existing checkpoint pattern

`FedAAWStrategy` should inherit from `FedAvg`, similar to `fedadp`, because the main customization point is `aggregate_fit`.

### 2. Client-Side Helper

Add:

- `src/fl_baselines/training/fedaaw.py`

This helper will compute the pre-update gradient norm on local data while keeping the local training loop otherwise aligned with `FedAvg`.

Responsibilities:

- estimate the squared gradient norm of the received global model on the client's local training data
- perform normal local training with existing SGD behavior
- return standard training metrics plus `fedaaw_grad_norm_sq`

The implementation should isolate the gradient-norm calculation here rather than embedding math directly into `torch_client.py`.

### 3. Client Routing

Update:

- `src/fl_baselines/clients/torch_client.py`

Add a dedicated branch:

- `algorithm == "fedaaw"` -> `_fit_fedaaw(...)`

This branch will:

- load global parameters into the local model
- read `local_epochs`, `learning_rate`, and FedAAW-specific config
- call `train_fedaaw_client(...)`
- return model parameters as usual
- return `fedaaw_grad_norm_sq` through metrics

No large client-side persistent state is required for the first FedAAW iteration.

## Aggregation-State Design

`FedAAWStrategy` needs a stable per-client key. The repo already uses `client_proxy.cid` when available in other custom strategies, so FedAAW should follow the same pattern.

Server state to keep:

- `global_parameters`: latest global model as NumPy arrays
- `gradient_trackers`: `dict[str, float]` mapping client id to `R_k^t`
- `last_aggregation_weights`: optional debug state for tests and inspection

### Tracker update rule

For each participating client:

- if it is the first observed round for that client, set tracker to the uploaded gradient norm
- otherwise update tracker using the paper's running average formula based on the server round index

Because Flower rounds are 1-indexed in this repo while the paper writes rounds from `t = 0`, the implementation should document the mapping clearly:

- use `round_index = server_round - 1` in the formula

This keeps behavior faithful and avoids off-by-one ambiguity.

## Config Design

Add new config fields in `ExperimentConfig` and `pyproject.toml`:

- `fedaaw_beta`
- `fedaaw_gamma`
- `fedaaw_epsilon`

Purpose:

- `fedaaw-beta`: controls weight contribution from inverse tracker
- `fedaaw-gamma`: constant shift term from the paper
- `fedaaw-epsilon`: implementation-only numerical guard for division safety

Defaults for the first integration:

- `fedaaw-beta = 0.01`
- `fedaaw-gamma = 1.0`
- `fedaaw-epsilon = 1e-8`

Validation:

- `fedaaw-beta` must be positive
- `fedaaw-gamma` must be non-negative
- `fedaaw-epsilon` must be positive

`fedaaw-epsilon` is a repo-fit adaptation, not a paper hyperparameter. It exists to prevent invalid values when a tracker is extremely small.

## Local Gradient-Norm Computation

The paper uses the squared full-batch gradient norm before local updates. In the repo, the most faithful practical implementation is:

1. load the received global model into the client model
2. run one loss accumulation pass over the full local training loader
3. backpropagate the averaged loss
4. flatten all parameter gradients
5. compute the squared `L2` norm
6. clear gradients
7. continue with normal local training

This design keeps the metric tied to the actual local objective while staying model-agnostic.

Trade-off:

- it adds one extra backward pass over local data each round
- it avoids approximating the paper with mini-batch-only noise unless we later need an optimization path

If a client has an empty training dataset, the helper should return:

- unchanged model parameters
- `fedaaw_grad_norm_sq = 0.0`
- zero-like training metrics following the repo's existing conventions

## Aggregation Rule

During `aggregate_fit`, the strategy should:

1. collect each client's uploaded model parameters
2. read `num_examples`
3. read `fedaaw_grad_norm_sq` from metrics
4. update that client's tracker
5. compute sample-size share among participating clients
6. compute `q_k`
7. compute normalized weights with softmax
8. aggregate client parameters with those adaptive weights

Numerical safety:

- divide by `max(tracker, epsilon)`
- if all weights become invalid, fall back to sample-size weighting
- keep all weight computations in `float64`

This fallback is an implementation safety net and should be documented as such.

## Testing Strategy

Tests should be added before implementation and cover:

### Config and registry

- `fedaaw` parses from kebab-case run config
- invalid `fedaaw-beta`, `fedaaw-gamma`, `fedaaw-epsilon` fail fast
- default registry includes `fedaaw`

### Builder and fit config

- `FedAAWBuilder` creates a `FedAAWStrategy`
- fit config includes `algorithm == "fedaaw"`
- fit config includes FedAAW hyperparameters

### Server behavior

- first-round tracker initializes from uploaded gradient norms
- later rounds update tracker with the expected running-average formula
- smaller tracker values lead to larger adaptive aggregation weights
- aggregation still returns valid model parameters

### Client behavior

- `TorchFlowerClient` routes `fedaaw` to dedicated training helper
- dedicated helper returns finite `fedaaw_grad_norm_sq`
- local training still updates parameters for supported models

### Model compatibility

Like recent baseline additions, include smoke coverage for the currently supported configurable models where feasible:

- `mnist_cnn`
- `lenet`
- `resnet9`
- `resnet18`
- `resnet34`
- `inception`

## Documentation Updates

After implementation, update:

- `README.md`
- `docs/algorithms/fedaaw.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

The README entry should mention the paper title and link, and the algorithm doc should clarify that this first version is a standalone `FedAvg`-based FedAAW integration.

## Risks And Decisions

### 1. Full-batch gradient cost

Risk:

- computing the gradient norm on the whole client dataset adds overhead

Decision:

- keep the first version faithful to the paper
- optimize later only if profiling shows it is necessary

### 2. Client identity across rounds

Risk:

- adaptive trackers depend on stable client ids

Decision:

- use Flower client proxy `cid` when available
- make tests assert tracker behavior with explicit client ids

### 3. Numerical instability from tiny trackers

Risk:

- `beta / R_k^t` can explode when `R_k^t` is very small

Decision:

- add `fedaaw_epsilon`
- add fallback to sample-size weighting if adaptive weights become invalid

### 4. Faithfulness versus repo simplicity

Risk:

- over-engineering now for future hybrid baselines would slow down the first integration

Decision:

- implement `fedaaw` only on `FedAvg` first
- keep strategy and training helper boundaries clean so future reuse is easy

## Deliverable

At the end of implementation, the repo should support:

- selecting `algorithm = "fedaaw"` from runtime config
- client upload of pre-update squared gradient norms
- server-side adaptive aggregation weights based on the paper
- passing tests and docs consistent with other built-in baselines
