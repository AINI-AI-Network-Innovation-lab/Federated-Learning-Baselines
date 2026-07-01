# FedDyn Design

## Goal

Integrate the FedDyn baseline into the existing Flower + PyTorch FL framework in a way that stays as faithful as practical to the paper "Federated Learning Based on Dynamic Regularization" while preserving the repo's current extension patterns, test style, and runtime behavior.

## Scope

This work adds a new algorithm baseline only.

Included:

- New `feddyn` algorithm registration and runtime config.
- New server strategy implementing the FedDyn server update.
- New client-side local training path for the FedDyn objective.
- Persistent per-client FedDyn state stored under `output-dir`.
- Tests for config, registry, strategy creation, strategy aggregation, client behavior, and model compatibility.
- Docs updates for visible algorithm lists and usage.

Not included:

- New datasets, models, or app entrypoint redesign.
- Reproducing every experiment setting from the paper.
- New optimizer families beyond the repo's current SGD-based local training style.

## Paper Mapping

The paper's key algorithmic pieces that must be preserved are:

1. Each selected client minimizes a dynamic local objective:

   `L_k(theta) - <g_k_prev, theta> + (alpha / 2) * ||theta - theta_global||^2`

2. Each client keeps a dynamic linear-state term across rounds, represented in the paper through the recursively updated local gradient-like quantity.

3. Non-participating clients keep their local state unchanged.

4. The server updates both:
   - the global auxiliary state `h_t`
   - the new global model `theta_t = average(selected local models) - h_t / alpha`

To fit the current repo, we will preserve these semantics but express client state as a persisted per-client vector matching model parameters, rather than requiring full explicit storage of every paper variable exactly as written.

## Chosen Approach

Use a `paper-faithful, repo-native` implementation:

- Add `src/fl_baselines/algorithms/feddyn.py` for the server-side strategy and builder.
- Add `src/fl_baselines/training/feddyn.py` for the client local objective optimization.
- Add a FedDyn-specific branch in `src/fl_baselines/clients/torch_client.py`.
- Add one new config hyperparameter, `feddyn-alpha`.
- Reuse the repo's current patterns for parameter serialization, checkpointing, metrics aggregation, and client state persistence.

This keeps the algorithm recognizable from the paper without forcing a foreign architecture onto the codebase.

## Architecture Changes

### Config

Add:

- `feddyn-alpha` in `pyproject.toml`
- `feddyn_alpha: float` in `ExperimentConfig`
- validation that `feddyn-alpha` is positive

### Registry

Register `feddyn` in `src/fl_baselines/defaults.py`.

### Server Strategy

`FedDynStrategy` will:

- extend `FedAvg` similarly to `FedAdpStrategy` and `FedNovaStrategy`
- keep `global_parameters` as the current server model
- keep `h_t` as a list of NumPy arrays matching model parameter shapes
- on each aggregation round:
  - average the selected client models using sample-count weights
  - update `h_t = h_{t-1} - alpha * (1/m) * sum_k(theta_k^t - theta^{t-1})`
  - set the new global model to `avg_selected_model - h_t / alpha`
- save round checkpoints using the existing checkpoint utilities

The strategy fit config will include:

- `algorithm = "feddyn"`
- `local_epochs`
- `learning_rate`
- `feddyn_alpha`

### Client Behavior

FedDyn requires persistent client-side dynamic state across rounds.

Each client will:

- receive the current global model parameters
- load its prior FedDyn linear state from `output-dir/feddyn_clients/<client_id>/`
- optimize the FedDyn local objective using local SGD
- update and persist the linear state after training
- return the trained full model parameters to the server

Inactive clients are handled naturally by Flower partial participation because their client process is not asked to fit and their persisted state remains unchanged.

### Local Training

Add a dedicated `train_feddyn_client(...)` helper.

For each minibatch, optimize:

- cross-entropy local loss
- minus the linear term induced by the stored FedDyn client state
- plus the quadratic proximal term to the current global model weighted by `feddyn_alpha`

Implementation note:

To stay numerically stable and codebase-friendly, the linear term will be represented as a sum over flattened parameter-wise inner products with the stored client state tensors. The quadratic term will reuse the same parameter-wise structure already familiar from `train_one_client` with FedProx.

After local optimization, update the stored client state using the paper-consistent recursion:

- new_state = old_state - alpha * (local_model - global_model)

This is the repo-native representation of the paper's recursive dynamic regularization state.

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/feddyn.py`
- `src/fl_baselines/training/feddyn.py`
- `src/fl_baselines/clients/torch_client.py`
- `tests/test_config.py`
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `README.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/extending-baselines.md`
- `docs/testing-and-artifacts.md`
- `docs/algorithms/index.md`
- `docs/algorithms/feddyn.md`

## Test Plan

TDD order:

1. Config test for `feddyn-alpha` parsing and validation.
2. Registry test ensuring `feddyn` is registered.
3. Strategy builder test ensuring `FedDynBuilder` creates `FedDynStrategy` and passes fit config correctly.
4. Strategy aggregation test for a small linear model verifying:
   - `h_t` updates as expected
   - aggregated global parameters follow the FedDyn server formula
5. Client fit test verifying:
   - FedDyn path returns full parameters
   - client state is saved under `output-dir`
   - second round reuses prior state
6. Model compatibility smoke tests ensuring `feddyn` strategy builds for all current model builders.

## Risks And Mitigations

Risk: mapping the paper's dynamic gradient notation too literally could fight the repo's parameter-based architecture.

Mitigation: persist parameter-shaped client linear-state tensors, which are equivalent for optimization and consistent with the repo's existing stateful algorithms.

Risk: confusion between sample-weighted averaging and the paper's notation for selected-device averaging.

Mitigation: keep server aggregation aligned with the repo's existing weighting conventions and make this explicit in tests and docs.

Risk: local objective bugs due to sign mistakes in the linear term.

Mitigation: add targeted unit tests for one-step aggregation and for persisted client-state updates.

## Verification

Before claiming completion, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Then remove generated `__pycache__` directories.

## Open Assumptions

- Local optimization will use SGD epochs in the same style as the repo's other baselines, rather than an exact solver.
- The server will use the repo's standard weighted aggregation utilities where appropriate.
- Client FedDyn state will be persisted on disk under `output-dir`, matching the repo's current persistence pattern for `moon`, `fedper`, and `fedrep`.
