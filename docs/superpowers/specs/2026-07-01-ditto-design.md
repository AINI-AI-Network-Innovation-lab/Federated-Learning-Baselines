# Ditto Design

## Goal

Integrate the Ditto baseline into the existing Flower + PyTorch FL framework in a way that preserves the core personalized-update idea from the paper while minimizing disruption to the repo's current global training and evaluation flow.

## Scope

This work adds a new algorithm baseline only.

Included:

- New `ditto` algorithm registration and runtime config.
- A Ditto builder that reuses the repo's global FL strategy flow.
- New client-side personalized Ditto training path.
- Persistent per-client personalized Ditto model state stored under `output-dir`.
- Tests for config, registry, builder creation, client behavior, and model compatibility.
- Docs updates for visible algorithm lists and usage.

Not included:

- New server-side personalized evaluation/reporting flow.
- New metrics for fairness or robustness.
- Full reproduction of the paper's fairness/robustness experiments.
- Changes to app entrypoints or artifact schemas beyond client-side state persistence.

## Paper Mapping

The paper's core idea we need to preserve is:

1. Learn a global model `w` using a standard federated objective and solver.
2. Simultaneously maintain a personalized model `v_k` for each client.
3. Update `v_k` on device `k` using the objective:

   `F_k(v_k) + (lambda / 2) * ||v_k - w||^2`

4. Keep the personalized model state across rounds for each client.

For this integration pass, we will preserve the personalized local objective and persistent per-client personalized state, but we will not extend the repo's global evaluation path to report personalized metrics.

## Chosen Approach

Use a `minimal Ditto integration`:

- Add `src/fl_baselines/algorithms/ditto.py` as a lightweight builder reusing the standard global strategy flow.
- Add `src/fl_baselines/training/ditto.py` for client-side personalized optimization.
- Add a Ditto-specific branch in `src/fl_baselines/clients/torch_client.py`.
- Add one new hyperparameter, `ditto-lambda`.
- Persist each client's personalized model under `output-dir/ditto_clients/<client_id>/personalized.pt`.

This keeps the implementation close to the paper's personalized-update mechanism while staying aligned with the repo's existing conventions.

## Architecture Changes

### Config

Add:

- `ditto-lambda` in `pyproject.toml`
- `ditto_lambda: float` in `ExperimentConfig`
- validation that `ditto-lambda` is non-negative

### Registry

Register `ditto` in `src/fl_baselines/defaults.py`.

### Server Strategy

The global model path for Ditto will reuse the repo's standard global aggregation flow.

Implementation choice:

- `DittoBuilder` will build a standard checkpointing FedAvg-style strategy for the global model.
- The fit config sent to clients will include:
  - `algorithm = "ditto"`
  - `local_epochs`
  - `learning_rate`
  - `ditto_lambda`

This matches the paper's modularity point: Ditto can reuse an existing global FL solver while adding personalization locally.

### Client Behavior

Each selected client will do two things:

1. Train the shared/global branch exactly as the repo already expects for the global model update returned to the server.
2. Separately train a personalized local model `v_k`, regularized toward the current global model `w`, and persist it for the next round.

For this integration pass:

- the client will return only the global model update to the server
- the personalized Ditto model will remain local and be stored under `output-dir`
- evaluation will remain on the global model path unless explicitly extended in a future pass

### Local Training

Add a dedicated `train_ditto_personalized(...)` helper.

This helper will:

- start from the stored personalized model if it exists, otherwise initialize from the current global model
- optimize:

  `cross_entropy + (ditto_lambda / 2) * sum ||v_param - w_param||^2`

- save the updated personalized state back to disk

The global branch will continue to use the repo's existing `train_one_client(...)` behavior so that server-side aggregation remains consistent with the rest of the framework.

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/ditto.py`
- `src/fl_baselines/training/ditto.py`
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
- `docs/algorithms/ditto.md`

## Test Plan

TDD order:

1. Config test for `ditto-lambda` parsing and validation.
2. Registry test ensuring `ditto` is registered.
3. Builder test ensuring `DittoBuilder` creates a strategy and passes fit config correctly.
4. Client fit test verifying:
   - Ditto returns a standard global-model payload to the server
   - personalized state is saved under `output-dir`
   - second round reuses and updates the saved personalized state
5. Model compatibility smoke tests ensuring Ditto builds for all current model builders.

## Risks And Mitigations

Risk: Ditto in the paper jointly optimizes global and personalized models, while this integration keeps global evaluation/reporting unchanged.

Mitigation: make this scope explicit in docs and preserve the reusable global-solver plus local-personalization structure, which is still faithful to the paper's modularity.

Risk: ambiguity about whether returned client parameters should be personalized or global.

Mitigation: keep returned parameters global-only for compatibility with the current server aggregation flow, and encode this clearly in tests and docs.

Risk: local personalized state silently drifting if initialized incorrectly.

Mitigation: add tests that verify first-round creation and second-round reuse of personalized state.

## Verification

Before claiming completion, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Then remove generated `__pycache__` directories.

## Open Assumptions

- The global Ditto branch will reuse the current SGD-based local training flow already used by FedAvg-style baselines.
- Personalized Ditto models will be stored locally per client and not aggregated.
- Personalized evaluation and fairness/robustness reporting are deferred to a later iteration.
