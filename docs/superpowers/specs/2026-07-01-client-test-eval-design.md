# Client Test Eval Design

## Goal

Update the FL pipeline so that every baseline consistently uses:

- server-side evaluation on the server test set
- client-side evaluation on a held-out client test split derived from that client's own local data partition

This change should clarify evaluation semantics without disrupting the current extensible algorithm architecture.

## Scope

Included:

- Add a configurable `client-test-fraction`
- Replace ambiguous client `validation` loaders with explicit client `test` loaders
- Split each client's local partition into train/test subsets deterministically
- Keep server evaluation on the existing server-side test loader
- Update tests and docs to match the new semantics

Not included:

- New fairness or robustness metrics
- Personalized evaluation-specific changes for Ditto or other baselines
- Changes to the server evaluation protocol beyond naming clarification

## Current Problem

Today the repo uses:

- server eval on `build_server_loader(...)`, which already maps to the dataset test split
- client eval on a loader named `validation`, which is currently built from the dataset test split and partitioned by client

This means client evaluation is not using a held-out subset of each client's own local training data, which is the behavior now desired.

## Chosen Approach

Use a `client-held-out test split` design:

1. Partition the dataset's training split across clients exactly as before.
2. For each client partition, split that partition into:
   - client-train
   - client-test
3. Use client-train for local optimization.
4. Use client-test for `TorchFlowerClient.evaluate(...)`.
5. Keep server eval on the dataset test split through `build_server_loader(...)`.

This yields the two evaluation modes the user wants while preserving the current Flower app entrypoints and algorithm contracts.

## Architecture Changes

### Config

Add:

- `client-test-fraction` in `pyproject.toml`
- `client_test_fraction: float` in `ExperimentConfig`
- validation that `0 < client-test-fraction < 1`

### Shared Types

Update `ClientDataLoaders`:

- rename `validation` to `test`

This makes the pipeline semantics explicit.

### Dataset Builders

Update `TorchVisionDatasetBuilder.build_client_loaders(...)`:

- load only the training split for client-side partitioning
- partition the training data across clients
- split the selected client partition into train/test subsets using `client-test-fraction`
- build:
  - shuffled `train` loader
  - non-shuffled `test` loader

Determinism rules:

- the train/test split must be deterministic by `seed`
- every client split must avoid data loss
- if a client partition is very small, the split must still behave sensibly

For server evaluation:

- `build_server_loader(...)` remains based on the dataset test split

### Client Evaluation

Update `TorchFlowerClient.evaluate(...)`:

- replace use of `self.loaders.validation`
- use `self.loaders.test`

Algorithm-specific behavior for FedPer/FedRep remains the same, only the loader changes.

## Splitting Behavior

Given a client partition with `n` samples:

- compute a deterministic held-out client test subset of size approximately `round(n * client_test_fraction)` with sensible safeguards
- keep the remainder for local training

Safeguards:

- both train and test subsets should remain non-empty whenever `n >= 2`
- if `n < 2`, keep current behavior predictable and document the edge case clearly

## Files To Change

- `pyproject.toml`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/core/types.py`
- `src/fl_baselines/datasets/base.py`
- `src/fl_baselines/datasets/vision.py`
- `src/fl_baselines/clients/torch_client.py`
- `tests/test_config.py`
- `tests/test_datasets.py`
- `tests/test_model_and_algorithm.py` if a loader field name change reaches those tests
- `README.md`
- `docs/README.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/testing-and-artifacts.md`

## Test Plan

TDD order:

1. Config test for `client-test-fraction` parsing and validation.
2. Dataset test ensuring `build_client_loaders(...)` returns `train` and `test`.
3. Dataset test ensuring client train/test split is deterministic and non-empty for normal cases.
4. Any necessary evaluation-path test updates caused by renaming `validation` to `test`.
5. Full regression suite.

## Risks And Mitigations

Risk: changing from test-split partitioning to held-out client-train splitting changes the meaning of past client metrics.

Mitigation: update docs clearly so evaluation semantics are explicit from now on.

Risk: very small client partitions can produce degenerate train/test splits.

Mitigation: enforce deterministic safeguards and cover small-size behavior in tests.

Risk: renaming `validation` to `test` may break algorithms that assume the old field name.

Mitigation: update the shared dataclass and the only consumer paths systematically, then verify full tests.

## Verification

Before claiming completion, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m compileall src tests
```

Then remove generated `__pycache__` directories.

## Open Assumptions

- Server evaluation should continue using the current dataset test split.
- Client test splitting should be deterministic and derived from each client's local partition of the training split.
- This change should apply uniformly across all baselines without special-casing individual algorithms.
