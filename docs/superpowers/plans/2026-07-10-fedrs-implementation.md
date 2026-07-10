# FedRS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FedRS as a first-class baseline for label-distribution-skew federated learning.

**Architecture:** FedRS keeps server aggregation identical to FedAvg and changes only the client-side classification loss. Each client derives an observed-class mask from its local training loader and applies restricted softmax so missing classes receive downscaled indirect pushing via `alpha`.

**Tech Stack:** Python, Flower strategies, PyTorch, `unittest`, project registry/config/docs.

## Global Constraints

- Preserve unrelated dirty workspace changes.
- Use TDD: tests must fail before production code is added.
- Keep the implementation FedAvg-compatible on the server side.
- Add config keys in kebab-case in `pyproject.toml` and snake_case in `ExperimentConfig`.
- Update visible docs because the component list changes.

---

### Task 1: FedRS Tests And Core Integration

**Files:**
- Create: `src/fl_baselines/algorithms/fedrs.py`
- Create: `src/fl_baselines/training/fedrs.py`
- Modify: `src/fl_baselines/clients/torch_client.py`
- Modify: `src/fl_baselines/core/config.py`
- Modify: `src/fl_baselines/defaults.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_model_and_algorithm.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `CheckpointingFedAvg`, `TorchFlowerClient.fit`, `DataLoader`.
- Produces: `FedRSBuilder.build_strategy(config, initial_model, evaluate_fn)`, `fedrs_loss(...)`, `train_fedrs_client(...)`.

- [ ] **Step 1: Write failing tests**

Add tests for:
- registry inclusion of `fedrs`
- config parsing/validation of `fedrs-alpha`
- builder creation returning a FedAvg-compatible strategy and fit config with `algorithm="fedrs"`
- client routing to a dedicated FedRS trainer
- finite restricted-softmax loss with missing classes
- local training updates model parameters

- [ ] **Step 2: Run focused tests and observe expected failure**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_config tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedrs_builder_creates_strategy tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedrs_loss_handles_missing_classes`

Expected: FAIL because `fedrs` modules/config/routing do not exist yet.

- [ ] **Step 3: Implement minimal FedRS integration**

Implement:
- `FedRSBuilder` using `CheckpointingFedAvg`
- local observed-class mask extraction from the client train loader
- restricted-softmax style loss using `alpha` to scale missing-class logits contribution
- `TorchFlowerClient` route for `fedrs`
- `fedrs-alpha` config parsing and validation

- [ ] **Step 4: Run focused tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_config tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedrs_builder_creates_strategy tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedrs_loss_handles_missing_classes`

Expected: PASS.

### Task 2: Docs And Final Verification

**Files:**
- Create: `docs/algorithms/fedrs.md`
- Modify: `README.md`
- Modify: `docs/algorithms/index.md`
- Modify: `docs/README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/overview.md`
- Modify: `docs/extending-baselines.md`
- Modify: `docs/testing-and-artifacts.md`

**Interfaces:**
- Consumes: `fedrs` algorithm key and `fedrs-alpha` config.
- Produces: user-facing docs for running FedRS.

- [ ] **Step 1: Update docs**

Document FedRS as local restricted softmax with `alpha` controlling missing-class indirect pushing while server aggregation remains FedAvg-style.

- [ ] **Step 2: Run full verification**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `PYTHONDONTWRITEBYTECODE=1 python -m compileall src tests`

Expected: compilation succeeds.

- [ ] **Step 3: Remove generated caches**

Remove generated `__pycache__` directories under `src/fl_baselines` and `tests`.
