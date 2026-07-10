# FedAdam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FedAdam from "Adaptive Federated Optimization" as a first-class Flower baseline.

**Architecture:** FedAdam is an algorithm-only baseline that keeps existing client-side local SGD behavior and swaps the server strategy to Flower `FedAdam`. The builder owns FedAdam hyperparameters and checkpointing; generic app/client entrypoints remain unchanged.

**Tech Stack:** Python, Flower server strategies, PyTorch models, `unittest`, project registry/config/docs.

## Global Constraints

- Preserve unrelated dirty workspace changes.
- Use TDD: tests must fail before production code is added.
- Add config keys in kebab-case in `pyproject.toml` and snake_case in `ExperimentConfig`.
- Register the algorithm in `src/fl_baselines/defaults.py`.
- Update visible docs because the component list changes.

---

### Task 1: FedAdam Tests And Builder

**Files:**
- Create: `src/fl_baselines/algorithms/fedadam.py`
- Modify: `tests/test_model_and_algorithm.py`
- Modify: `tests/test_registry.py`
- Modify: `tests/test_config.py`
- Modify: `src/fl_baselines/core/config.py`
- Modify: `src/fl_baselines/defaults.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `ExperimentConfig`, `get_model_parameters`, `CheckpointingStrategyMixin`, `weighted_average`.
- Produces: `FedAdamBuilder.build_strategy(config, initial_model, evaluate_fn)` returning Flower `FedAdam`.

- [ ] **Step 1: Write failing tests**

Add tests that import `FedAdamBuilder`, assert default registration includes `fedadam`, parse `fedadam-eta`, `fedadam-eta-l`, `fedadam-beta-1`, `fedadam-beta-2`, `fedadam-tau`, validate invalid adaptive parameters, and assert the builder creates a Flower `FedAdam` with expected values.

- [ ] **Step 2: Run focused tests and observe expected failure**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_config tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedadam_builder_creates_flower_strategy`

Expected: FAIL because `fl_baselines.algorithms.fedadam` and config fields are missing.

- [ ] **Step 3: Implement minimal FedAdam integration**

Create `FedAdamBuilder`, add config fields and parsing/validation, register `fedadam`, and add default run config values.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_config tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_fedadam_builder_creates_flower_strategy`

Expected: PASS.

### Task 2: Docs And Final Verification

**Files:**
- Create: `docs/algorithms/fedadam.md`
- Modify: `README.md`
- Modify: `docs/algorithms/index.md`
- Modify: `docs/README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/overview.md`
- Modify: `docs/extending-baselines.md`
- Modify: `docs/testing-and-artifacts.md`

**Interfaces:**
- Consumes: `fedadam` algorithm key and config keys from Task 1.
- Produces: User-visible documentation for running FedAdam.

- [ ] **Step 1: Update docs**

Document FedAdam as local SGD plus adaptive server optimizer with `eta`, `eta_l`, `beta_1`, `beta_2`, and `tau`.

- [ ] **Step 2: Run full verification**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `PYTHONDONTWRITEBYTECODE=1 python -m compileall src tests`

Expected: compilation succeeds.

- [ ] **Step 3: Remove generated caches**

Remove generated `__pycache__` directories under `src/fl_baselines` and `tests`.
