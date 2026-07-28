# Faithful FedADMM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `g=0` FedADMM adaptation with a faithful Algorithm 2 integration supporting client primal/dual state, delta-hat partial participation, and server-side proximal operators.

**Architecture:** Keep Flower entrypoints generic. Add a small proximal-operator registry, make `FedADMMStrategy` own `tilde_x` and `bar_x`, and make each client persist `x_i`, `z_i`, and `hat_x_i`. Use explicit client-id metadata in the fit payload so server state is independent of Flower proxy naming.

**Tech Stack:** Python 3.10+, Flower 1.26.1, PyTorch 2.8, NumPy, existing `unittest` suite.

## Global Constraints

- Preserve existing `algorithm="fedadmm"` and `fedadmm-alpha` configurations.
- Add `fedadmm-penalty` as the canonical alias for paper parameter `eta`.
- Support `identity`, `l1`, and `box` proximal operators through a registry.
- Keep local neural-network minimization inexact and bounded by explicit SGD steps.
- Do not change unrelated algorithms or dataset/model builders.
- Use TDD: every production behavior is introduced by a failing test first.

---

### Task 1: Add proximal operator contracts and config parsing

**Files:**
- Create: `src/fl_baselines/training/proximal.py`
- Modify: `src/fl_baselines/core/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`
- Test: `tests/test_model_and_algorithm.py`

**Interfaces:**
- Produces `ProxOperator` protocol, `PROX_OPERATORS`, `IdentityProx`, `L1Prox`, and `BoxProx`.
- Produces config fields `fedadmm_penalty`, `fedadmm_local_steps`, `fedadmm_tolerance`, `fedadmm_prox`, `fedadmm_l1_weight`, `fedadmm_box_min`, and `fedadmm_box_max`.

- [ ] Write tests for identity, L1 soft-thresholding, box projection, alias precedence, and invalid parameters.
- [ ] Run focused tests and confirm they fail because the new module/fields do not exist.
- [ ] Implement the proximal protocol and registry with shape/dtype-preserving NumPy operations.
- [ ] Parse canonical `fedadmm-penalty`; if absent, use `fedadmm-alpha`; validate positive penalty/steps/tolerance and prox-specific parameters.
- [ ] Add default Flower config keys while retaining `fedadmm-alpha` compatibility.
- [ ] Run focused tests and confirm they pass.

### Task 2: Replace FedADMM client training with faithful state/payload semantics

**Files:**
- Modify: `src/fl_baselines/training/fedadmm.py`
- Modify: `src/fl_baselines/clients/torch_client.py`
- Test: `tests/test_model_and_algorithm.py`

**Interfaces:**
- `train_fedadmm_client(...) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]` returns metrics, local `x`, dual `z`, and transformed `hat_x`.
- `TorchFlowerClient._fit_fedadmm(...)` returns `delta_hat` as the NumPy parameter list and places the stable client identity in the returned metrics dictionary under `fedadmm_client_id`.

- [ ] Add a failing tiny-linear-model test for `z_new = z_old + eta*(x_new-bar_x)` and `hat_x = x + z/eta`.
- [ ] Add a failing persistence test requiring `x`, `z`, and `hat_x` to survive a second fit.
- [ ] Add a failing local-steps test proving `fedadmm_local_steps` bounds updates independently of dataset length.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement a state schema containing model tensors, dual tensors, transformed tensors, and a version marker.
- [ ] Implement bounded SGD local minimization with the augmented Lagrangian and optional tolerance-based early stop.
- [ ] Load state per client, initialize from the first received consensus model, update/persist state, and return delta-hat payload plus client identity metadata.
- [ ] Preserve existing model parameter ordering and ensure tuple/list model outputs remain supported by the current training conventions.
- [ ] Run focused client/training tests and confirm they pass.

### Task 3: Implement server-side Algorithm 2 recurrence and proximal step

**Files:**
- Modify: `src/fl_baselines/algorithms/fedadmm.py`
- Modify: `src/fl_baselines/logging/checkpointing.py` only if state serialization needs a focused helper
- Test: `tests/test_model_and_algorithm.py`

**Interfaces:**
- `FedADMMStrategy` owns `tilde_parameters`, `bar_parameters`, and `client_hat_models` keyed by explicit client identity.
- `aggregate_fit` consumes delta-hat payloads and returns proximal consensus model parameters.

- [ ] Add a failing full-participation recurrence test where the server result equals the mean of client `hat_x` values.
- [ ] Add a failing partial-participation test proving unselected clients retain their old `hat_x` and the update scale is exactly `1/n`.
- [ ] Add a failing non-numeric Flower proxy-ID test proving payload identity, not `proxy.cid`, selects the cache entry.
- [ ] Add a failing non-zero-prox test proving `bar_x = prox_g(tilde_x)` before the next broadcast.
- [ ] Run these tests and confirm they fail against the current model+dual averaging implementation.
- [ ] Rewrite strategy aggregation to validate exact payload lengths/shapes and apply delta-hat updates only for selected clients.
- [ ] Resolve the configured prox operator once in the builder and apply it after every server aggregation, including initialization.
- [ ] Keep scalar client identity metadata out of weighted metric aggregation and retain checkpoint behavior for the consensus model.
- [ ] Add serializable server state under `output-dir/fedadmm_server/state.pt` with atomic-enough write ordering for normal run restarts.
- [ ] Run strategy tests and confirm they pass.

### Task 4: Wire builders, registry, documentation, and compatibility tests

**Files:**
- Modify: `src/fl_baselines/algorithms/fedadmm.py`
- Modify: `src/fl_baselines/defaults.py`
- Modify: `src/fl_baselines/algorithms/__init__.py` only if exports require it
- Modify: `docs/algorithms/fedadmm.md`
- Modify: `docs/algorithms/index.md`
- Modify: `docs/README.md`
- Modify: `docs/extending-baselines.md`
- Modify: `tests/test_registry.py`

- [ ] Add failing tests for canonical penalty fit config, prox config propagation, and registry availability.
- [ ] Implement builder wiring without adding algorithm-specific logic to server/client entrypoints.
- [ ] Document Algorithm 2 state, payload, prox choices, paper-compatible local-step configuration, and persisted artifacts.
- [ ] Run registry/config/documentation-focused checks and confirm they pass.

### Task 5: Verify model compatibility and end-to-end execution

**Files:**
- Test: `tests/test_model_and_algorithm.py`
- Test: `tests/test_client_app.py`
- Modify: `docs/testing-and-artifacts.md` if the new state artifacts need listing

- [ ] Add/extend smoke tests for `mnist_cnn`, `lenet`, `resnet9`, `resnet18`, `resnet34`, and `inception` with one local FedADMM step.
- [ ] Add a client/server resolution test using a non-numeric Flower proxy identity and a stable partition identity.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 python -m compileall src tests`.
- [ ] Run a one-round Flower smoke command with a small configured run if data is already available; do not download or launch a long experiment without approval.
- [ ] Inspect generated state/checkpoint paths and remove only generated `__pycache__` directories.
- [ ] Report exact commands, pass counts, and any environment limitation.

## Validation Plan

- Unit tests prove proximal formulas and primal/dual state equations.
- Strategy tests prove `1/n` delta aggregation and partial participation.
- Client tests prove state persistence and payload shape.
- Full suite and compile checks protect existing baselines.
- A minimal Flower smoke run proves registry → server app → client app → strategy integration.

## Risks

- Flower may not expose a stable proxy identity matching partition IDs; explicit payload metadata must remain authoritative.
- Existing `NumPyClient` payloads accept only NumPy arrays, so identity metadata will use the returned `FitRes.metrics` scalar dictionary rather than a reserved parameter-array slot.
- Arbitrary user-defined `g` functions cannot be serialized through `pyproject.toml`; the registry is the supported extension point.
- The paper's convergence assumptions do not automatically hold for neural-network cross-entropy; tests will verify equations, not theorem applicability.
