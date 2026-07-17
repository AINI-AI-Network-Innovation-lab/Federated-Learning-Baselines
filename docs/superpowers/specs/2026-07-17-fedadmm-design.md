# FedADMM Design Spec

**Goal:** Add a practical FedADMM baseline to this Flower codebase, starting with the `g = 0` variant that is compatible with the existing classification pipeline and client-participation model.

**Recommended scope:** Implement the paper's core partial-participation primal-dual behavior, including per-client dual state persistence and an ADMM-style local objective, while deferring a fully generic proximal operator for arbitrary `g`.

## Problem Framing

The paper's FedADMM is a federated primal-dual method for composite optimization. In this repository, the nearest practical target is a FedADMM baseline that:

- keeps partial participation,
- persists a dual variable per client,
- updates local models with an ADMM-style objective,
- aggregates selected client updates on the server,
- uses the existing model/dataset abstractions unchanged.

This is a better fit than trying to expose a fully generic `g` oracle immediately, because the current repo is centered on supervised vision classification rather than general constrained optimization.

## Proposed Architecture

### Server-side

Add a new algorithm builder that is Flower-compatible and registers as `fedadmm`.

The server should:

- initialize the global model from the selected model builder,
- broadcast the current global model to selected clients,
- aggregate selected client updates with the existing weighted aggregation convention,
- keep FedADMM-specific state minimal on the server unless needed for payload compatibility.

If the implementation needs additional server payload beyond the model parameters, that payload should be explicit in the strategy interface rather than hidden in metrics.

### Client-side

Add a dedicated FedADMM training helper and a client-side state store keyed by `client_id`.

The client should:

- load the current global model parameters,
- load the persisted dual state for that client,
- optimize the ADMM-style local objective for a configurable number of local epochs,
- update and persist the dual state after training,
- return the updated model parameters to the server.

This mirrors the existing repo patterns for client-specific persistent state, especially `FedDyn`, `MOON`, and `SCAFFOLD`.

### Configuration

Add algorithm-specific config fields to `ExperimentConfig` for the FedADMM penalty and any local-solver controls needed by the training helper.

Minimum likely fields:

- `fedadmm_alpha`
- `fedadmm_local_solver_steps` or equivalent if local minimization is approximated

Keep defaults conservative so the baseline works out of the box on MNIST-style tasks.

## File Boundaries

### New files

- `src/fl_baselines/algorithms/fedadmm.py`
- `src/fl_baselines/training/fedadmm.py`
- `docs/algorithms/fedadmm.md`

### Files to modify

- `src/fl_baselines/clients/torch_client.py`
- `src/fl_baselines/core/config.py`
- `src/fl_baselines/defaults.py`
- `src/fl_baselines/algorithms/__init__.py` if needed for export style consistency
- `tests/test_registry.py`
- `tests/test_model_and_algorithm.py`
- `docs/algorithms/index.md`
- `docs/extending-baselines.md` if the algorithm list or extension guidance needs to mention FedADMM

## Data Flow

1. Server starts with the selected model and default Flower strategy wiring.
2. In each round, the strategy selects a subset of clients.
3. The server sends the current global model to those clients.
4. Each selected client loads its persisted FedADMM state and runs the local ADMM-style update.
5. The client saves the updated dual state and returns the updated model weights.
6. The server aggregates the selected client weights into the next global model.

## Testing Strategy

Add tests that verify:

- `fedadmm` is registered in the default algorithm registry,
- `TorchFlowerClient.fit()` routes to the FedADMM branch,
- client state persistence survives multiple calls for the same `client_id`,
- the FedADMM training helper produces the expected payload shape and metrics contract,
- a minimal end-to-end strategy construction works with the existing config and model builders.

## Risks and Constraints

- The paper's full `g`-prox formulation is not directly represented in the current supervised-learning pipeline.
- A faithful general-purpose ADMM implementation would require a new prox abstraction, which is out of scope for the first pass.
- The local solver may need to be approximate rather than closed-form because the repo trains neural networks, not convex objectives.
- Because of that, this baseline should be described as a practical FedADMM adaptation for this codebase, not as a complete solver for arbitrary constrained objectives.

## Acceptance Criteria

- `algorithm="fedadmm"` runs through the same Flower entrypoints as existing baselines.
- The client persists and reloads its dual state by `client_id`.
- FedADMM uses partial participation and remains compatible with the current model/dataset registry.
- The docs clearly state the supported scope and the gap to the paper's general `g` formulation.
