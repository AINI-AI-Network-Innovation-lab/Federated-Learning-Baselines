# Faithful FedADMM Integration Design

## Goal

Integrate the full FedADMM algorithm from Wang, Marella, and Anderson's
`2203.15104v1.pdf` into this Flower/PyTorch repository, including the general
composite objective with a server-side proximal operator, exact partial-
participation state transitions, and reproducible local inexact solves.

## Scope

The implementation will support the paper's Algorithm 2 for neural-network
classification workloads. It will expose a pluggable proximal-operator
interface for the global term `g`, ship identity (`g = 0`), L1
soft-thresholding, and box projection operators, and preserve the existing
`fedadmm-alpha` configuration as a compatibility alias for the paper's
penalty parameter `eta`.

The implementation will not claim closed-form exact local minimization for
non-convex neural networks. Local minimization will be an explicit, bounded
SGD approximation with configurable steps and optional stopping tolerance.

## Paper-to-code mapping

| Paper quantity | Repository representation |
| --- | --- |
| `eta` | `ExperimentConfig.fedadmm_penalty`, alias `fedadmm_alpha` |
| `x_i` | Per-client persisted local model parameters |
| `z_i` | Per-client persisted dual tensors |
| `hat{x}_i = x_i + z_i/eta` | Per-client persisted transformed parameters |
| `tilde{x}` | Server strategy state |
| `bar{x}` | Server model parameters after `prox_g(tilde_x)` |
| `Delta hat{x}_i` | Client fit payload after the control metadata |
| `prox_{g/eta}` | Registered `ProxOperator` implementation |
| Proper subset sampling | Flower strategy's configured client sampling |

## Architecture

### Proximal operators

Add a focused proximal module with a protocol accepting a list of NumPy
arrays and returning a list of arrays. A registry will resolve an operator
from `ExperimentConfig` without changing the server strategy. Built-ins:

- `identity`: returns `x` unchanged;
- `l1`: applies element-wise soft-thresholding with configurable weight;
- `box`: clips each tensor element to configurable lower/upper bounds.

The operator receives the effective threshold/penalty and must preserve array
order, dtype, and shape. Invalid operator names or invalid parameters fail
during config validation.

### Server strategy

`FedADMMStrategy` will maintain:

- one `hat_x_i` cache per configured client identity;
- `tilde_x` as the running aggregate;
- the current consensus parameters `bar_x`.

On aggregation it will validate payload shape and client identity, update only
selected clients, compute:

```text
tilde_x <- tilde_x + (1 / n) * sum(delta_hat_i)
bar_x   <- prox_g(tilde_x)
```

and return `bar_x` as the next Flower global model. Client identity will come
from explicit payload metadata matching the client-side partition identity,
not from an assumed Flower proxy naming convention. Server state will be
serializable under the run output directory for restart-safe experiments.

### Client state and local solver

`TorchFlowerClient` will route `algorithm="fedadmm"` to a dedicated helper.
The helper will load or initialize `x_i`, `z_i`, and `hat_x_i`, set the
received `bar_x` as the current consensus reference, and optimize the local
augmented Lagrangian for a bounded number of SGD steps:

```text
CE(model(x), y)
+ <z_i, model - bar_x>
+ (eta / 2) * ||model - bar_x||^2
```

After the solve it will update `z_i`, derive the new `hat_x_i`, persist all
three state components, and return `Delta hat_x_i` plus non-aggregated client
identity metadata. The existing `local-epochs` behavior remains as a fallback
when the FedADMM-specific step count is omitted.

### Configuration

Add kebab-case runtime keys and typed fields for:

- `fedadmm-penalty` / compatibility alias `fedadmm-alpha`;
- `fedadmm-local-steps`;
- `fedadmm-tolerance`;
- `fedadmm-prox`;
- `fedadmm-l1-weight`;
- `fedadmm-box-min` and `fedadmm-box-max`.

Defaults reproduce the current `g=0` behavior while allowing the paper's
setup to be requested explicitly (`penalty=1`, batch size `2`, learning rate
`0.01`, and `local-steps=300`).

## Data flow

1. Server initializes `tilde_x`, all `hat_x_i = x_0`, and `bar_x = prox_g(tilde_x)`.
2. Flower samples a proper subset of clients.
3. Selected clients receive `bar_x` and load their persisted state.
4. Each selected client performs the inexact augmented-Lagrangian solve.
5. Each selected client updates/persists `z_i`, `x_i`, and `hat_x_i`.
6. The client returns `Delta hat_x_i` and its stable client identity.
7. The server applies the `1/n` delta update only for selected clients.
8. The server applies the configured proximal operator and broadcasts the new
   consensus model.

## Error handling

- Reject non-positive penalty, local steps, or tolerance.
- Reject unknown prox operators and invalid L1/box parameters during config
  validation.
- Reject duplicate or unknown client identities in server payloads.
- Reject malformed payloads whose tensor count, shape, or dtype does not match
  the model state.
- Keep metadata out of weighted metric aggregation.

## Testing and validation

Tests will cover:

- identity, L1, and box proximal operators;
- config parsing, alias precedence, and validation;
- exact dual and transformed-state equations on a tiny linear model;
- local payload shape and state persistence across multiple fits;
- server delta aggregation under full and partial participation;
- non-zero `g` changing `bar_x` through the proximal step;
- non-numeric Flower proxy IDs not changing client identity semantics;
- compatibility with all existing model builders;
- a minimal Flower smoke run with two clients and one or more rounds.

The existing full unit suite and `compileall` remain required gates. The
implementation will be described as faithful to Algorithm 2's state updates
and server proximal step, while local neural-network solves remain inexact as
allowed by the paper.

## Non-goals

- Reproducing the paper's synthetic-data generator or FEMNIST benchmark data
  loader in this change.
- Claiming the paper's convergence theorem without checking its assumptions
  for a particular neural-network experiment.
- Refactoring unrelated algorithms or the global client architecture.
