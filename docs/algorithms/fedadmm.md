# FedADMM

This repository implements Algorithm 2 from Wang, Marella, and Anderson,
“FedADMM: A Federated Primal-Dual Algorithm Allowing Partial Participation”.
The method supports the general composite objective `f(x) + g(x)` through a
server-side proximal operator and keeps the local neural-network minimization
inexact, as allowed by the paper.

## Files

- `src/fl_baselines/algorithms/fedadmm.py`: server `tilde_x`/`bar_x` state,
  partial-participation delta-hat aggregation, and server prox.
- `src/fl_baselines/training/fedadmm.py`: augmented-Lagrangian SGD solver and
  primal/dual/hat state updates.
- `src/fl_baselines/training/proximal.py`: proximal operator protocol and
  built-in operators.
- `src/fl_baselines/clients/torch_client.py`: per-client state persistence and
  FedADMM routing.

## Runtime configuration

```toml
algorithm = "fedadmm"
fedadmm-penalty = 1.0
fedadmm-prox = "identity"
fedadmm-local-steps = 300
fedadmm-tolerance = 0.0
batch-size = 2
learning-rate = 0.01
```

`fedadmm-alpha` remains accepted as a compatibility alias for
`fedadmm-penalty`. If both are provided, the canonical `fedadmm-penalty`
value wins.

Available proximal operators:

- `identity`: `g = 0`;
- `l1`: element-wise soft-thresholding, configured with
  `fedadmm-l1-weight`;
- `box`: projection to `[fedadmm-box-min, fedadmm-box-max]`.

## Algorithm state

For every configured client, the server tracks `hat_x_i`. Each client
persists `x_i`, `z_i`, and `hat_x_i` at:

```text
output-dir/fedadmm_clients/<client-id>/state.pt
```

The server restart state is stored at:

```text
output-dir/fedadmm_server/state.pt
```

The client sends `Delta hat_x_i` as its parameter payload and
`fedadmm_client_id` as fit metadata. The server applies the paper's
`1 / num_total_clients` update to `tilde_x`, then computes:

```text
bar_x = prox_{g / eta}(tilde_x)
```

Only selected clients update their state in a round; non-selected clients'
`hat_x_i` remains unchanged.

## Paper-compatible setup

The paper's FEMNIST experiments use 30 clients, 10 active clients, penalty
`eta = 1`, batch size 2, learning rate 0.01, and 300 local SGD iterations.
The repository does not add the paper's synthetic-data generator or FEMNIST
loader in this change, so those datasets must be provided through an existing
dataset builder or a future extension.

The convergence theorem's assumptions are mathematical conditions on the
client objectives and sampling scheme; passing the code tests does not by
itself establish those assumptions for a neural-network experiment.
