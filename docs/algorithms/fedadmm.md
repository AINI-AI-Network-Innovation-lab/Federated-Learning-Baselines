# FedADMM

FedADMM in this repository is a practical adaptation of the paper's partial-participation primal-dual idea for supervised classification workloads.

## Files

- `src/fl_baselines/algorithms/fedadmm.py`
- `src/fl_baselines/training/fedadmm.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "fedadmm"`
- `fedadmm-alpha = 1.0`
- `local-epochs`
- `learning-rate`

## Notes

- Each client persists its dual state under `output-dir/fedadmm_clients/<client-id>/state.pt`.
- The client local objective uses an augmented-Lagrangian style loss over the current model, the broadcast global model, and the persisted dual state.
- The server keeps a cache of transformed client models and averages that cache each round.
- This first pass intentionally supports the `g = 0` practical path only; it does not expose a generic proximal oracle for arbitrary constraints.
