# FedGEN

`fedgen` implements a practical FedGEN integration for this repository by learning and aggregating a global feature mask alongside the normal model parameters.

## Files

- `src/fl_baselines/algorithms/fedgen.py`: `FedGENStrategy` and `FedGENBuilder`.
- `src/fl_baselines/training/fedgen.py`: local masked training loop and masked evaluation helper.
- `src/fl_baselines/training/features.py`: feature extraction, classifier projection, and final classifier weight helpers.
- `src/fl_baselines/clients/torch_client.py`: routes FedGEN fit/evaluate with the extra global mask payload.
- `src/fl_baselines/app/server_app.py`: applies masked server-side evaluation for FedGEN.

## Integration Shape

The paper works on sequential data and learns masks over input features. In this repository, the integration is adapted to the shared representation space so that existing models still work unchanged:

1. Each round, the server sends `model parameters + global mask`.
2. Each client extracts features with `extract_features(...)`.
3. The client applies `sigmoid(mask) * features` before the final classifier head.
4. Local training minimizes:
   - prediction loss
   - `L1` regularization on model parameters
   - a FedGEN-style penalty built from the classifier weight matrix and local loss gradient
5. The client updates its local mask from running mean/variance statistics of classifier-feature weights.
6. The server aggregates both model parameters and masks with the standard weighted average.

## Config

- `fedgen-alpha`: positive scaling factor for mask updates; default `1.5`
- `fedgen-lambda`: non-negative penalty coefficient; default `0.1`
- `fedgen-beta`: EMA coefficient for running mean; default `0.9`
- `fedgen-delta`: EMA coefficient for running variance; default `0.9`
- `fedgen-warmup-epochs`: number of local epochs before mask updates begin; default `1`
- `fedgen-l1-weight`: non-negative L1 regularization weight; default `1e-4`

## Run

```bash
flwr run . --run-config 'algorithm="fedgen" fedgen-alpha=1.5 fedgen-lambda=0.1 fedgen-warmup-epochs=1' --stream
```

## Notes

- This integration is faithful to the collaborative masking idea of the paper, but adapted from raw input masking to representation masking so it remains compatible with all current repo models.
- FedGEN evaluation also uses the aggregated global mask on both client-side and server-side evaluation.
- FedGEN does not create extra persisted client artifacts under `outputs/`; the mask is carried in the FL payload itself.
