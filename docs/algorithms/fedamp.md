# FedAMP

FedAMP trains personalized client models through attentive message passing. Instead of one global model for every client, the server keeps a personalized cloud model per client and updates each cloud model as a convex combination of known client models. Similar client models receive larger message weights.

## Files

- `src/fl_baselines/algorithms/fedamp.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Config

- `algorithm = "fedamp"`
- `fedamp-lambda = 0.1`
- `fedamp-alpha = 0.1`
- `fedamp-sigma = 1.0`
- `local-epochs`
- `learning-rate`

## Behavior

- The server overrides `configure_fit` so each client receives its own personalized cloud parameters when available.
- The client trains locally with a proximal term toward the received cloud model using coefficient `fedamp-lambda / fedamp-alpha`.
- The server stores the returned local models and recomputes personalized cloud models using the negative-exponential attention function from FedAMP.

## Notes

- This implementation covers FedAMP from the paper. HeurFedAMP, the cosine-similarity DNN heuristic, is left as a separate extension.
- The strategy returns the mean of participating personalized cloud models as the Flower-compatible server parameter object for checkpointing and server evaluation.
