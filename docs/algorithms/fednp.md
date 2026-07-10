# FedNP

FedNP in this repository is integrated as a latent Gaussian regularizer over representation features. The server maintains global latent mean and variance, broadcasts them with the model, and each client returns updated model parameters plus latent sufficient statistics. This keeps the core intuition from the paper: explicitly estimate a global data distribution and use it to regularize local training under non-IID partitions.

## Files Chinh

- `src/fl_baselines/algorithms/fednp.py`
- `src/fl_baselines/training/fednp.py`
- `src/fl_baselines/training/features.py`
- `src/fl_baselines/clients/torch_client.py`
  - route `algorithm == "fednp"`

## Config

- `algorithm = "fednp"`
- `fednp-lambda = 0.1`
- `fednp-prior-variance = 1.0`
- `fednp-stability-eps = 1e-6`

## Hanh Vi

Moi round:

1. Server sends global model parameters together with latent Gaussian stats `(mean, variance)`.
2. Client extracts penultimate features for each batch.
3. Client optimizes:
   - classification loss
   - `+ fednp_lambda * KL(local_latent || global_latent)`
4. Client uploads:
   - updated model parameters
   - latent feature sums
   - latent squared-feature sums
   - latent sample count
5. Server aggregates model parameters with FedAvg-style weighted averaging and updates latent Gaussian stats by moment matching.

## Chay Nhanh

```bash
flwr run . --run-config 'algorithm="fednp" fednp-lambda=0.1 fednp-prior-variance=1.0' --stream
```
