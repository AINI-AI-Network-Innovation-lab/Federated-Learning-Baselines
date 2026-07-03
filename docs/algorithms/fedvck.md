# FedVCK

FedVCK them client-side valuable knowledge condensation va server-side replay/update tren condensed knowledge, trong khi van giu model aggregation theo weighted average.

## Files

- `src/fl_baselines/algorithms/fedvck.py`
- `src/fl_baselines/training/fedvck.py`
- `src/fl_baselines/clients/torch_client.py`

## Runtime Flow

1. Server strategy gui global model cung cac hyperparameter condensation/replay xuong client.
2. Moi client train local model, tinh importance score theo prediction error, va chon mot condensed payload nho tu local data.
3. Client upload model parameters da train, condensed inputs/labels, va class-wise logit prototype sums/counts.
4. Server aggregate raw model parameters, cap nhat global logit prototypes, luu condensed memory theo round, roi replay tren memory bang cross-entropy va relational contrastive loss.

## Config

- `fedvck-condensed-ratio`: ti le local examples duoc dua vao condensed payload
- `fedvck-condensed-steps`: so buoc condensation noi bo
- `fedvck-condensed-learning-rate`: learning rate cho condensation helper
- `fedvck-importance-alpha`: he so lam muot prediction error giua current va previous model
- `fedvck-server-replay-epochs`: so epoch replay tren server moi round
- `fedvck-server-replay-learning-rate`: learning rate cho replay tren server
- `fedvck-contrastive-temperature`: nhiet do cho relational contrastive branch
- `fedvck-hard-negative-k`: so hard negative classes moi label
- `fedvck-enable-latent-constraints`: bat/tat latent distribution constraint theo kieu repo-fit
- `fedvck-max-memory-rounds`: gioi han so rounds condensed memory duoc giu tren server

## Artifacts

- `outputs/fedvck_clients/<client-id>/state.pt`: luu `previous_model_state` cho model-guided importance scoring

## Notes

- Day la ban tich hop theo huong faithful-while-fit-repo cho stack classification hien tai, khong phai full reproduction medical benchmark cua paper.
- Latent constraints duoc giu theo huong graceful degradation: neu model khong co du intermediate normalization structure, helper se fallback thay vi pha vo training path chung.
