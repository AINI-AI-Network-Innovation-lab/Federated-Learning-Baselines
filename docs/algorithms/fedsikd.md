# FedSiKD

`FedSiKD` implements *FedSiKD: Clients Similarity and Knowledge Distillation: Addressing Non-i.i.d. and Constraints in Federated Learning* as a clustered FedAvg-style baseline.

## Files

- `src/fl_baselines/algorithms/fedsikd.py`: server-side clustering strategy and builder
- `src/fl_baselines/training/fedsikd.py`: local client statistics and teacher-student distillation helper
- `src/fl_baselines/clients/torch_client.py`: routes `algorithm="fedsikd"` to the dedicated trainer

## Behavior

1. Each client computes simple dataset statistics from its local train loader: mean, standard deviation, and skewness.
2. The server clusters participating clients from those statistics using a deterministic numpy k-means implementation.
3. Each client receives the current global student model plus a cluster teacher snapshot.
4. Local training minimizes cross-entropy plus a soft distillation loss against the teacher.
5. The server aggregates client updates cluster-wise and then averages cluster teachers to form the next global student model.

This implementation keeps the Flower app entrypoints generic and stores no extra client state on disk. The only persisted artifacts are the normal round checkpoints written by the shared checkpoint helper.

## Config

- `algorithm = "fedsikd"`
- `fedsikd-num-clusters = 0`
- `fedsikd-max-clusters = 5`
- `fedsikd-kd-alpha = 0.5`
- `fedsikd-kd-temperature = 1.0`

If `fedsikd-num-clusters` is `0`, the server selects the cluster count automatically from the client statistics with silhouette, Calinski-Harabasz, and Davies-Bouldin style scores.

## Notes

- The paper describes leader-client selection as resource-aware; this repo uses cluster-average teacher snapshots because it fits the current Flower wiring better.
- The baseline is most meaningful with `fraction-train = 1.0`, because the clustering step is derived from participating clients in the current round.
