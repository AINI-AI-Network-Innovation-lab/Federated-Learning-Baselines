# FedVCK Design

## Goal

Integrate the FedVCK algorithm from Yan et al. into the current Flower baseline repository as a first-class algorithm option that works with the existing classification datasets and model builders already supported by the codebase.

This integration should preserve the main method components described in the paper:

- client-side valuable knowledge condensation
- model-guided importance sampling
- upload of condensed knowledge and class-wise logit prototypes
- server-side global model update with supervised loss and relational contrastive loss

The implementation target is the current repository architecture, not the paper's exact medical benchmark stack.

## Scope

In scope:

- new `fedvck` algorithm registration and configuration
- client training path for FedVCK
- client state persisted under `output_dir`
- condensed knowledge payload transfer from clients to server
- server aggregation of model weights and uploaded prototype statistics
- server-side replay/update using condensed knowledge
- tests for config, registry, strategy behavior, client routing, and basic payload handling
- concise algorithm documentation

Out of scope for this iteration:

- adding paper-specific medical datasets
- reproducing full paper benchmark numbers
- building a separate experiment harness outside the current repo
- exact paper-architecture replication when it conflicts with the repo's model abstraction

## Constraints And Repo Fit

The current repository is built around:

- strategy builders in `src/fl_baselines/algorithms/`
- algorithm-gated client logic in `src/fl_baselines/clients/torch_client.py`
- reusable local training helpers in `src/fl_baselines/training/`
- runtime configuration in `src/fl_baselines/core/config.py` and `pyproject.toml`
- NumPy-array parameter transport through Flower

FedVCK does not fit into a pure FedAvg-style local training loop because it requires non-model payloads and server-side optimization on uploaded synthetic data. The design therefore treats FedVCK as a hybrid:

- client updates still return model weights for normal federated aggregation
- clients also return extra payload arrays representing condensed knowledge and prototype statistics
- the server consumes both the aggregated weights and the auxiliary payload to perform an additional replay/update step on the global model

This keeps the integration aligned with repo patterns already used by algorithms such as `fedproto`, `feddc`, `feddyn`, and `fedent`.

## High-Level Architecture

### 1. Algorithm Module

Add `src/fl_baselines/algorithms/fedvck.py`.

Responsibilities:

- define `FedVCKStrategy`
- define `FedVCKBuilder`
- prepare fit config for clients
- parse returned mixed payloads from clients
- aggregate class-wise logit prototype statistics
- run server-side replay/update on accumulated condensed knowledge
- save round checkpoints

`FedVCKStrategy` will extend `FedAvg` plus checkpointing behavior, similar to other custom algorithms in the repo.

### 2. Client Training Module

Add `src/fl_baselines/training/fedvck.py`.

Responsibilities:

- compute sample importance from current and previous model predictions
- perform model-guided sampling from local training data
- optimize a small synthetic dataset that matches selected local knowledge
- optionally enforce latent distribution constraints using batch-normalization statistics captured from real batches
- compute class-wise logit prototype sums and counts
- train the local model on real client data in the same round
- package condensed knowledge, labels, and prototype statistics into transportable arrays

This file should hold the algorithm-specific math and tensor handling rather than spreading it across the client wrapper.

### 3. Client Wrapper Integration

Extend `src/fl_baselines/clients/torch_client.py`.

Responsibilities:

- route `algorithm == "fedvck"` to a dedicated `_fit_fedvck`
- load and save per-client FedVCK state
- deserialize server-provided round context if needed
- return a mixed payload containing:
  - updated model parameters
  - condensed images/features
  - condensed labels
  - class-wise logit prototype sums
  - class-wise logit prototype counts

Per-client state lives under:

- `output_dir/fedvck_clients/<client_id>/state.pt`

This state will include at least:

- previous-round model snapshot or logits support needed for smoothed importance scoring
- any persistent synthetic-data initialization state if reused across rounds

### 4. Server Replay State

The server strategy will hold:

- current global parameters
- accumulated condensed knowledge memory across rounds
- aggregated global logit prototypes
- projector head parameters for relational contrastive learning if implemented as a persistent module

Accumulated condensed knowledge is needed because the paper's update objective uses the collected knowledge up to round `t`, not only the current round upload.

## Data Flow

### Round Start

The server sends:

- current global model parameters
- algorithm tag and local hyperparameters
- FedVCK hyperparameters for condensation and server-aware behavior

### On Each Client

For each selected client:

1. Load the global model.
2. Load previous FedVCK client state if present.
3. Score local examples using prediction error smoothed by current and previous model behavior.
4. Sample informative local batches using importance sampling.
5. Condense selected knowledge into a small synthetic set.
6. Record latent batch-normalization statistics from real data and reuse them during synthetic optimization when latent constraints are enabled.
7. Compute class-wise logit prototype sums and counts from local real data.
8. Run local model training for the round.
9. Return updated model parameters plus synthetic payload arrays and prototype statistics.

### On The Server

For each completed round:

1. Separate model parameters from FedVCK auxiliary arrays.
2. Aggregate client model parameters with weighted averaging.
3. Aggregate class-wise prototype statistics into global logit prototypes.
4. Append uploaded condensed knowledge into server memory.
5. Run server-side replay/update on the global model using:
   - cross-entropy on condensed knowledge
   - relational contrastive loss using class prototypes and hard negatives
6. Publish the replay-updated global model as the next round's parameters.

## FedVCK Components

### Client-Side Valuable Knowledge Condensation

The synthetic dataset will be represented as tensors plus integer labels. For the repo-fit implementation, each client produces a bounded amount of synthetic data per round using the local input shape of the configured dataset/model pair.

The condensation loop will:

- initialize synthetic samples from noise
- repeatedly sample real batches with class awareness
- match knowledge between real and synthetic batches
- optimize synthetic tensors directly for a fixed number of steps

The exact matching loss in the paper is distribution matching based. In the implementation, the design target is:

- match class-aware feature statistics in the model encoder space
- keep the loss formulation isolated so it can be improved later without changing the client contract

This preserves the algorithmic role of condensed knowledge while staying compatible with different models in the repo.

### Latent Distribution Constraints

When enabled, the client will:

- run real batches through the model
- capture batch-normalization-style mean and variance statistics from intermediate activations
- enforce those recorded statistics while embedding synthetic data during condensation

If a model does not expose enough normalization layers or intermediate structure cleanly, the helper will degrade gracefully:

- use the subset of supported normalization-bearing layers
- skip the constraint entirely if no compatible layers exist

This is an intentional repo-fit adaptation to preserve broad model compatibility.

### Model-Guided Knowledge Selection

Importance sampling will be based on cross-entropy error, smoothed across current and previous round behavior, following the paper's intent.

Client state therefore needs a previous model snapshot or equivalent prediction reference.

The implementation will compute per-example or per-batch importance scores, normalize them safely, and sample with replacement when necessary for small local datasets.

### Class-Wise Logit Prototypes

Clients compute class-wise sums and counts of pre-softmax logits over local real data.

The server aggregates these into global class-wise prototype vectors. These vectors are then used to identify hard negative classes for relational contrastive learning.

Using logits instead of hidden features matches the paper's prototype construction for the class relation signal and avoids requiring a stable shared feature dimension across every model family for this specific part.

### Server-Side Relational Contrastive Replay

The server replay step uses the accumulated condensed knowledge memory and the current model.

The update objective will include:

- cross-entropy loss on condensed samples
- relational contrastive loss that pushes each sample toward its own class relation target and away from hard negative classes

The replay helper should:

- derive current feature embeddings from the model
- maintain or build class-wise feature prototypes from the condensed knowledge memory
- select top-k hard negative classes using global logit prototypes
- optimize the global model for a small number of server replay epochs/steps each round

If a model does not expose a clean feature extractor, replay will use the repo's existing feature extraction helper patterns. Any unsupported model shape should fail clearly in tests rather than silently producing invalid behavior.

## Payload Format

Flower transports NumPy arrays. FedVCK therefore needs a deterministic payload layout.

Client return payload order:

1. model parameter arrays
2. condensed sample tensor array
3. condensed label array
4. prototype sum array
5. prototype count array

If extra metadata is needed, it should be encoded in metrics or as fixed-shape arrays rather than ad-hoc Python objects.

The server strategy must validate payload length and shapes before use.

## Configuration

Add a new algorithm key: `fedvck`.

Add FedVCK-specific config fields in `ExperimentConfig` and default values in `pyproject.toml`.

Planned config surface:

- `fedvck-condensed-ratio`
- `fedvck-condensed-steps`
- `fedvck-condensed-learning-rate`
- `fedvck-importance-alpha`
- `fedvck-server-replay-epochs`
- `fedvck-server-replay-learning-rate`
- `fedvck-contrastive-temperature`
- `fedvck-hard-negative-k`
- `fedvck-enable-latent-constraints`
- `fedvck-prototype-momentum` if smoothing is needed
- `fedvck-max-memory-rounds` or equivalent memory cap

Validation rules:

- ratios and learning rates must be positive
- `importance-alpha` must be in `[0, 1]`
- integer step counts must be positive
- `hard-negative-k` must be less than the number of classes when classes are known

## Testing Strategy

Tests should be added before implementation for the missing behavior.

Minimum coverage:

- config parsing and validation for new FedVCK keys
- registry/default registration includes `fedvck`
- builder creates a strategy successfully
- strategy fit config exposes expected FedVCK keys
- strategy aggregates mixed client payloads correctly
- client routing sends `fedvck` to the dedicated path
- local FedVCK helper returns finite synthetic/prototype payloads
- server replay step mutates the global model in a controlled way
- compatibility smoke tests across current supported models where feasible

Tests should stay lightweight and use synthetic toy tensors rather than full experiments.

## Documentation

Add:

- `docs/algorithms/fedvck.md`

Update:

- `README.md`
- `docs/algorithms/index.md`
- `docs/README.md`
- `docs/overview.md`
- `docs/testing-and-artifacts.md`
- `docs/extending-baselines.md`

Docs should clearly state that this is a repo integration of FedVCK for the current classification stack, not a full reproduction of the paper's medical dataset benchmark environment.

## Risks And Mitigations

### Payload Size

Condensed samples can make client payloads much larger than weight-only FL.

Mitigation:

- keep condensed ratio configurable and conservative by default
- add memory caps on server accumulation
- test payload shape assumptions carefully

### Model Compatibility

Intermediate-feature and latent-statistics hooks may vary by model.

Mitigation:

- use shared helper functions for feature extraction
- degrade gracefully when latent constraints are unsupported
- keep failures explicit for unsupported configurations

### Server-Side Replay Complexity

Server optimization adds state and can destabilize aggregation if not bounded.

Mitigation:

- keep replay epochs small by default
- isolate replay code in a helper module
- test for finite loss and parameter updates

### Faithfulness Gaps

Some paper details may not map perfectly onto the repo abstraction.

Mitigation:

- preserve the core algorithmic roles of each component
- document every repo-fit adaptation
- keep helper boundaries clean so future refinement is easy

## Acceptance Criteria

The design is complete when:

- `fedvck` can be selected like existing algorithms
- clients can produce condensed-knowledge and prototype payloads
- the server can aggregate those payloads and run replay updates
- the full test suite passes
- documentation explains the new algorithm and its artifacts

## Repo-Fit Decisions

The following implementation decisions are intentional:

- target existing classification datasets and model builders only
- treat synthetic knowledge as array payloads carried through Flower parameters/results
- use graceful degradation for latent constraints on models without suitable intermediate normalization structure
- implement the paper's full component set, but modularize it around the repo's strategy/client/training architecture
