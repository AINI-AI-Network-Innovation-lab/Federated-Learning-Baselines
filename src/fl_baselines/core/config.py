"""Typed experiment configuration parsed from Flower run config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RunConfig = dict[str, bool | float | int | str]


def _value(config: RunConfig, key: str, default: Any) -> Any:
    return config.get(key, config.get(key.replace("-", "_"), default))


def _as_int(config: RunConfig, key: str, default: int) -> int:
    value = _value(config, key, default)
    return int(value)


def _as_float(config: RunConfig, key: str, default: float) -> float:
    value = _value(config, key, default)
    return float(value)


def _as_str(config: RunConfig, key: str, default: str) -> str:
    value = _value(config, key, default)
    return str(value)


def _as_bool(config: RunConfig, key: str, default: bool) -> bool:
    value = _value(config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = "fedavg"
    dataset: str = "mnist"
    model: str = "mnist_cnn"
    num_server_rounds: int = 3
    num_supernodes: int = 10
    fraction_train: float = 1.0
    fraction_evaluate: float = 1.0
    client_test_fraction: float = 0.2
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    proximal_mu: float = 0.1
    moon_mu: float = 1.0
    moon_temperature: float = 0.5
    server_learning_rate: float = 1.0
    server_momentum: float = 0.9
    fedadp_alpha: float = 5.0
    feddyn_alpha: float = 0.1
    feddc_alpha: float = 0.01
    feddecorr_beta: float = 0.1
    fedexp_epsilon: float = 0.001
    fedspeed_lambda: float = 0.1
    fedspeed_alpha: float = 1.0
    fedspeed_rho: float = 0.1
    fedsam_rho: float = 0.5
    fedgen_alpha: float = 1.5
    fedgen_lambda: float = 0.1
    fedgen_beta: float = 0.9
    fedgen_delta: float = 0.9
    fedgen_warmup_epochs: int = 1
    fedgen_l1_weight: float = 0.0001
    gamf_sigma: float = 2.0
    gamf_initial_tau: float = 0.05
    gamf_descent_factor: float = 0.9
    gamf_min_tau: float = 0.005
    gamf_max_iters: int = 200
    fedma_matching_epsilon: float = 0.0
    fedcda_memory_size: int = 3
    fedcda_num_batches: int = 3
    fedcda_warmup_rounds: int = 50
    fedcda_loss_weight: float = 1.0
    feddrl_actor_learning_rate: float = 0.0001
    feddrl_critic_learning_rate: float = 0.001
    feddrl_discount_factor: float = 0.99
    feddrl_target_tau: float = 0.02
    feddrl_hidden_size: int = 256
    feddrl_replay_buffer_size: int = 100000
    feddrl_batch_size: int = 32
    feddrl_updates_per_round: int = 1
    feddrl_noise_scale: float = 0.1
    feddrl_std_scale: float = 0.5
    fedent_beta: float = 0.99
    fedent_gamma: float = 0.99
    fedent_epsilon: float = 1e-8
    fedent_fixed_point_steps: int = 1
    fedent_max_learning_rate: float = 1.0
    fedent_enable_decay: bool = True
    fedlaw_server_epochs: int = 1
    fedlaw_server_learning_rate: float = 0.01
    fedlaw_gamma_init: float = 1.0
    fedaaw_beta: float = 0.01
    fedaaw_gamma: float = 1.0
    fedaaw_epsilon: float = 1e-8
    feddisco_discrepancy_weight: float = 0.5
    feddisco_bias: float = 0.1
    feddisco_metric: str = "kl"
    feddisco_epsilon: float = 1e-8
    fedvck_condensed_ratio: float = 0.01
    fedvck_condensed_steps: int = 1
    fedvck_condensed_learning_rate: float = 0.1
    fedvck_importance_alpha: float = 0.5
    fedvck_server_replay_epochs: int = 1
    fedvck_server_replay_learning_rate: float = 0.01
    fedvck_contrastive_temperature: float = 0.1
    fedvck_hard_negative_k: int = 1
    fedvck_enable_latent_constraints: bool = True
    fedvck_max_memory_rounds: int = 4
    fedproto_lambda: float = 1.0
    fedntd_beta: float = 1.0
    fedntd_temperature: float = 1.0
    ditto_lambda: float = 0.1
    pfedme_lambda: float = 15.0
    pfedme_beta: float = 1.0
    pfedme_personal_learning_rate: float = 0.01
    pfedme_personal_steps: int = 5
    fednova_server_momentum: float = 0.0
    fedper_personal_layers: int = 1
    fedrep_personal_layers: int = 1
    fedrep_representation_epochs: int = 1
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    num_classes: int = 10
    partitioner: str = "iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs"
    device: str = "cpu"
    emnist_split: str = "balanced"

    @classmethod
    def from_run_config(cls, run_config: RunConfig) -> "ExperimentConfig":
        config = cls(
            algorithm=_as_str(run_config, "algorithm", cls.algorithm),
            dataset=_as_str(run_config, "dataset", cls.dataset),
            model=_as_str(run_config, "model", cls.model),
            num_server_rounds=_as_int(
                run_config, "num-server-rounds", cls.num_server_rounds
            ),
            num_supernodes=_as_int(run_config, "num-supernodes", cls.num_supernodes),
            fraction_train=_as_float(run_config, "fraction-train", cls.fraction_train),
            fraction_evaluate=_as_float(
                run_config, "fraction-evaluate", cls.fraction_evaluate
            ),
            client_test_fraction=_as_float(
                run_config,
                "client-test-fraction",
                cls.client_test_fraction,
            ),
            local_epochs=_as_int(run_config, "local-epochs", cls.local_epochs),
            batch_size=_as_int(run_config, "batch-size", cls.batch_size),
            learning_rate=_as_float(run_config, "learning-rate", cls.learning_rate),
            proximal_mu=_as_float(run_config, "proximal-mu", cls.proximal_mu),
            moon_mu=_as_float(run_config, "moon-mu", cls.moon_mu),
            moon_temperature=_as_float(
                run_config, "moon-temperature", cls.moon_temperature
            ),
            server_learning_rate=_as_float(
                run_config,
                "server-learning-rate",
                cls.server_learning_rate,
            ),
            server_momentum=_as_float(
                run_config,
                "server-momentum",
                cls.server_momentum,
            ),
            fedadp_alpha=_as_float(
                run_config,
                "fedadp-alpha",
                cls.fedadp_alpha,
            ),
            feddyn_alpha=_as_float(
                run_config,
                "feddyn-alpha",
                cls.feddyn_alpha,
            ),
            feddc_alpha=_as_float(
                run_config,
                "feddc-alpha",
                cls.feddc_alpha,
            ),
            feddecorr_beta=_as_float(
                run_config,
                "feddecorr-beta",
                cls.feddecorr_beta,
            ),
            fedexp_epsilon=_as_float(
                run_config,
                "fedexp-epsilon",
                cls.fedexp_epsilon,
            ),
            fedspeed_lambda=_as_float(
                run_config,
                "fedspeed-lambda",
                cls.fedspeed_lambda,
            ),
            fedspeed_alpha=_as_float(
                run_config,
                "fedspeed-alpha",
                cls.fedspeed_alpha,
            ),
            fedspeed_rho=_as_float(
                run_config,
                "fedspeed-rho",
                cls.fedspeed_rho,
            ),
            fedsam_rho=_as_float(
                run_config,
                "fedsam-rho",
                cls.fedsam_rho,
            ),
            fedgen_alpha=_as_float(
                run_config,
                "fedgen-alpha",
                cls.fedgen_alpha,
            ),
            fedgen_lambda=_as_float(
                run_config,
                "fedgen-lambda",
                cls.fedgen_lambda,
            ),
            fedgen_beta=_as_float(
                run_config,
                "fedgen-beta",
                cls.fedgen_beta,
            ),
            fedgen_delta=_as_float(
                run_config,
                "fedgen-delta",
                cls.fedgen_delta,
            ),
            fedgen_warmup_epochs=_as_int(
                run_config,
                "fedgen-warmup-epochs",
                cls.fedgen_warmup_epochs,
            ),
            fedgen_l1_weight=_as_float(
                run_config,
                "fedgen-l1-weight",
                cls.fedgen_l1_weight,
            ),
            gamf_sigma=_as_float(
                run_config,
                "gamf-sigma",
                cls.gamf_sigma,
            ),
            gamf_initial_tau=_as_float(
                run_config,
                "gamf-initial-tau",
                cls.gamf_initial_tau,
            ),
            gamf_descent_factor=_as_float(
                run_config,
                "gamf-descent-factor",
                cls.gamf_descent_factor,
            ),
            gamf_min_tau=_as_float(
                run_config,
                "gamf-min-tau",
                cls.gamf_min_tau,
            ),
            gamf_max_iters=_as_int(
                run_config,
                "gamf-max-iters",
                cls.gamf_max_iters,
            ),
            fedma_matching_epsilon=_as_float(
                run_config,
                "fedma-matching-epsilon",
                cls.fedma_matching_epsilon,
            ),
            fedcda_memory_size=_as_int(
                run_config,
                "fedcda-memory-size",
                cls.fedcda_memory_size,
            ),
            fedcda_num_batches=_as_int(
                run_config,
                "fedcda-num-batches",
                cls.fedcda_num_batches,
            ),
            fedcda_warmup_rounds=_as_int(
                run_config,
                "fedcda-warmup-rounds",
                cls.fedcda_warmup_rounds,
            ),
            fedcda_loss_weight=_as_float(
                run_config,
                "fedcda-loss-weight",
                cls.fedcda_loss_weight,
            ),
            feddrl_actor_learning_rate=_as_float(
                run_config,
                "feddrl-actor-learning-rate",
                cls.feddrl_actor_learning_rate,
            ),
            feddrl_critic_learning_rate=_as_float(
                run_config,
                "feddrl-critic-learning-rate",
                cls.feddrl_critic_learning_rate,
            ),
            feddrl_discount_factor=_as_float(
                run_config,
                "feddrl-discount-factor",
                cls.feddrl_discount_factor,
            ),
            feddrl_target_tau=_as_float(
                run_config,
                "feddrl-target-tau",
                cls.feddrl_target_tau,
            ),
            feddrl_hidden_size=_as_int(
                run_config,
                "feddrl-hidden-size",
                cls.feddrl_hidden_size,
            ),
            feddrl_replay_buffer_size=_as_int(
                run_config,
                "feddrl-replay-buffer-size",
                cls.feddrl_replay_buffer_size,
            ),
            feddrl_batch_size=_as_int(
                run_config,
                "feddrl-batch-size",
                cls.feddrl_batch_size,
            ),
            feddrl_updates_per_round=_as_int(
                run_config,
                "feddrl-updates-per-round",
                cls.feddrl_updates_per_round,
            ),
            feddrl_noise_scale=_as_float(
                run_config,
                "feddrl-noise-scale",
                cls.feddrl_noise_scale,
            ),
            feddrl_std_scale=_as_float(
                run_config,
                "feddrl-std-scale",
                cls.feddrl_std_scale,
            ),
            fedent_beta=_as_float(
                run_config,
                "fedent-beta",
                cls.fedent_beta,
            ),
            fedent_gamma=_as_float(
                run_config,
                "fedent-gamma",
                cls.fedent_gamma,
            ),
            fedent_epsilon=_as_float(
                run_config,
                "fedent-epsilon",
                cls.fedent_epsilon,
            ),
            fedent_fixed_point_steps=_as_int(
                run_config,
                "fedent-fixed-point-steps",
                cls.fedent_fixed_point_steps,
            ),
            fedent_max_learning_rate=_as_float(
                run_config,
                "fedent-max-learning-rate",
                cls.fedent_max_learning_rate,
            ),
            fedent_enable_decay=_as_bool(
                run_config,
                "fedent-enable-decay",
                cls.fedent_enable_decay,
            ),
            fedlaw_server_epochs=_as_int(
                run_config,
                "fedlaw-server-epochs",
                cls.fedlaw_server_epochs,
            ),
            fedlaw_server_learning_rate=_as_float(
                run_config,
                "fedlaw-server-learning-rate",
                cls.fedlaw_server_learning_rate,
            ),
            fedlaw_gamma_init=_as_float(
                run_config,
                "fedlaw-gamma-init",
                cls.fedlaw_gamma_init,
            ),
            fedaaw_beta=_as_float(
                run_config,
                "fedaaw-beta",
                cls.fedaaw_beta,
            ),
            fedaaw_gamma=_as_float(
                run_config,
                "fedaaw-gamma",
                cls.fedaaw_gamma,
            ),
            fedaaw_epsilon=_as_float(
                run_config,
                "fedaaw-epsilon",
                cls.fedaaw_epsilon,
            ),
            feddisco_discrepancy_weight=_as_float(
                run_config,
                "feddisco-discrepancy-weight",
                cls.feddisco_discrepancy_weight,
            ),
            feddisco_bias=_as_float(
                run_config,
                "feddisco-bias",
                cls.feddisco_bias,
            ),
            feddisco_metric=_as_str(
                run_config,
                "feddisco-metric",
                cls.feddisco_metric,
            ),
            feddisco_epsilon=_as_float(
                run_config,
                "feddisco-epsilon",
                cls.feddisco_epsilon,
            ),
            fedvck_condensed_ratio=_as_float(
                run_config,
                "fedvck-condensed-ratio",
                cls.fedvck_condensed_ratio,
            ),
            fedvck_condensed_steps=_as_int(
                run_config,
                "fedvck-condensed-steps",
                cls.fedvck_condensed_steps,
            ),
            fedvck_condensed_learning_rate=_as_float(
                run_config,
                "fedvck-condensed-learning-rate",
                cls.fedvck_condensed_learning_rate,
            ),
            fedvck_importance_alpha=_as_float(
                run_config,
                "fedvck-importance-alpha",
                cls.fedvck_importance_alpha,
            ),
            fedvck_server_replay_epochs=_as_int(
                run_config,
                "fedvck-server-replay-epochs",
                cls.fedvck_server_replay_epochs,
            ),
            fedvck_server_replay_learning_rate=_as_float(
                run_config,
                "fedvck-server-replay-learning-rate",
                cls.fedvck_server_replay_learning_rate,
            ),
            fedvck_contrastive_temperature=_as_float(
                run_config,
                "fedvck-contrastive-temperature",
                cls.fedvck_contrastive_temperature,
            ),
            fedvck_hard_negative_k=_as_int(
                run_config,
                "fedvck-hard-negative-k",
                cls.fedvck_hard_negative_k,
            ),
            fedvck_enable_latent_constraints=_as_bool(
                run_config,
                "fedvck-enable-latent-constraints",
                cls.fedvck_enable_latent_constraints,
            ),
            fedvck_max_memory_rounds=_as_int(
                run_config,
                "fedvck-max-memory-rounds",
                cls.fedvck_max_memory_rounds,
            ),
            fedproto_lambda=_as_float(
                run_config,
                "fedproto-lambda",
                cls.fedproto_lambda,
            ),
            fedntd_beta=_as_float(
                run_config,
                "fedntd-beta",
                cls.fedntd_beta,
            ),
            fedntd_temperature=_as_float(
                run_config,
                "fedntd-temperature",
                cls.fedntd_temperature,
            ),
            ditto_lambda=_as_float(
                run_config,
                "ditto-lambda",
                cls.ditto_lambda,
            ),
            pfedme_lambda=_as_float(
                run_config,
                "pfedme-lambda",
                cls.pfedme_lambda,
            ),
            pfedme_beta=_as_float(
                run_config,
                "pfedme-beta",
                cls.pfedme_beta,
            ),
            pfedme_personal_learning_rate=_as_float(
                run_config,
                "pfedme-personal-learning-rate",
                cls.pfedme_personal_learning_rate,
            ),
            pfedme_personal_steps=_as_int(
                run_config,
                "pfedme-personal-steps",
                cls.pfedme_personal_steps,
            ),
            fednova_server_momentum=_as_float(
                run_config,
                "fednova-server-momentum",
                cls.fednova_server_momentum,
            ),
            fedper_personal_layers=_as_int(
                run_config,
                "fedper-personal-layers",
                cls.fedper_personal_layers,
            ),
            fedrep_personal_layers=_as_int(
                run_config,
                "fedrep-personal-layers",
                cls.fedrep_personal_layers,
            ),
            fedrep_representation_epochs=_as_int(
                run_config,
                "fedrep-representation-epochs",
                cls.fedrep_representation_epochs,
            ),
            input_channels=_as_int(run_config, "input-channels", cls.input_channels),
            input_height=_as_int(run_config, "input-height", cls.input_height),
            input_width=_as_int(run_config, "input-width", cls.input_width),
            num_classes=_as_int(run_config, "num-classes", cls.num_classes),
            partitioner=_as_str(run_config, "partitioner", cls.partitioner),
            dirichlet_alpha=_as_float(
                run_config, "dirichlet-alpha", cls.dirichlet_alpha
            ),
            seed=_as_int(run_config, "seed", cls.seed),
            data_dir=_as_str(run_config, "data-dir", cls.data_dir),
            output_dir=_as_str(run_config, "output-dir", cls.output_dir),
            device=_as_str(run_config, "device", cls.device),
            emnist_split=_as_str(run_config, "emnist-split", cls.emnist_split),
        )
        config.validate()
        return config

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_channels, self.input_height, self.input_width)

    def validate(self) -> None:
        if self.num_server_rounds <= 0:
            raise ValueError("num-server-rounds must be positive")
        if self.num_supernodes <= 0:
            raise ValueError("num-supernodes must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local-epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning-rate must be positive")
        if self.proximal_mu < 0:
            raise ValueError("proximal-mu must be non-negative")
        if self.moon_mu < 0:
            raise ValueError("moon-mu must be non-negative")
        if self.moon_temperature <= 0:
            raise ValueError("moon-temperature must be positive")
        if self.server_learning_rate <= 0:
            raise ValueError("server-learning-rate must be positive")
        if self.server_momentum < 0:
            raise ValueError("server-momentum must be non-negative")
        if self.fedadp_alpha <= 0:
            raise ValueError("fedadp-alpha must be positive")
        if self.feddyn_alpha <= 0:
            raise ValueError("feddyn-alpha must be positive")
        if self.feddc_alpha <= 0:
            raise ValueError("feddc-alpha must be positive")
        if self.feddecorr_beta < 0:
            raise ValueError("feddecorr-beta must be non-negative")
        if self.fedexp_epsilon < 0:
            raise ValueError("fedexp-epsilon must be non-negative")
        if self.fedspeed_lambda <= 0:
            raise ValueError("fedspeed-lambda must be positive")
        if not 0 <= self.fedspeed_alpha <= 1:
            raise ValueError("fedspeed-alpha must be in [0, 1]")
        if self.fedspeed_rho < 0:
            raise ValueError("fedspeed-rho must be non-negative")
        if self.fedsam_rho < 0:
            raise ValueError("fedsam-rho must be non-negative")
        if self.fedgen_alpha <= 0:
            raise ValueError("fedgen-alpha must be positive")
        if self.fedgen_lambda < 0:
            raise ValueError("fedgen-lambda must be non-negative")
        if not 0 <= self.fedgen_beta <= 1:
            raise ValueError("fedgen-beta must be in [0, 1]")
        if not 0 <= self.fedgen_delta <= 1:
            raise ValueError("fedgen-delta must be in [0, 1]")
        if self.fedgen_warmup_epochs < 0:
            raise ValueError("fedgen-warmup-epochs must be non-negative")
        if self.fedgen_l1_weight < 0:
            raise ValueError("fedgen-l1-weight must be non-negative")
        if self.gamf_sigma <= 0:
            raise ValueError("gamf-sigma must be positive")
        if self.gamf_initial_tau <= 0:
            raise ValueError("gamf-initial-tau must be positive")
        if not 0 < self.gamf_descent_factor <= 1:
            raise ValueError("gamf-descent-factor must be in (0, 1]")
        if self.gamf_min_tau <= 0:
            raise ValueError("gamf-min-tau must be positive")
        if self.gamf_max_iters <= 0:
            raise ValueError("gamf-max-iters must be positive")
        if self.fedma_matching_epsilon < 0:
            raise ValueError("fedma-matching-epsilon must be non-negative")
        if self.fedcda_memory_size <= 0:
            raise ValueError("fedcda-memory-size must be positive")
        if self.fedcda_num_batches <= 0:
            raise ValueError("fedcda-num-batches must be positive")
        if self.fedcda_warmup_rounds < 0:
            raise ValueError("fedcda-warmup-rounds must be non-negative")
        if self.fedcda_loss_weight <= 0:
            raise ValueError("fedcda-loss-weight must be positive")
        if self.feddrl_actor_learning_rate <= 0:
            raise ValueError("feddrl-actor-learning-rate must be positive")
        if self.feddrl_critic_learning_rate <= 0:
            raise ValueError("feddrl-critic-learning-rate must be positive")
        if not 0 < self.feddrl_discount_factor < 1:
            raise ValueError("feddrl-discount-factor must be in (0, 1)")
        if not 0 < self.feddrl_target_tau <= 1:
            raise ValueError("feddrl-target-tau must be in (0, 1]")
        if self.feddrl_hidden_size <= 0:
            raise ValueError("feddrl-hidden-size must be positive")
        if self.feddrl_replay_buffer_size <= 0:
            raise ValueError("feddrl-replay-buffer-size must be positive")
        if self.feddrl_batch_size <= 0:
            raise ValueError("feddrl-batch-size must be positive")
        if self.feddrl_updates_per_round < 0:
            raise ValueError("feddrl-updates-per-round must be non-negative")
        if self.feddrl_noise_scale < 0:
            raise ValueError("feddrl-noise-scale must be non-negative")
        if not 0 < self.feddrl_std_scale <= 1:
            raise ValueError("feddrl-std-scale must be in (0, 1]")
        if not 0 < self.fedent_beta < 1:
            raise ValueError("fedent-beta must be in (0, 1)")
        if not 0 <= self.fedent_gamma < 1:
            raise ValueError("fedent-gamma must be in [0, 1)")
        if self.fedent_epsilon <= 0:
            raise ValueError("fedent-epsilon must be positive")
        if self.fedent_fixed_point_steps <= 0:
            raise ValueError("fedent-fixed-point-steps must be positive")
        if self.fedent_max_learning_rate <= 0:
            raise ValueError("fedent-max-learning-rate must be positive")
        if self.fedlaw_server_epochs <= 0:
            raise ValueError("fedlaw-server-epochs must be positive")
        if self.fedlaw_server_learning_rate <= 0:
            raise ValueError("fedlaw-server-learning-rate must be positive")
        if self.fedlaw_gamma_init <= 0:
            raise ValueError("fedlaw-gamma-init must be positive")
        if self.fedaaw_beta <= 0:
            raise ValueError("fedaaw-beta must be positive")
        if self.fedaaw_gamma < 0:
            raise ValueError("fedaaw-gamma must be non-negative")
        if self.fedaaw_epsilon <= 0:
            raise ValueError("fedaaw-epsilon must be positive")
        if self.feddisco_discrepancy_weight < 0:
            raise ValueError("feddisco-discrepancy-weight must be non-negative")
        if self.feddisco_bias < 0:
            raise ValueError("feddisco-bias must be non-negative")
        if self.feddisco_metric not in {"kl", "l1", "l2", "cosine"}:
            raise ValueError("feddisco-metric must be one of: kl, l1, l2, cosine")
        if self.feddisco_epsilon <= 0:
            raise ValueError("feddisco-epsilon must be positive")
        if self.fedvck_condensed_ratio <= 0:
            raise ValueError("fedvck-condensed-ratio must be positive")
        if self.fedvck_condensed_steps <= 0:
            raise ValueError("fedvck-condensed-steps must be positive")
        if self.fedvck_condensed_learning_rate <= 0:
            raise ValueError("fedvck-condensed-learning-rate must be positive")
        if not 0 <= self.fedvck_importance_alpha <= 1:
            raise ValueError("fedvck-importance-alpha must be in [0, 1]")
        if self.fedvck_server_replay_epochs <= 0:
            raise ValueError("fedvck-server-replay-epochs must be positive")
        if self.fedvck_server_replay_learning_rate <= 0:
            raise ValueError("fedvck-server-replay-learning-rate must be positive")
        if self.fedvck_contrastive_temperature <= 0:
            raise ValueError("fedvck-contrastive-temperature must be positive")
        if self.fedvck_hard_negative_k <= 0:
            raise ValueError("fedvck-hard-negative-k must be positive")
        if self.fedvck_max_memory_rounds <= 0:
            raise ValueError("fedvck-max-memory-rounds must be positive")
        if self.fedproto_lambda < 0:
            raise ValueError("fedproto-lambda must be non-negative")
        if self.fedntd_beta < 0:
            raise ValueError("fedntd-beta must be non-negative")
        if self.fedntd_temperature <= 0:
            raise ValueError("fedntd-temperature must be positive")
        if self.ditto_lambda < 0:
            raise ValueError("ditto-lambda must be non-negative")
        if self.pfedme_lambda <= 0:
            raise ValueError("pfedme-lambda must be positive")
        if self.pfedme_beta <= 0:
            raise ValueError("pfedme-beta must be positive")
        if self.pfedme_beta > 1:
            raise ValueError("pfedme-beta must be in (0, 1]")
        if self.pfedme_personal_learning_rate <= 0:
            raise ValueError("pfedme-personal-learning-rate must be positive")
        if self.pfedme_personal_steps <= 0:
            raise ValueError("pfedme-personal-steps must be positive")
        if self.fednova_server_momentum < 0:
            raise ValueError("fednova-server-momentum must be non-negative")
        if self.fedper_personal_layers <= 0:
            raise ValueError("fedper-personal-layers must be positive")
        if self.fedrep_personal_layers <= 0:
            raise ValueError("fedrep-personal-layers must be positive")
        if self.fedrep_representation_epochs <= 0:
            raise ValueError("fedrep-representation-epochs must be positive")
        if self.input_channels <= 0 or self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input dimensions must be positive")
        if self.num_classes <= 0:
            raise ValueError("num-classes must be positive")
        if not 0 < self.fraction_train <= 1:
            raise ValueError("fraction-train must be in (0, 1]")
        if not 0 < self.fraction_evaluate <= 1:
            raise ValueError("fraction-evaluate must be in (0, 1]")
        if not 0 < self.client_test_fraction < 1:
            raise ValueError("client-test-fraction must be in (0, 1)")
        if self.partitioner not in {"iid", "dirichlet"}:
            raise ValueError("partitioner must be one of: iid, dirichlet")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet-alpha must be positive")
