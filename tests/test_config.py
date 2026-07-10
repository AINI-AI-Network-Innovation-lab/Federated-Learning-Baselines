import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from fl_baselines.core.config import ExperimentConfig


class ExperimentConfigTest(unittest.TestCase):
    def test_parse_flower_run_config_with_defaults(self) -> None:
        config = ExperimentConfig.from_run_config({})

        self.assertEqual(config.algorithm, "fedavg")
        self.assertEqual(config.dataset, "mnist")
        self.assertEqual(config.model, "mnist_cnn")
        self.assertEqual(config.num_server_rounds, 3)
        self.assertEqual(config.num_supernodes, 10)
        self.assertEqual(config.partitioner, "iid")

    def test_parse_flower_run_config_overrides_kebab_case_keys(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedavg",
                "dataset": "mnist",
                "model": "mnist_cnn",
                "num-server-rounds": 5,
                "num-supernodes": 2,
                "local-epochs": 4,
                "batch-size": 16,
                "learning-rate": 0.05,
                "partitioner": "dirichlet",
                "dirichlet-alpha": 0.3,
                "proximal-mu": 0.2,
                "moon-mu": 1.5,
                "moon-temperature": 0.7,
                "server-learning-rate": 0.8,
                "server-momentum": 0.95,
                "fedadagrad-eta": 0.15,
                "fedadagrad-eta-l": 0.04,
                "fedadagrad-tau": 1e-7,
                "fedadam-eta": 0.2,
                "fedadam-eta-l": 0.03,
                "fedadam-beta-1": 0.8,
                "fedadam-beta-2": 0.97,
                "fedadam-tau": 1e-6,
                "fedyogi-eta": 0.02,
                "fedyogi-eta-l": 0.025,
                "fedyogi-beta-1": 0.85,
                "fedyogi-beta-2": 0.96,
                "fedyogi-tau": 1e-4,
                "fedadp-alpha": 4.0,
                "feddyn-alpha": 0.2,
                "feddc-alpha": 0.05,
                "feddecorr-beta": 0.3,
                "fedexp-epsilon": 0.01,
                "fedspeed-lambda": 0.2,
                "fedspeed-alpha": 0.8,
                "fedspeed-rho": 0.15,
                "fedsam-rho": 0.5,
                "fedgen-alpha": 1.5,
                "fedgen-lambda": 0.2,
                "fedgen-beta": 0.9,
                "fedgen-delta": 0.8,
                "fedgen-warmup-epochs": 2,
                "fedgen-l1-weight": 1e-4,
                "gamf-sigma": 2.0,
                "gamf-initial-tau": 0.05,
                "gamf-descent-factor": 0.9,
                "gamf-min-tau": 0.005,
                "gamf-max-iters": 200,
                "fedma-matching-epsilon": 0.0,
                "fedcda-memory-size": 4,
                "fedcda-num-batches": 2,
                "fedcda-warmup-rounds": 3,
                "fedcda-loss-weight": 1.5,
                "feddrl-actor-learning-rate": 2e-4,
                "feddrl-critic-learning-rate": 2e-3,
                "feddrl-discount-factor": 0.95,
                "feddrl-target-tau": 0.05,
                "feddrl-hidden-size": 128,
                "feddrl-replay-buffer-size": 512,
                "feddrl-batch-size": 16,
                "feddrl-updates-per-round": 3,
                "feddrl-noise-scale": 0.15,
                "feddrl-std-scale": 0.4,
                "fedent-beta": 0.99,
                "fedent-gamma": 0.95,
                "fedent-epsilon": 1e-7,
                "fedent-fixed-point-steps": 3,
                "fedent-max-learning-rate": 0.8,
                "fedlaw-server-epochs": 4,
                "fedlaw-server-learning-rate": 0.03,
                "fedlaw-gamma-init": 0.9,
                "fedaaw-beta": 0.02,
                "fedaaw-gamma": 1.5,
                "fedaaw-epsilon": 1e-7,
                "feddisco-discrepancy-weight": 0.4,
                "feddisco-bias": 0.2,
                "feddisco-metric": "l2",
                "feddisco-epsilon": 1e-7,
                "fedvck-condensed-ratio": 0.02,
                "fedvck-condensed-steps": 3,
                "fedvck-condensed-learning-rate": 0.4,
                "fedvck-importance-alpha": 0.7,
                "fedvck-server-replay-epochs": 2,
                "fedvck-server-replay-learning-rate": 0.05,
                "fedvck-contrastive-temperature": 0.2,
                "fedvck-hard-negative-k": 2,
                "fedvck-enable-latent-constraints": False,
                "fedvck-max-memory-rounds": 4,
                "fedproto-lambda": 0.2,
                "fedmeta-method": "meta-sgd",
                "fedmeta-inner-learning-rate": 0.05,
                "fedmeta-outer-learning-rate": 0.01,
                "fedmeta-support-fraction": 0.4,
                "fedmeta-inner-steps": 2,
                "fedmeta-first-order": False,
                "fedmeta-alpha-init": 0.03,
                "fednp-lambda": 0.6,
                "fednp-prior-variance": 1.5,
                "fednp-stability-eps": 1e-5,
                "fedcurv-lambda": 0.4,
                "fedcurv-fisher-batches": 3,
                "fedcurv-stability-eps": 1e-6,
                "fedmmd-sigma": 0.5,
                "fedmmd-sknq-threshold": 0.4,
                "fedmmd-min-clients": 2,
                "fedmmd-entropy-eps": 1e-8,
                "apfl-alpha": 0.6,
                "apfl-personal-learning-rate": 0.02,
                "apfl-adaptive-alpha": False,
                "apfl-alpha-learning-rate": 0.01,
                "fedntd-beta": 1.2,
                "fedntd-temperature": 2.0,
                "fedlc-tau": 0.5,
                "fedlc-epsilon": 1e-4,
                "fedrs-alpha": 0.5,
                "fedlama-base-interval": 1,
                "fedlama-interval-factor": 2.0,
                "ditto-lambda": 0.3,
                "pfedme-lambda": 15.0,
                "pfedme-beta": 0.7,
                "pfedme-personal-learning-rate": 0.02,
                "pfedme-personal-steps": 4,
                "client-test-fraction": 0.25,
                "fednova-server-momentum": 0.25,
                "fedper-personal-layers": 2,
                "fedrep-personal-layers": 2,
                "fedrep-representation-epochs": 3,
                "fedala-eta": 0.5,
                "fedala-rand-percent": 40,
                "fedala-layer-count": 2,
                "fedala-threshold": 0.02,
                "fedala-num-pre-loss": 4,
                "fedala-start-max-steps": 8,
                "fedamp-lambda": 0.2,
                "fedamp-alpha": 0.05,
                "fedamp-sigma": 2.0,
                "fedlaa-beta": 3.0,
                "input-channels": 3,
                "input-height": 32,
                "input-width": 32,
                "num-classes": 100,
                "emnist-split": "letters",
                "seed": 7,
            }
        )

        self.assertEqual(config.num_server_rounds, 5)
        self.assertEqual(config.num_supernodes, 2)
        self.assertEqual(config.local_epochs, 4)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.partitioner, "dirichlet")
        self.assertEqual(config.dirichlet_alpha, 0.3)
        self.assertEqual(config.proximal_mu, 0.2)
        self.assertEqual(config.moon_mu, 1.5)
        self.assertEqual(config.moon_temperature, 0.7)
        self.assertEqual(config.server_learning_rate, 0.8)
        self.assertEqual(config.server_momentum, 0.95)
        self.assertEqual(config.fedadagrad_eta, 0.15)
        self.assertEqual(config.fedadagrad_eta_l, 0.04)
        self.assertEqual(config.fedadagrad_tau, 1e-7)
        self.assertEqual(config.fedadam_eta, 0.2)
        self.assertEqual(config.fedadam_eta_l, 0.03)
        self.assertEqual(config.fedadam_beta_1, 0.8)
        self.assertEqual(config.fedadam_beta_2, 0.97)
        self.assertEqual(config.fedadam_tau, 1e-6)
        self.assertEqual(config.fedyogi_eta, 0.02)
        self.assertEqual(config.fedyogi_eta_l, 0.025)
        self.assertEqual(config.fedyogi_beta_1, 0.85)
        self.assertEqual(config.fedyogi_beta_2, 0.96)
        self.assertEqual(config.fedyogi_tau, 1e-4)
        self.assertEqual(config.fedadp_alpha, 4.0)
        self.assertEqual(config.feddyn_alpha, 0.2)
        self.assertEqual(config.feddc_alpha, 0.05)
        self.assertEqual(config.feddecorr_beta, 0.3)
        self.assertEqual(config.fedexp_epsilon, 0.01)
        self.assertEqual(config.fedspeed_lambda, 0.2)
        self.assertEqual(config.fedspeed_alpha, 0.8)
        self.assertEqual(config.fedspeed_rho, 0.15)
        self.assertEqual(config.fedsam_rho, 0.5)
        self.assertEqual(config.fedgen_alpha, 1.5)
        self.assertEqual(config.fedgen_lambda, 0.2)
        self.assertEqual(config.fedgen_beta, 0.9)
        self.assertEqual(config.fedgen_delta, 0.8)
        self.assertEqual(config.fedgen_warmup_epochs, 2)
        self.assertEqual(config.fedgen_l1_weight, 1e-4)
        self.assertEqual(config.gamf_sigma, 2.0)
        self.assertEqual(config.gamf_initial_tau, 0.05)
        self.assertEqual(config.gamf_descent_factor, 0.9)
        self.assertEqual(config.gamf_min_tau, 0.005)
        self.assertEqual(config.gamf_max_iters, 200)
        self.assertEqual(config.fedma_matching_epsilon, 0.0)
        self.assertEqual(config.fedcda_memory_size, 4)
        self.assertEqual(config.fedcda_num_batches, 2)
        self.assertEqual(config.fedcda_warmup_rounds, 3)
        self.assertEqual(config.fedcda_loss_weight, 1.5)
        self.assertEqual(config.feddrl_actor_learning_rate, 2e-4)
        self.assertEqual(config.feddrl_critic_learning_rate, 2e-3)
        self.assertEqual(config.feddrl_discount_factor, 0.95)
        self.assertEqual(config.feddrl_target_tau, 0.05)
        self.assertEqual(config.feddrl_hidden_size, 128)
        self.assertEqual(config.feddrl_replay_buffer_size, 512)
        self.assertEqual(config.feddrl_batch_size, 16)
        self.assertEqual(config.feddrl_updates_per_round, 3)
        self.assertEqual(config.feddrl_noise_scale, 0.15)
        self.assertEqual(config.feddrl_std_scale, 0.4)
        self.assertEqual(config.fedent_beta, 0.99)
        self.assertEqual(config.fedent_gamma, 0.95)
        self.assertEqual(config.fedent_epsilon, 1e-7)
        self.assertEqual(config.fedent_fixed_point_steps, 3)
        self.assertEqual(config.fedent_max_learning_rate, 0.8)
        self.assertEqual(config.fedlaw_server_epochs, 4)
        self.assertEqual(config.fedlaw_server_learning_rate, 0.03)
        self.assertEqual(config.fedlaw_gamma_init, 0.9)
        self.assertEqual(config.fedaaw_beta, 0.02)
        self.assertEqual(config.fedaaw_gamma, 1.5)
        self.assertEqual(config.fedaaw_epsilon, 1e-7)
        self.assertEqual(config.feddisco_discrepancy_weight, 0.4)
        self.assertEqual(config.feddisco_bias, 0.2)
        self.assertEqual(config.feddisco_metric, "l2")
        self.assertEqual(config.feddisco_epsilon, 1e-7)
        self.assertEqual(config.fedvck_condensed_ratio, 0.02)
        self.assertEqual(config.fedvck_condensed_steps, 3)
        self.assertEqual(config.fedvck_condensed_learning_rate, 0.4)
        self.assertEqual(config.fedvck_importance_alpha, 0.7)
        self.assertEqual(config.fedvck_server_replay_epochs, 2)
        self.assertEqual(config.fedvck_server_replay_learning_rate, 0.05)
        self.assertEqual(config.fedvck_contrastive_temperature, 0.2)
        self.assertEqual(config.fedvck_hard_negative_k, 2)
        self.assertFalse(config.fedvck_enable_latent_constraints)
        self.assertEqual(config.fedvck_max_memory_rounds, 4)
        self.assertEqual(config.fedproto_lambda, 0.2)
        self.assertEqual(config.fedmeta_method, "meta-sgd")
        self.assertEqual(config.fedmeta_inner_learning_rate, 0.05)
        self.assertEqual(config.fedmeta_outer_learning_rate, 0.01)
        self.assertEqual(config.fedmeta_support_fraction, 0.4)
        self.assertEqual(config.fedmeta_inner_steps, 2)
        self.assertFalse(config.fedmeta_first_order)
        self.assertEqual(config.fedmeta_alpha_init, 0.03)
        self.assertEqual(config.fednp_lambda, 0.6)
        self.assertEqual(config.fednp_prior_variance, 1.5)
        self.assertEqual(config.fednp_stability_eps, 1e-5)
        self.assertEqual(config.fedcurv_lambda, 0.4)
        self.assertEqual(config.fedcurv_fisher_batches, 3)
        self.assertEqual(config.fedcurv_stability_eps, 1e-6)
        self.assertEqual(config.fedmmd_sigma, 0.5)
        self.assertEqual(config.fedmmd_sknq_threshold, 0.4)
        self.assertEqual(config.fedmmd_min_clients, 2)
        self.assertEqual(config.fedmmd_entropy_eps, 1e-8)
        self.assertEqual(config.apfl_alpha, 0.6)
        self.assertEqual(config.apfl_personal_learning_rate, 0.02)
        self.assertFalse(config.apfl_adaptive_alpha)
        self.assertEqual(config.apfl_alpha_learning_rate, 0.01)
        self.assertEqual(config.fedntd_beta, 1.2)
        self.assertEqual(config.fedntd_temperature, 2.0)
        self.assertEqual(config.fedlc_tau, 0.5)
        self.assertEqual(config.fedlc_epsilon, 1e-4)
        self.assertEqual(config.fedrs_alpha, 0.5)
        self.assertEqual(config.fedlama_base_interval, 1)
        self.assertEqual(config.fedlama_interval_factor, 2.0)
        self.assertEqual(config.ditto_lambda, 0.3)
        self.assertEqual(config.pfedme_lambda, 15.0)
        self.assertEqual(config.pfedme_beta, 0.7)
        self.assertEqual(config.pfedme_personal_learning_rate, 0.02)
        self.assertEqual(config.pfedme_personal_steps, 4)
        self.assertEqual(config.client_test_fraction, 0.25)
        self.assertEqual(config.fednova_server_momentum, 0.25)
        self.assertEqual(config.fedper_personal_layers, 2)
        self.assertEqual(config.fedrep_personal_layers, 2)
        self.assertEqual(config.fedrep_representation_epochs, 3)
        self.assertEqual(config.fedala_eta, 0.5)
        self.assertEqual(config.fedala_rand_percent, 40)
        self.assertEqual(config.fedala_layer_count, 2)
        self.assertEqual(config.fedala_threshold, 0.02)
        self.assertEqual(config.fedala_num_pre_loss, 4)
        self.assertEqual(config.fedala_start_max_steps, 8)
        self.assertEqual(config.fedamp_lambda, 0.2)
        self.assertEqual(config.fedamp_alpha, 0.05)
        self.assertEqual(config.fedamp_sigma, 2.0)
        self.assertEqual(config.fedlaa_beta, 3.0)
        self.assertEqual(config.input_shape, (3, 32, 32))
        self.assertEqual(config.num_classes, 100)
        self.assertEqual(config.emnist_split, "letters")
        self.assertEqual(config.seed, 7)

    def test_invalid_model_shape_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "input dimensions must be positive"):
            ExperimentConfig.from_run_config({"input-height": 0})

        with self.assertRaisesRegex(ValueError, "num-classes must be positive"):
            ExperimentConfig.from_run_config({"num-classes": 0})

    def test_invalid_proximal_mu_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "proximal-mu must be non-negative"):
            ExperimentConfig.from_run_config({"proximal-mu": -0.1})

    def test_invalid_moon_hyperparameters_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "moon-mu must be non-negative"):
            ExperimentConfig.from_run_config({"moon-mu": -0.1})

        with self.assertRaisesRegex(ValueError, "moon-temperature must be positive"):
            ExperimentConfig.from_run_config({"moon-temperature": 0.0})

    def test_invalid_server_optimizer_hyperparameters_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "server-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"server-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "server-momentum must be non-negative"):
            ExperimentConfig.from_run_config({"server-momentum": -0.1})

        with self.assertRaisesRegex(ValueError, "fedadagrad-eta must be positive"):
            ExperimentConfig.from_run_config({"fedadagrad-eta": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadagrad-eta-l must be positive"):
            ExperimentConfig.from_run_config({"fedadagrad-eta-l": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadagrad-tau must be positive"):
            ExperimentConfig.from_run_config({"fedadagrad-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadam-eta must be positive"):
            ExperimentConfig.from_run_config({"fedadam-eta": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadam-eta-l must be positive"):
            ExperimentConfig.from_run_config({"fedadam-eta-l": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadam-beta-1 must be in"):
            ExperimentConfig.from_run_config({"fedadam-beta-1": 1.0})

        with self.assertRaisesRegex(ValueError, "fedadam-beta-2 must be in"):
            ExperimentConfig.from_run_config({"fedadam-beta-2": -0.1})

        with self.assertRaisesRegex(ValueError, "fedadam-tau must be positive"):
            ExperimentConfig.from_run_config({"fedadam-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "fedyogi-eta must be positive"):
            ExperimentConfig.from_run_config({"fedyogi-eta": 0.0})

        with self.assertRaisesRegex(ValueError, "fedyogi-eta-l must be positive"):
            ExperimentConfig.from_run_config({"fedyogi-eta-l": 0.0})

        with self.assertRaisesRegex(ValueError, "fedyogi-beta-1 must be in"):
            ExperimentConfig.from_run_config({"fedyogi-beta-1": 1.0})

        with self.assertRaisesRegex(ValueError, "fedyogi-beta-2 must be in"):
            ExperimentConfig.from_run_config({"fedyogi-beta-2": -0.1})

        with self.assertRaisesRegex(ValueError, "fedyogi-tau must be positive"):
            ExperimentConfig.from_run_config({"fedyogi-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "fedadp-alpha must be positive"):
            ExperimentConfig.from_run_config({"fedadp-alpha": 0.0})

        with self.assertRaisesRegex(ValueError, "feddyn-alpha must be positive"):
            ExperimentConfig.from_run_config({"feddyn-alpha": 0.0})

        with self.assertRaisesRegex(ValueError, "feddc-alpha must be positive"):
            ExperimentConfig.from_run_config({"feddc-alpha": 0.0})

        with self.assertRaisesRegex(ValueError, "feddecorr-beta must be non-negative"):
            ExperimentConfig.from_run_config({"feddecorr-beta": -0.1})

        with self.assertRaisesRegex(ValueError, "fedexp-epsilon must be non-negative"):
            ExperimentConfig.from_run_config({"fedexp-epsilon": -0.1})

        with self.assertRaisesRegex(ValueError, "fedspeed-lambda must be positive"):
            ExperimentConfig.from_run_config({"fedspeed-lambda": 0.0})

        with self.assertRaisesRegex(ValueError, "fedspeed-alpha must be in"):
            ExperimentConfig.from_run_config({"fedspeed-alpha": -0.1})

        with self.assertRaisesRegex(ValueError, "fedspeed-alpha must be in"):
            ExperimentConfig.from_run_config({"fedspeed-alpha": 1.1})

        with self.assertRaisesRegex(ValueError, "fedspeed-rho must be non-negative"):
            ExperimentConfig.from_run_config({"fedspeed-rho": -0.1})

        with self.assertRaisesRegex(ValueError, "fedsam-rho must be non-negative"):
            ExperimentConfig.from_run_config({"fedsam-rho": -0.1})

        with self.assertRaisesRegex(ValueError, "fedlaw-server-epochs must be positive"):
            ExperimentConfig.from_run_config({"fedlaw-server-epochs": 0})

        with self.assertRaisesRegex(
            ValueError, "fedlaw-server-learning-rate must be positive"
        ):
            ExperimentConfig.from_run_config({"fedlaw-server-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "fedlaw-gamma-init must be positive"):
            ExperimentConfig.from_run_config({"fedlaw-gamma-init": 0.0})

        with self.assertRaisesRegex(ValueError, "fedgen-alpha must be positive"):
            ExperimentConfig.from_run_config({"fedgen-alpha": 0.0})

        with self.assertRaisesRegex(ValueError, "fedgen-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"fedgen-lambda": -0.1})

        with self.assertRaisesRegex(ValueError, "fedgen-beta must be in"):
            ExperimentConfig.from_run_config({"fedgen-beta": 1.1})

        with self.assertRaisesRegex(ValueError, "fedgen-delta must be in"):
            ExperimentConfig.from_run_config({"fedgen-delta": -0.1})

        with self.assertRaisesRegex(ValueError, "fedgen-warmup-epochs must be non-negative"):
            ExperimentConfig.from_run_config({"fedgen-warmup-epochs": -1})

        with self.assertRaisesRegex(ValueError, "fedgen-l1-weight must be non-negative"):
            ExperimentConfig.from_run_config({"fedgen-l1-weight": -1e-4})

        with self.assertRaisesRegex(ValueError, "gamf-sigma must be positive"):
            ExperimentConfig.from_run_config({"gamf-sigma": 0.0})

        with self.assertRaisesRegex(ValueError, "gamf-initial-tau must be positive"):
            ExperimentConfig.from_run_config({"gamf-initial-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "gamf-descent-factor must be in"):
            ExperimentConfig.from_run_config({"gamf-descent-factor": 1.1})

        with self.assertRaisesRegex(ValueError, "gamf-min-tau must be positive"):
            ExperimentConfig.from_run_config({"gamf-min-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "gamf-max-iters must be positive"):
            ExperimentConfig.from_run_config({"gamf-max-iters": 0})

        with self.assertRaisesRegex(
            ValueError, "fedma-matching-epsilon must be non-negative"
        ):
            ExperimentConfig.from_run_config({"fedma-matching-epsilon": -0.1})

        with self.assertRaisesRegex(ValueError, "fedcda-memory-size must be positive"):
            ExperimentConfig.from_run_config({"fedcda-memory-size": 0})

        with self.assertRaisesRegex(ValueError, "fedcda-num-batches must be positive"):
            ExperimentConfig.from_run_config({"fedcda-num-batches": 0})

        with self.assertRaisesRegex(ValueError, "fedcda-warmup-rounds must be non-negative"):
            ExperimentConfig.from_run_config({"fedcda-warmup-rounds": -1})

        with self.assertRaisesRegex(ValueError, "fedcda-loss-weight must be positive"):
            ExperimentConfig.from_run_config({"fedcda-loss-weight": 0.0})

        with self.assertRaisesRegex(ValueError, "feddrl-actor-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"feddrl-actor-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "feddrl-critic-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"feddrl-critic-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "feddrl-discount-factor must be in"):
            ExperimentConfig.from_run_config({"feddrl-discount-factor": 1.0})

        with self.assertRaisesRegex(ValueError, "feddrl-target-tau must be in"):
            ExperimentConfig.from_run_config({"feddrl-target-tau": 0.0})

        with self.assertRaisesRegex(ValueError, "feddrl-hidden-size must be positive"):
            ExperimentConfig.from_run_config({"feddrl-hidden-size": 0})

        with self.assertRaisesRegex(ValueError, "feddrl-replay-buffer-size must be positive"):
            ExperimentConfig.from_run_config({"feddrl-replay-buffer-size": 0})

        with self.assertRaisesRegex(ValueError, "feddrl-batch-size must be positive"):
            ExperimentConfig.from_run_config({"feddrl-batch-size": 0})

        with self.assertRaisesRegex(ValueError, "feddrl-updates-per-round must be non-negative"):
            ExperimentConfig.from_run_config({"feddrl-updates-per-round": -1})

        with self.assertRaisesRegex(ValueError, "feddrl-noise-scale must be non-negative"):
            ExperimentConfig.from_run_config({"feddrl-noise-scale": -0.1})

        with self.assertRaisesRegex(ValueError, "feddrl-std-scale must be in"):
            ExperimentConfig.from_run_config({"feddrl-std-scale": 0.0})

        with self.assertRaisesRegex(ValueError, "fedent-beta must be in"):
            ExperimentConfig.from_run_config({"fedent-beta": 1.0})

        with self.assertRaisesRegex(ValueError, "fedent-gamma must be in"):
            ExperimentConfig.from_run_config({"fedent-gamma": 1.0})

        with self.assertRaisesRegex(ValueError, "fedent-epsilon must be positive"):
            ExperimentConfig.from_run_config({"fedent-epsilon": 0.0})

        with self.assertRaisesRegex(ValueError, "fedaaw-beta must be positive"):
            ExperimentConfig.from_run_config({"fedaaw-beta": 0.0})

        with self.assertRaisesRegex(ValueError, "fedaaw-gamma must be non-negative"):
            ExperimentConfig.from_run_config({"fedaaw-gamma": -0.1})

        with self.assertRaisesRegex(ValueError, "fedaaw-epsilon must be positive"):
            ExperimentConfig.from_run_config({"fedaaw-epsilon": 0.0})

        with self.assertRaisesRegex(
            ValueError,
            "feddisco-discrepancy-weight must be non-negative",
        ):
            ExperimentConfig.from_run_config({"feddisco-discrepancy-weight": -0.1})

        with self.assertRaisesRegex(ValueError, "feddisco-bias must be non-negative"):
            ExperimentConfig.from_run_config({"feddisco-bias": -0.1})

        with self.assertRaisesRegex(ValueError, "feddisco-metric must be one of"):
            ExperimentConfig.from_run_config({"feddisco-metric": "bad"})

        with self.assertRaisesRegex(ValueError, "feddisco-epsilon must be positive"):
            ExperimentConfig.from_run_config({"feddisco-epsilon": 0.0})

        with self.assertRaisesRegex(ValueError, "fedvck-condensed-ratio must be positive"):
            ExperimentConfig.from_run_config({"fedvck-condensed-ratio": 0.0})

        with self.assertRaisesRegex(ValueError, "fedvck-importance-alpha must be in"):
            ExperimentConfig.from_run_config({"fedvck-importance-alpha": 1.1})

        with self.assertRaisesRegex(ValueError, "fedvck-hard-negative-k must be positive"):
            ExperimentConfig.from_run_config({"fedvck-hard-negative-k": 0})

        with self.assertRaisesRegex(ValueError, "fedproto-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"fedproto-lambda": -0.1})

        with self.assertRaisesRegex(ValueError, "fedmeta-method must be one of"):
            ExperimentConfig.from_run_config({"fedmeta-method": "bad"})

        with self.assertRaisesRegex(ValueError, "fedmeta-inner-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"fedmeta-inner-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "fedmeta-outer-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"fedmeta-outer-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "fedmeta-support-fraction must be in"):
            ExperimentConfig.from_run_config({"fedmeta-support-fraction": 1.0})

        with self.assertRaisesRegex(ValueError, "fedmeta-inner-steps must be positive"):
            ExperimentConfig.from_run_config({"fedmeta-inner-steps": 0})

        with self.assertRaisesRegex(ValueError, "fedmeta-alpha-init must be positive"):
            ExperimentConfig.from_run_config({"fedmeta-alpha-init": 0.0})

        with self.assertRaisesRegex(ValueError, "fednp-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"fednp-lambda": -0.1})

        with self.assertRaisesRegex(ValueError, "fednp-prior-variance must be positive"):
            ExperimentConfig.from_run_config({"fednp-prior-variance": 0.0})

        with self.assertRaisesRegex(ValueError, "fednp-stability-eps must be positive"):
            ExperimentConfig.from_run_config({"fednp-stability-eps": 0.0})
        with self.assertRaisesRegex(ValueError, "fedcurv-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"fedcurv-lambda": -0.1})
        with self.assertRaisesRegex(ValueError, "fedcurv-fisher-batches must be positive"):
            ExperimentConfig.from_run_config({"fedcurv-fisher-batches": 0})
        with self.assertRaisesRegex(ValueError, "fedcurv-stability-eps must be positive"):
            ExperimentConfig.from_run_config({"fedcurv-stability-eps": 0.0})
        with self.assertRaisesRegex(ValueError, "fedmmd-sigma must be positive"):
            ExperimentConfig.from_run_config({"fedmmd-sigma": 0.0})
        with self.assertRaisesRegex(ValueError, "fedmmd-sknq-threshold must be in"):
            ExperimentConfig.from_run_config({"fedmmd-sknq-threshold": 1.0})
        with self.assertRaisesRegex(ValueError, "fedmmd-min-clients must be at least 1"):
            ExperimentConfig.from_run_config({"fedmmd-min-clients": 0})
        with self.assertRaisesRegex(ValueError, "fedmmd-entropy-eps must be positive"):
            ExperimentConfig.from_run_config({"fedmmd-entropy-eps": 0.0})
        with self.assertRaisesRegex(ValueError, "apfl-alpha must be in"):
            ExperimentConfig.from_run_config({"apfl-alpha": 1.0})
        with self.assertRaisesRegex(ValueError, "apfl-personal-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"apfl-personal-learning-rate": 0.0})
        with self.assertRaisesRegex(ValueError, "apfl-alpha-learning-rate must be positive"):
            ExperimentConfig.from_run_config({"apfl-alpha-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "fedntd-beta must be non-negative"):
            ExperimentConfig.from_run_config({"fedntd-beta": -0.1})

        with self.assertRaisesRegex(ValueError, "fedntd-temperature must be positive"):
            ExperimentConfig.from_run_config({"fedntd-temperature": 0.0})

        with self.assertRaisesRegex(ValueError, "fedlc-tau must be non-negative"):
            ExperimentConfig.from_run_config({"fedlc-tau": -0.1})

        with self.assertRaisesRegex(ValueError, "fedlc-epsilon must be positive"):
            ExperimentConfig.from_run_config({"fedlc-epsilon": 0.0})

        with self.assertRaisesRegex(ValueError, "fedrs-alpha must be in"):
            ExperimentConfig.from_run_config({"fedrs-alpha": -0.1})

        with self.assertRaisesRegex(ValueError, "fedrs-alpha must be in"):
            ExperimentConfig.from_run_config({"fedrs-alpha": 1.1})

        with self.assertRaisesRegex(ValueError, "fedlama-base-interval must be positive"):
            ExperimentConfig.from_run_config({"fedlama-base-interval": 0})

        with self.assertRaisesRegex(
            ValueError, "fedlama-interval-factor must be at least 1"
        ):
            ExperimentConfig.from_run_config({"fedlama-interval-factor": 0.5})

        with self.assertRaisesRegex(ValueError, "ditto-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"ditto-lambda": -0.1})

        with self.assertRaisesRegex(ValueError, "pfedme-lambda must be positive"):
            ExperimentConfig.from_run_config({"pfedme-lambda": 0.0})

        with self.assertRaisesRegex(ValueError, "pfedme-beta must be positive"):
            ExperimentConfig.from_run_config({"pfedme-beta": 0.0})

        with self.assertRaisesRegex(ValueError, "pfedme-beta must be in"):
            ExperimentConfig.from_run_config({"pfedme-beta": 1.1})

        with self.assertRaisesRegex(
            ValueError,
            "pfedme-personal-learning-rate must be positive",
        ):
            ExperimentConfig.from_run_config({"pfedme-personal-learning-rate": 0.0})

        with self.assertRaisesRegex(ValueError, "pfedme-personal-steps must be positive"):
            ExperimentConfig.from_run_config({"pfedme-personal-steps": 0})

        with self.assertRaisesRegex(ValueError, "client-test-fraction must be in"):
            ExperimentConfig.from_run_config({"client-test-fraction": 0.0})

        with self.assertRaisesRegex(ValueError, "client-test-fraction must be in"):
            ExperimentConfig.from_run_config({"client-test-fraction": 1.0})

        with self.assertRaisesRegex(ValueError, "fednova-server-momentum must be non-negative"):
            ExperimentConfig.from_run_config({"fednova-server-momentum": -0.1})

        with self.assertRaisesRegex(ValueError, "fedper-personal-layers must be positive"):
            ExperimentConfig.from_run_config({"fedper-personal-layers": 0})

        with self.assertRaisesRegex(ValueError, "fedrep-personal-layers must be positive"):
            ExperimentConfig.from_run_config({"fedrep-personal-layers": 0})

        with self.assertRaisesRegex(ValueError, "fedrep-representation-epochs must be positive"):
            ExperimentConfig.from_run_config({"fedrep-representation-epochs": 0})

        with self.assertRaisesRegex(ValueError, "fedala-eta must be positive"):
            ExperimentConfig.from_run_config({"fedala-eta": 0.0})

        with self.assertRaisesRegex(ValueError, "fedala-rand-percent must be in"):
            ExperimentConfig.from_run_config({"fedala-rand-percent": 0})

        with self.assertRaisesRegex(ValueError, "fedala-layer-count must be positive"):
            ExperimentConfig.from_run_config({"fedala-layer-count": 0})

        with self.assertRaisesRegex(ValueError, "fedala-threshold must be positive"):
            ExperimentConfig.from_run_config({"fedala-threshold": 0.0})

        with self.assertRaisesRegex(ValueError, "fedala-num-pre-loss must be positive"):
            ExperimentConfig.from_run_config({"fedala-num-pre-loss": 0})

        with self.assertRaisesRegex(ValueError, "fedala-start-max-steps must be positive"):
            ExperimentConfig.from_run_config({"fedala-start-max-steps": 0})

        with self.assertRaisesRegex(ValueError, "fedamp-lambda must be positive"):
            ExperimentConfig.from_run_config({"fedamp-lambda": 0.0})

        with self.assertRaisesRegex(ValueError, "fedamp-alpha must be positive"):
            ExperimentConfig.from_run_config({"fedamp-alpha": 0.0})

        with self.assertRaisesRegex(ValueError, "fedamp-sigma must be positive"):
            ExperimentConfig.from_run_config({"fedamp-sigma": 0.0})

        with self.assertRaisesRegex(ValueError, "fedlaa-beta must be positive"):
            ExperimentConfig.from_run_config({"fedlaa-beta": 0.0})

    def test_invalid_partitioner_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "partitioner must be one of"):
            ExperimentConfig.from_run_config({"partitioner": "shard"})


if __name__ == "__main__":
    unittest.main()
