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
                "fedadp-alpha": 4.0,
                "feddyn-alpha": 0.2,
                "feddc-alpha": 0.05,
                "feddecorr-beta": 0.3,
                "fedexp-epsilon": 0.01,
                "fedspeed-lambda": 0.2,
                "fedspeed-alpha": 0.8,
                "fedspeed-rho": 0.15,
                "fedsam-rho": 0.5,
                "fedent-beta": 0.99,
                "fedent-gamma": 0.95,
                "fedent-epsilon": 1e-7,
                "fedent-fixed-point-steps": 3,
                "fedent-max-learning-rate": 0.8,
                "fedaaw-beta": 0.02,
                "fedaaw-gamma": 1.5,
                "fedaaw-epsilon": 1e-7,
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
                "fedntd-beta": 1.2,
                "fedntd-temperature": 2.0,
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
        self.assertEqual(config.fedadp_alpha, 4.0)
        self.assertEqual(config.feddyn_alpha, 0.2)
        self.assertEqual(config.feddc_alpha, 0.05)
        self.assertEqual(config.feddecorr_beta, 0.3)
        self.assertEqual(config.fedexp_epsilon, 0.01)
        self.assertEqual(config.fedspeed_lambda, 0.2)
        self.assertEqual(config.fedspeed_alpha, 0.8)
        self.assertEqual(config.fedspeed_rho, 0.15)
        self.assertEqual(config.fedsam_rho, 0.5)
        self.assertEqual(config.fedent_beta, 0.99)
        self.assertEqual(config.fedent_gamma, 0.95)
        self.assertEqual(config.fedent_epsilon, 1e-7)
        self.assertEqual(config.fedent_fixed_point_steps, 3)
        self.assertEqual(config.fedent_max_learning_rate, 0.8)
        self.assertEqual(config.fedaaw_beta, 0.02)
        self.assertEqual(config.fedaaw_gamma, 1.5)
        self.assertEqual(config.fedaaw_epsilon, 1e-7)
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
        self.assertEqual(config.fedntd_beta, 1.2)
        self.assertEqual(config.fedntd_temperature, 2.0)
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

        with self.assertRaisesRegex(ValueError, "fedvck-condensed-ratio must be positive"):
            ExperimentConfig.from_run_config({"fedvck-condensed-ratio": 0.0})

        with self.assertRaisesRegex(ValueError, "fedvck-importance-alpha must be in"):
            ExperimentConfig.from_run_config({"fedvck-importance-alpha": 1.1})

        with self.assertRaisesRegex(ValueError, "fedvck-hard-negative-k must be positive"):
            ExperimentConfig.from_run_config({"fedvck-hard-negative-k": 0})

        with self.assertRaisesRegex(ValueError, "fedproto-lambda must be non-negative"):
            ExperimentConfig.from_run_config({"fedproto-lambda": -0.1})

        with self.assertRaisesRegex(ValueError, "fedntd-beta must be non-negative"):
            ExperimentConfig.from_run_config({"fedntd-beta": -0.1})

        with self.assertRaisesRegex(ValueError, "fedntd-temperature must be positive"):
            ExperimentConfig.from_run_config({"fedntd-temperature": 0.0})

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

    def test_invalid_partitioner_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "partitioner must be one of"):
            ExperimentConfig.from_run_config({"partitioner": "shard"})


if __name__ == "__main__":
    unittest.main()
