"""FedDRL algorithm builder."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from torch import nn

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.feddrl import compute_feddrl_reward


@dataclass
class _Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray


class _ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buckets: dict[int, deque[_Experience]] = {}

    def add(self, experience: _Experience) -> None:
        num_clients = int(experience.state.shape[0])
        bucket = self._buckets.setdefault(num_clients, deque(maxlen=self._capacity))
        bucket.append(experience)

    def sample(
        self,
        *,
        num_clients: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[_Experience]:
        bucket = self._buckets.get(num_clients)
        if not bucket:
            return []
        indices = rng.choice(len(bucket), size=min(batch_size, len(bucket)), replace=False)
        return [bucket[int(index)] for index in indices]


class _ActorNetwork(nn.Module):
    def __init__(self, hidden_size: int, std_scale: float) -> None:
        super().__init__()
        self.std_scale = std_scale
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_std_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(state)
        mean = self.mean_head(encoded).squeeze(-1)
        std = torch.sigmoid(self.log_std_head(encoded)).squeeze(-1) * self.std_scale
        std = std.clamp_min(1e-6)
        return mean, std


class _CriticNetwork(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat((state, action.unsqueeze(-1)), dim=-1)
        values = self.encoder(critic_input).squeeze(-1)
        return values.mean(dim=-1)


class FedDRLStrategy(FedAvg):
    """FedDRL adaptive aggregation with an online actor-critic policy."""

    def __init__(
        self,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float,
        discount_factor: float,
        target_tau: float,
        hidden_size: int,
        replay_buffer_size: int,
        batch_size: int,
        updates_per_round: int,
        noise_scale: float,
        std_scale: float,
        seed: int,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedDRLStrategy requires initial_parameters")
        self.discount_factor = discount_factor
        self.target_tau = target_tau
        self.batch_size = batch_size
        self.updates_per_round = updates_per_round
        self.noise_scale = noise_scale
        self.hidden_size = hidden_size
        self.std_scale = std_scale
        self._rng = np.random.default_rng(seed)
        self._replay_buffer = _ReplayBuffer(replay_buffer_size)
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

        self.actor: _ActorNetwork | None = None
        self.actor_target: _ActorNetwork | None = None
        self.critic: _CriticNetwork | None = None
        self.critic_target: _CriticNetwork | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None

        self.last_aggregation_weights: list[float] = []
        self.last_reward: float | None = None
        self._previous_state: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        state = self._build_state(results, total_examples)
        self._observe_transition(state)
        raw_action, aggregation_weights = self._select_action(state, explore=True)
        self.last_aggregation_weights = aggregation_weights

        client_models = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        aggregated_model = aggregate(list(zip(client_models, aggregation_weights)))
        aggregated_parameters = ndarrays_to_parameters(aggregated_model)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        if self.last_reward is not None:
            metrics_aggregated["feddrl_reward"] = self.last_reward

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )

        self._previous_state = state.copy()
        self._previous_action = raw_action.copy()
        return aggregated_parameters, metrics_aggregated

    def _build_state(
        self,
        results: list[tuple[Any, Any]],
        total_examples: int,
    ) -> np.ndarray:
        state_rows: list[list[float]] = []
        for _, fit_res in results:
            state_rows.append(
                [
                    float(fit_res.metrics.get("feddrl_pre_train_loss", 0.0)),
                    float(fit_res.metrics.get("feddrl_post_train_loss", 0.0)),
                    float(fit_res.num_examples) / float(total_examples),
                ]
            )
        return np.asarray(state_rows, dtype=np.float32)

    def _observe_transition(self, state: np.ndarray) -> None:
        if self._previous_state is None or self._previous_action is None:
            self.last_reward = None
            return
        if self._previous_state.shape != state.shape:
            self._previous_state = None
            self._previous_action = None
            self.last_reward = None
            return

        reward = compute_feddrl_reward(state[:, 0].tolist())
        self.last_reward = reward
        self._replay_buffer.add(
            _Experience(
                state=self._previous_state.copy(),
                action=self._previous_action.copy(),
                reward=reward,
                next_state=state.copy(),
            )
        )
        self._train_from_replay(state.shape[0])

    def _select_action(
        self,
        state: np.ndarray,
        *,
        explore: bool,
    ) -> tuple[np.ndarray, list[float]]:
        self._ensure_networks()
        assert self.actor is not None

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
        raw_action = mean.squeeze(0)
        if explore and self.noise_scale > 0:
            noise = torch.randn_like(raw_action) * std.squeeze(0) * self.noise_scale
            raw_action = raw_action + noise

        weights = torch.softmax(raw_action, dim=-1)
        return (
            raw_action.detach().cpu().numpy().astype(np.float32),
            [float(weight) for weight in weights.detach().cpu().numpy()],
        )

    def _train_from_replay(self, num_clients: int) -> None:
        if self.updates_per_round <= 0:
            return
        self._ensure_networks()
        assert self.actor is not None
        assert self.actor_target is not None
        assert self.critic is not None
        assert self.critic_target is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None

        for _ in range(self.updates_per_round):
            batch = self._replay_buffer.sample(
                num_clients=num_clients,
                batch_size=self.batch_size,
                rng=self._rng,
            )
            if not batch:
                return

            states = torch.as_tensor(
                np.stack([experience.state for experience in batch]),
                dtype=torch.float32,
            )
            actions = torch.as_tensor(
                np.stack([experience.action for experience in batch]),
                dtype=torch.float32,
            )
            rewards = torch.as_tensor(
                [experience.reward for experience in batch],
                dtype=torch.float32,
            )
            next_states = torch.as_tensor(
                np.stack([experience.next_state for experience in batch]),
                dtype=torch.float32,
            )

            with torch.no_grad():
                next_action, _ = self.actor_target(next_states)
                target_q = rewards + self.discount_factor * self.critic_target(
                    next_states,
                    next_action,
                )

            critic_loss = torch.nn.functional.mse_loss(
                self.critic(states, actions),
                target_q,
            )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            predicted_action, _ = self.actor(states)
            actor_loss = -self.critic(states, predicted_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

    def _ensure_networks(self) -> None:
        if self.actor is not None:
            return
        self.actor = _ActorNetwork(self.hidden_size, self.std_scale)
        self.actor_target = _ActorNetwork(self.hidden_size, self.std_scale)
        self.critic = _CriticNetwork(self.hidden_size)
        self.critic_target = _CriticNetwork(self.hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self._actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self._critic_learning_rate,
        )

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_parameter, source_parameter in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_parameter.data.mul_(1.0 - self.target_tau).add_(
                source_parameter.data,
                alpha=self.target_tau,
            )


class FedDRLBuilder:
    name = "feddrl"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedDRLStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedDRLStrategy(
            fraction_fit=config.fraction_train,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=config.num_supernodes,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=self._fit_config(config),
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            actor_learning_rate=config.feddrl_actor_learning_rate,
            critic_learning_rate=config.feddrl_critic_learning_rate,
            discount_factor=config.feddrl_discount_factor,
            target_tau=config.feddrl_target_tau,
            hidden_size=config.feddrl_hidden_size,
            replay_buffer_size=config.feddrl_replay_buffer_size,
            batch_size=config.feddrl_batch_size,
            updates_per_round=config.feddrl_updates_per_round,
            noise_scale=config.feddrl_noise_scale,
            std_scale=config.feddrl_std_scale,
            seed=config.seed,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
            }

        return fn
