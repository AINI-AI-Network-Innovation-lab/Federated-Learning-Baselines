"""Model checkpoint helpers shared by Flower strategies."""

from __future__ import annotations

import torch
from flwr.common import Parameters, parameters_to_ndarrays

from fl_baselines.clients.torch_client import set_model_parameters
from fl_baselines.logging.artifacts import save_model


def save_round_checkpoints(
    model: torch.nn.Module,
    parameters: Parameters,
    output_dir: str,
    server_round: int,
) -> None:
    set_model_parameters(model, parameters_to_ndarrays(parameters))
    save_model(model, output_dir, "final_model.pt")
    save_model(model, output_dir, f"round_{server_round}_model.pt")


class CheckpointingStrategyMixin:
    def __init__(
        self,
        *,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            save_round_checkpoints(
                self._checkpoint_model,
                parameters,
                self._output_dir,
                server_round,
            )
        return parameters, metrics
