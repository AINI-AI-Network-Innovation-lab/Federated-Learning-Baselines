"""Flower ClientApp entrypoint."""

from __future__ import annotations

from typing import cast

from flwr.app import Context
from flwr.client import Client
from flwr.clientapp import ClientApp

from fl_baselines import register_default_components
from fl_baselines.clients.torch_client import TorchFlowerClient
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.registry import DATASETS, MODELS
from fl_baselines.datasets.base import DatasetBuilder
from fl_baselines.models.base import ModelBuilder


def _partition_id(context: Context, num_partitions: int) -> int:
    if num_partitions <= 0:
        raise ValueError("num-partitions must be positive")

    raw_partition_id = context.node_config.get("partition-id")
    if raw_partition_id is None:
        return int(context.node_id) % num_partitions

    partition_id = int(raw_partition_id)
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError("partition-id must be in [0, num-partitions)")
    return partition_id


def _num_partitions(context: Context, config: ExperimentConfig) -> int:
    raw_num_partitions = context.node_config.get("num-partitions")
    if raw_num_partitions is None:
        return config.num_supernodes

    num_partitions = int(raw_num_partitions)
    if num_partitions <= 0:
        raise ValueError("num-partitions must be positive")
    return num_partitions


def client_fn(context: Context) -> Client:
    register_default_components()
    config = ExperimentConfig.from_run_config(context.run_config)

    dataset = cast(DatasetBuilder, DATASETS.get(config.dataset))
    model_builder = cast(ModelBuilder, MODELS.get(config.model))
    num_partitions = _num_partitions(context, config)
    partition_id = _partition_id(context, num_partitions)

    loaders = dataset.build_client_loaders(
        config,
        partition_id=partition_id,
        num_partitions=num_partitions,
    )
    model = model_builder.build_model(config)
    return TorchFlowerClient(
        model,
        loaders,
        config,
        client_id=str(partition_id),
    ).to_client()


app = ClientApp(client_fn=client_fn)
