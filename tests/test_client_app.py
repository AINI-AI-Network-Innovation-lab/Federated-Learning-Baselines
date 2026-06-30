import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from fl_baselines.app.client_app import _num_partitions, _partition_id
from fl_baselines.core.config import ExperimentConfig


class ClientAppPartitionTest(unittest.TestCase):
    def test_partition_id_prefers_explicit_node_config(self) -> None:
        context = SimpleNamespace(node_config={"partition-id": 3}, node_id=99)

        self.assertEqual(_partition_id(context, num_partitions=10), 3)

    def test_partition_id_fallback_uses_node_id_modulo(self) -> None:
        context = SimpleNamespace(node_config={}, node_id=12)

        self.assertEqual(_partition_id(context, num_partitions=10), 2)

    def test_partition_id_rejects_invalid_explicit_value(self) -> None:
        context = SimpleNamespace(node_config={"partition-id": 10}, node_id=0)

        with self.assertRaisesRegex(ValueError, "partition-id must be in"):
            _partition_id(context, num_partitions=10)

    def test_num_partitions_rejects_non_positive_node_config(self) -> None:
        context = SimpleNamespace(node_config={"num-partitions": 0}, node_id=0)
        config = ExperimentConfig.from_run_config({})

        with self.assertRaisesRegex(ValueError, "num-partitions must be positive"):
            _num_partitions(context, config)


if __name__ == "__main__":
    unittest.main()
