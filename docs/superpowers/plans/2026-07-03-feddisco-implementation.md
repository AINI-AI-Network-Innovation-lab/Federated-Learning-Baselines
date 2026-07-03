# FedDisco Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `feddisco` as a selectable Flower baseline with discrepancy-aware client aggregation from Ye et al., ICML 2023.

**Architecture:** Implement FedDisco as a standalone `FedAvg`-derived strategy. Clients compute one scalar label-distribution discrepancy via a dedicated training helper, return it in fit metrics, and the server uses the paper's `ReLU(n_k - a * d_k + b)` weighting rule with safe fallback to sample weighting.

**Tech Stack:** Python, Flower, PyTorch, NumPy, `unittest`, existing repo registries/config/docs.

---

## File Structure

- Create `src/fl_baselines/training/feddisco.py`: local label-distribution discrepancy helpers plus `train_feddisco_client`.
- Create `src/fl_baselines/algorithms/feddisco.py`: `FedDiscoStrategy` and `FedDiscoBuilder`.
- Modify `src/fl_baselines/core/config.py`: add FedDisco config defaults, parsing, and validation.
- Modify `src/fl_baselines/defaults.py`: register `FedDiscoBuilder`.
- Modify `src/fl_baselines/clients/torch_client.py`: route `algorithm == "feddisco"` to `_fit_feddisco`.
- Modify `tests/test_config.py`: config parse and validation coverage.
- Modify `tests/test_registry.py`: default registry includes `feddisco`.
- Modify `tests/test_model_and_algorithm.py`: builder, strategy, helper, and client routing coverage.
- Modify `pyproject.toml`: default FedDisco run-config keys.
- Modify `README.md` and docs algorithm indexes: document `feddisco` and paper link.

## Task 1: Config Surface

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/fl_baselines/core/config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing config tests**

Add FedDisco overrides to `test_parse_flower_run_config_overrides_kebab_case_keys`:

```python
"feddisco-discrepancy-weight": 0.4,
"feddisco-bias": 0.2,
"feddisco-metric": "l2",
"feddisco-epsilon": 1e-7,
```

Add assertions:

```python
self.assertEqual(config.feddisco_discrepancy_weight, 0.4)
self.assertEqual(config.feddisco_bias, 0.2)
self.assertEqual(config.feddisco_metric, "l2")
self.assertEqual(config.feddisco_epsilon, 1e-7)
```

Add validation cases:

```python
with self.assertRaisesRegex(ValueError, "feddisco-discrepancy-weight must be non-negative"):
    ExperimentConfig.from_run_config({"feddisco-discrepancy-weight": -0.1})

with self.assertRaisesRegex(ValueError, "feddisco-bias must be non-negative"):
    ExperimentConfig.from_run_config({"feddisco-bias": -0.1})

with self.assertRaisesRegex(ValueError, "feddisco-metric must be one of"):
    ExperimentConfig.from_run_config({"feddisco-metric": "bad"})

with self.assertRaisesRegex(ValueError, "feddisco-epsilon must be positive"):
    ExperimentConfig.from_run_config({"feddisco-epsilon": 0.0})
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config -v
```

Expected: FAIL because `ExperimentConfig` has no FedDisco fields yet.

- [ ] **Step 3: Implement config fields**

Add dataclass fields near existing FedAAW/FedVCK config:

```python
feddisco_discrepancy_weight: float = 0.5
feddisco_bias: float = 0.1
feddisco_metric: str = "kl"
feddisco_epsilon: float = 1e-8
```

Parse in `from_run_config`:

```python
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
```

Validate:

```python
if self.feddisco_discrepancy_weight < 0:
    raise ValueError("feddisco-discrepancy-weight must be non-negative")
if self.feddisco_bias < 0:
    raise ValueError("feddisco-bias must be non-negative")
if self.feddisco_metric not in {"kl", "l1", "l2", "cosine"}:
    raise ValueError("feddisco-metric must be one of: kl, l1, l2, cosine")
if self.feddisco_epsilon <= 0:
    raise ValueError("feddisco-epsilon must be positive")
```

Add to `[tool.flwr.app.config]` in `pyproject.toml`:

```toml
feddisco-discrepancy-weight = 0.5
feddisco-bias = 0.1
feddisco-metric = "kl"
feddisco-epsilon = 1e-8
```

- [ ] **Step 4: Run config tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py src/fl_baselines/core/config.py pyproject.toml
git commit -m "feat: add FedDisco config"
```

## Task 2: Local Discrepancy Helper

**Files:**
- Modify: `tests/test_model_and_algorithm.py`
- Create: `src/fl_baselines/training/feddisco.py`

- [ ] **Step 1: Write failing helper tests**

Import:

```python
from fl_baselines.training.feddisco import (
    compute_label_distribution,
    compute_label_distribution_discrepancy,
    train_feddisco_client,
)
```

Add tests:

```python
def test_feddisco_discrepancy_is_lower_for_uniform_labels(self) -> None:
    uniform_loader = DataLoader(
        TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    skewed_loader = DataLoader(
        TensorDataset(torch.ones(4, 1), torch.tensor([0, 0, 0, 0])),
        batch_size=2,
    )

    uniform = compute_label_distribution(uniform_loader, num_classes=2)
    skewed = compute_label_distribution(skewed_loader, num_classes=2)

    self.assertLess(
        compute_label_distribution_discrepancy(uniform, metric="kl", epsilon=1e-8),
        compute_label_distribution_discrepancy(skewed, metric="kl", epsilon=1e-8),
    )
```

```python
def test_feddisco_discrepancy_supports_all_metrics(self) -> None:
    distribution = torch.tensor([1.0, 0.0])

    for metric in ["kl", "l1", "l2", "cosine"]:
        with self.subTest(metric=metric):
            discrepancy = compute_label_distribution_discrepancy(
                distribution,
                metric=metric,
                epsilon=1e-8,
            )
            self.assertGreaterEqual(discrepancy, 0.0)
```

```python
def test_train_feddisco_client_returns_discrepancy_metric(self) -> None:
    model = torch.nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )

    metrics = train_feddisco_client(
        model,
        loader,
        epochs=1,
        learning_rate=0.01,
        device="cpu",
        num_classes=2,
        metric="kl",
        epsilon=1e-8,
    )

    self.assertIn("feddisco_discrepancy", metrics)
    self.assertGreaterEqual(metrics["feddisco_discrepancy"], 0.0)
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_discrepancy_is_lower_for_uniform_labels -v
```

Expected: FAIL because `fl_baselines.training.feddisco` does not exist.

- [ ] **Step 3: Implement helper module**

Create `src/fl_baselines/training/feddisco.py` with:

```python
"""FedDisco local training helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.train import train_one_client


def compute_label_distribution(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, targets in train_loader:
        labels = targets.detach().cpu().long().flatten()
        valid_labels = labels[(labels >= 0) & (labels < num_classes)]
        if valid_labels.numel() > 0:
            counts += torch.bincount(valid_labels, minlength=num_classes).to(torch.float64)
    total = torch.sum(counts)
    if total <= 0:
        return counts
    return counts / total


def compute_label_distribution_discrepancy(
    distribution: torch.Tensor,
    *,
    metric: str,
    epsilon: float,
) -> float:
    distribution = distribution.to(dtype=torch.float64)
    if distribution.numel() == 0 or float(torch.sum(distribution)) <= 0:
        return 0.0
    target = torch.full_like(distribution, 1.0 / distribution.numel())
    if metric == "kl":
        value = torch.sum(distribution * torch.log((distribution + epsilon) / (target + epsilon)))
    elif metric == "l1":
        value = torch.sum(torch.abs(distribution - target))
    elif metric == "l2":
        value = torch.linalg.vector_norm(distribution - target, ord=2)
    elif metric == "cosine":
        numerator = torch.sum(distribution * target)
        denominator = torch.linalg.vector_norm(distribution) * torch.linalg.vector_norm(target)
        value = 1.0 - (numerator / torch.clamp(denominator, min=epsilon))
    else:
        raise ValueError("FedDisco metric must be one of: kl, l1, l2, cosine")
    return float(torch.clamp(value, min=0.0).item())


def train_feddisco_client(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    num_classes: int,
    metric: str,
    epsilon: float,
) -> dict[str, float]:
    distribution = compute_label_distribution(train_loader, num_classes)
    discrepancy = compute_label_distribution_discrepancy(
        distribution,
        metric=metric,
        epsilon=epsilon,
    )
    metrics = train_one_client(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    metrics["feddisco_discrepancy"] = discrepancy
    return metrics
```

- [ ] **Step 4: Run helper tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_discrepancy_is_lower_for_uniform_labels tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_discrepancy_supports_all_metrics tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_train_feddisco_client_returns_discrepancy_metric -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_and_algorithm.py src/fl_baselines/training/feddisco.py
git commit -m "feat: add FedDisco client discrepancy helper"
```

## Task 3: Strategy And Registry

**Files:**
- Modify: `tests/test_model_and_algorithm.py`
- Modify: `tests/test_registry.py`
- Create: `src/fl_baselines/algorithms/feddisco.py`
- Modify: `src/fl_baselines/defaults.py`

- [ ] **Step 1: Write failing strategy and registry tests**

Import:

```python
from fl_baselines.algorithms.feddisco import FedDiscoBuilder, FedDiscoStrategy
```

Add `feddisco` to the algorithm list in `tests/test_registry.py`.

Add builder test:

```python
def test_feddisco_builder_creates_strategy(self) -> None:
    config = ExperimentConfig.from_run_config(
        {
            "algorithm": "feddisco",
            "num-supernodes": 4,
            "feddisco-discrepancy-weight": 0.4,
            "feddisco-bias": 0.2,
            "feddisco-metric": "l2",
        }
    )
    model = MnistCnnBuilder().build_model(config)

    strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
    fit_config = strategy.on_fit_config_fn(1)

    self.assertIsInstance(strategy, FedDiscoStrategy)
    self.assertEqual(strategy.min_fit_clients, 4)
    self.assertEqual(strategy.discrepancy_weight, 0.4)
    self.assertEqual(strategy.bias, 0.2)
    self.assertEqual(fit_config["algorithm"], "feddisco")
    self.assertEqual(fit_config["feddisco_metric"], "l2")
```

Add aggregation tests using the existing `FitRes` pattern:

```python
def test_feddisco_strategy_gives_higher_weight_to_lower_discrepancy(self) -> None:
    config = ExperimentConfig.from_run_config({"algorithm": "feddisco", "num-supernodes": 2})
    model = MnistCnnBuilder().build_model(config)
    strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
    first = np.asarray([1.0], dtype=np.float32)
    second = np.asarray([3.0], dtype=np.float32)

    strategy.aggregate_fit(
        1,
        [
            (_ClientProxy("low"), _fit_res([first], 10, {"feddisco_discrepancy": 0.0})),
            (_ClientProxy("high"), _fit_res([second], 10, {"feddisco_discrepancy": 1.0})),
        ],
        [],
    )

    self.assertGreater(strategy.last_aggregation_weights[0], strategy.last_aggregation_weights[1])
```

```python
def test_feddisco_strategy_falls_back_when_relu_scores_are_zero(self) -> None:
    config = ExperimentConfig.from_run_config(
        {
            "algorithm": "feddisco",
            "num-supernodes": 2,
            "feddisco-discrepancy-weight": 10.0,
            "feddisco-bias": 0.0,
        }
    )
    model = MnistCnnBuilder().build_model(config)
    strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)

    strategy.aggregate_fit(
        1,
        [
            (_ClientProxy("a"), _fit_res([np.asarray([1.0], dtype=np.float32)], 1, {"feddisco_discrepancy": 1.0})),
            (_ClientProxy("b"), _fit_res([np.asarray([3.0], dtype=np.float32)], 3, {"feddisco_discrepancy": 1.0})),
        ],
        [],
    )

    self.assertEqual(strategy.last_aggregation_weights, [0.25, 0.75])
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_builder_creates_strategy -v
```

Expected: FAIL because `feddisco` algorithm module and registration do not exist.

- [ ] **Step 3: Implement strategy module**

Create `src/fl_baselines/algorithms/feddisco.py` with `FedDiscoStrategy(FedAvg)`, `_client_key`, `_feddisco_weights`, and `FedDiscoBuilder`. The core scoring code must be:

```python
score = sample_weight - (self.discrepancy_weight * discrepancy) + self.bias
weights = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
total = float(np.sum(weights))
if not math.isfinite(total) or total <= 0:
    return sample_weights
return [float(weight / total) for weight in weights]
```

`aggregate_fit` should aggregate with Flower's `aggregate`, save checkpoints, and expose `last_aggregation_weights`.

- [ ] **Step 4: Register defaults**

In `src/fl_baselines/defaults.py`, import and register:

```python
from fl_baselines.algorithms.feddisco import FedDiscoBuilder
```

```python
if "feddisco" not in ALGORITHMS:
    ALGORITHMS.register("feddisco", FedDiscoBuilder())
```

- [ ] **Step 5: Run strategy tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_registry tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_builder_creates_strategy tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_strategy_gives_higher_weight_to_lower_discrepancy tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_strategy_falls_back_when_relu_scores_are_zero -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_registry.py tests/test_model_and_algorithm.py src/fl_baselines/algorithms/feddisco.py src/fl_baselines/defaults.py
git commit -m "feat: add FedDisco strategy"
```

## Task 4: Client Route

**Files:**
- Modify: `tests/test_model_and_algorithm.py`
- Modify: `src/fl_baselines/clients/torch_client.py`

- [ ] **Step 1: Write failing client routing test**

Add test:

```python
def test_torch_flower_client_routes_feddisco_fit(self) -> None:
    config = ExperimentConfig.from_run_config(
        {
            "algorithm": "feddisco",
            "num-classes": 2,
            "feddisco-metric": "kl",
        }
    )
    model = torch.nn.Linear(1, 2)
    loaders = _client_loaders(
        TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
    )
    client = TorchFlowerClient(model, loaders, config, client_id="feddisco-route")

    with patch.object(
        torch_client_module,
        "train_feddisco_client",
        return_value={"loss": 0.5, "feddisco_discrepancy": 0.25},
    ) as train_mock:
        _, _, metrics = client.fit(
            get_model_parameters(model),
            {
                "algorithm": "feddisco",
                "local_epochs": 1,
                "learning_rate": 0.1,
                "num_classes": 2,
                "feddisco_metric": "kl",
                "feddisco_epsilon": 1e-8,
            },
        )

    train_mock.assert_called_once()
    self.assertIn("feddisco_discrepancy", metrics)
```

- [ ] **Step 2: Run targeted test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_torch_flower_client_routes_feddisco_fit -v
```

Expected: FAIL because `TorchFlowerClient.fit` does not route `feddisco`.

- [ ] **Step 3: Implement client route**

Import:

```python
from fl_baselines.training.feddisco import train_feddisco_client
```

Add route:

```python
if algorithm == "feddisco":
    return self._fit_feddisco(parameters, config)
```

Add method:

```python
def _fit_feddisco(
    self,
    parameters: list[np.ndarray],
    config: dict[str, bool | bytes | float | int | str],
) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
    set_model_parameters(self.model, parameters)
    local_epochs = int(config.get("local_epochs", self.config.local_epochs))
    learning_rate = float(config.get("learning_rate", self.config.learning_rate))
    num_classes = int(config.get("num_classes", self.config.num_classes))
    metric = str(config.get("feddisco_metric", self.config.feddisco_metric))
    epsilon = float(config.get("feddisco_epsilon", self.config.feddisco_epsilon))
    metrics = train_feddisco_client(
        self.model,
        self.loaders.train,
        epochs=local_epochs,
        learning_rate=learning_rate,
        device=self.config.device,
        num_classes=num_classes,
        metric=metric,
        epsilon=epsilon,
    )
    return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics
```

- [ ] **Step 4: Run client route test and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_torch_flower_client_routes_feddisco_fit -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_and_algorithm.py src/fl_baselines/clients/torch_client.py
git commit -m "feat: route FedDisco client training"
```

## Task 5: Docs And Compatibility

**Files:**
- Modify: `tests/test_model_and_algorithm.py`
- Modify: `README.md`
- Modify: docs algorithm index files if they already list all algorithms

- [ ] **Step 1: Add current-model builder smoke test**

Add:

```python
def test_feddisco_builder_supports_current_models(self) -> None:
    cases = [
        (MnistCnnBuilder(), {}),
        (LeNetBuilder(), {}),
        (ResNet9Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}),
        (ResNet18Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}),
        (ResNet34Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}),
        (InceptionBuilder(), {"input-channels": 3, "input-height": 75, "input-width": 75}),
    ]

    for model_builder, overrides in cases:
        with self.subTest(model=model_builder.name):
            config = ExperimentConfig.from_run_config(
                {"algorithm": "feddisco", "num-supernodes": 2, **overrides}
            )
            model = model_builder.build_model(config)
            strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
            self.assertIsInstance(strategy, FedDiscoStrategy)
            self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "feddisco")
```

- [ ] **Step 2: Run smoke test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_feddisco_builder_supports_current_models -v
```

Expected: PASS after Task 3.

- [ ] **Step 3: Update docs**

Update README algorithm table/list with:

```markdown
| `feddisco` | FedDisco: discrepancy-aware collaboration for non-IID label distributions. Paper: https://proceedings.mlr.press/v202/ye23f.html |
```

If docs algorithm index files list every baseline, add `feddisco` there too with the same concise description.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_and_algorithm.py README.md docs
git commit -m "docs: document FedDisco baseline"
```

## Task 6: Final Verification

**Files:**
- All changed files

- [ ] **Step 1: Run focused suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config tests.test_registry tests.test_model_and_algorithm -v
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 3: Compile**

Run:

```bash
python -m compileall src tests
```

Expected: PASS with no syntax errors.

- [ ] **Step 4: Remove generated bytecode caches**

Run:

```bash
rm -rf ./tests/__pycache__ ./src/fl_baselines/clients/__pycache__ ./src/fl_baselines/core/__pycache__ ./src/fl_baselines/app/__pycache__ ./src/fl_baselines/training/__pycache__ ./src/fl_baselines/datasets/__pycache__ ./src/fl_baselines/algorithms/__pycache__ ./src/fl_baselines/models/__pycache__ ./src/fl_baselines/__pycache__ ./src/fl_baselines/logging/__pycache__
```

Expected: generated caches removed.

- [ ] **Step 5: Final commit if needed**

If verification required changes:

```bash
git add .
git commit -m "test: verify FedDisco integration"
```

## Self-Review

- Spec coverage: covers config, helper, client metrics, server aggregation, fallback, registry, docs, and verification.
- Placeholder scan: no `TBD`, no vague “handle edge cases” steps without concrete behavior.
- Type consistency: uses `feddisco_discrepancy_weight`, `feddisco_bias`, `feddisco_metric`, `feddisco_epsilon`, and metric key `feddisco_discrepancy` consistently.
