# Client Test Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every baseline use server evaluation on the server test set and client evaluation on a deterministic held-out test split from each client's own local partition.

**Architecture:** Add a new `client-test-fraction` config, rename client loader semantics from `validation` to `test`, and update dataset builders so client loaders come from a split of the client training partition rather than the global test split. Keep server evaluation on the dataset test split exactly as before.

**Tech Stack:** Python 3.10+, Flower 1.26.1, PyTorch 2.8.0, `unittest`, NumPy

---

## File Map

- `pyproject.toml`
  Add default `client-test-fraction`.
- `src/fl_baselines/core/config.py`
  Parse and validate `client_test_fraction`.
- `src/fl_baselines/core/types.py`
  Rename client loader field from `validation` to `test`.
- `src/fl_baselines/datasets/base.py`
  Update protocol docstrings to reflect train/test client loaders.
- `src/fl_baselines/datasets/vision.py`
  Split client partitions into train/test subsets deterministically.
- `src/fl_baselines/clients/torch_client.py`
  Use `loaders.test` in client evaluation.
- `tests/test_config.py`
  Config coverage for `client-test-fraction`.
- `tests/test_datasets.py`
  Client train/test split behavior coverage.
- `tests/test_model_and_algorithm.py`
  Update any loader field-name assumptions.
- `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/quickstart.md`, `docs/testing-and-artifacts.md`
  Clarify evaluation semantics.

### Task 1: Add Red Tests For Config And Dataset Split

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing config test**

In `tests/test_config.py`, add `"client-test-fraction": 0.25` to the override case and assert:

```python
        self.assertEqual(config.client_test_fraction, 0.25)
```

Add validation coverage:

```python
        with self.assertRaisesRegex(ValueError, "client-test-fraction must be in"):
            ExperimentConfig.from_run_config({"client-test-fraction": 0.0})

        with self.assertRaisesRegex(ValueError, "client-test-fraction must be in"):
            ExperimentConfig.from_run_config({"client-test-fraction": 1.0})
```

- [ ] **Step 2: Write the failing dataset split tests**

In `tests/test_datasets.py`, change the client loader assertions to use `loaders.test` and add:

```python
    def test_torchvision_builder_splits_client_partition_into_train_and_test(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"batch-size": 2, "client-test-fraction": 0.5, "seed": 7}
        )

        loaders = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )

        self.assertGreater(len(loaders.train.dataset), 0)
        self.assertGreater(len(loaders.test.dataset), 0)
        self.assertEqual(
            len(loaders.train.dataset) + len(loaders.test.dataset),
            3,
        )
```

and:

```python
    def test_torchvision_builder_client_split_is_deterministic(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"batch-size": 2, "client-test-fraction": 0.5, "seed": 11}
        )

        first = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )
        second = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )

        self.assertEqual(len(first.train.dataset), len(second.train.dataset))
        self.assertEqual(len(first.test.dataset), len(second.test.dataset))
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run:

```bash
python -m unittest tests.test_config tests.test_datasets -v
```

Expected:
- missing `client_test_fraction`
- missing `test` field on `ClientDataLoaders`
- old dataset builder behavior still using the global test split

### Task 2: Make Config And Loader Semantics Pass

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/fl_baselines/core/config.py`
- Modify: `src/fl_baselines/core/types.py`
- Modify: `src/fl_baselines/datasets/base.py`

- [ ] **Step 1: Add the config default**

Add to `pyproject.toml`:

```toml
client-test-fraction = 0.2
```

- [ ] **Step 2: Add parsed config support**

In `src/fl_baselines/core/config.py`, add:

```python
    client_test_fraction: float = 0.2
```

parse it:

```python
            client_test_fraction=_as_float(
                run_config,
                "client-test-fraction",
                cls.client_test_fraction,
            ),
```

validate it:

```python
        if not 0 < self.client_test_fraction < 1:
            raise ValueError("client-test-fraction must be in (0, 1)")
```

- [ ] **Step 3: Rename the client loader field**

In `src/fl_baselines/core/types.py`, change:

```python
class ClientDataLoaders:
    train: "DataLoader"
    test: "DataLoader"
```

In `src/fl_baselines/datasets/base.py`, update the docstring to say `train and test loaders`.

- [ ] **Step 4: Run the targeted tests**

Run:

```bash
python -m unittest tests.test_config -v
```

Expected:
- config tests pass
- dataset tests still fail until the builder logic changes

### Task 3: Implement Client Train/Test Split

**Files:**
- Modify: `src/fl_baselines/datasets/vision.py`
- Modify: `tests/test_datasets.py`

- [ ] **Step 1: Update `build_client_loaders(...)`**

In `src/fl_baselines/datasets/vision.py`:

- stop loading the dataset test split for client evaluation
- use only `train_dataset = self._load(train=True, config=config)` for client partitioning
- compute `partition_indices`
- split that partition deterministically into train/test subsets
- return:

```python
        return ClientDataLoaders(
            train=DataLoader(
                Subset(train_dataset, client_train_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=True,
            ),
            test=DataLoader(
                Subset(train_dataset, client_test_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=False,
            ),
        )
```

- [ ] **Step 2: Add deterministic split helper**

Add a private helper similar to:

```python
    def _split_client_train_test(
        self,
        indices: np.ndarray,
        config: ExperimentConfig,
        partition_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(indices) < 2:
            return indices, indices[:0]

        rng = np.random.default_rng(config.seed + 10_000 + partition_id)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        test_size = int(round(len(shuffled) * config.client_test_fraction))
        test_size = max(1, min(len(shuffled) - 1, test_size))
        test_indices = shuffled[:test_size]
        train_indices = shuffled[test_size:]
        return train_indices, test_indices
```

- [ ] **Step 3: Keep server loader unchanged**

`build_server_loader(...)` should still use:

```python
        dataset = self._load(train=False, config=config)
```

- [ ] **Step 4: Run the dataset tests**

Run:

```bash
python -m unittest tests.test_datasets -v
```

Expected:
- dataset tests pass

### Task 4: Update Client Evaluation Path

**Files:**
- Modify: `src/fl_baselines/clients/torch_client.py`
- Modify: `tests/test_model_and_algorithm.py`

- [ ] **Step 1: Swap client evaluation to `loaders.test`**

In `src/fl_baselines/clients/torch_client.py`, change:

```python
            self.loaders.validation,
```

to:

```python
            self.loaders.test,
```

- [ ] **Step 2: Update any test loader mocks if needed**

Where fake loaders are constructed in `tests/test_model_and_algorithm.py`, replace `validation` with `test`:

```python
{"train": loader, "test": loader}
```

- [ ] **Step 3: Run the model/algorithm tests**

Run:

```bash
python -m unittest tests.test_model_and_algorithm -v
```

Expected:
- all model and algorithm tests pass with the renamed client test loader

### Task 5: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/testing-and-artifacts.md`

- [ ] **Step 1: Document the new config**

Add `client-test-fraction = 0.2` to the relevant config examples in `docs/quickstart.md`.

- [ ] **Step 2: Clarify evaluation semantics**

Add short notes stating:
- server eval uses server test set
- client eval uses held-out client test split from each client partition

- [ ] **Step 3: Mention the renamed semantics**

Where docs say `validation` for client evaluation, update to `test`.

### Task 6: Full Verification And Cleanup

**Files:**
- Verify existing modified files

- [ ] **Step 1: Run the full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Expected:
- all tests pass

- [ ] **Step 2: Run compile verification**

Run:

```bash
python -m compileall src tests
```

Expected:
- exit code 0

- [ ] **Step 3: Remove generated caches**

Run:

```bash
rm -rf ./tests/__pycache__ ./src/fl_baselines/clients/__pycache__ ./src/fl_baselines/core/__pycache__ ./src/fl_baselines/app/__pycache__ ./src/fl_baselines/training/__pycache__ ./src/fl_baselines/datasets/__pycache__ ./src/fl_baselines/algorithms/__pycache__ ./src/fl_baselines/models/__pycache__ ./src/fl_baselines/__pycache__ ./src/fl_baselines/logging/__pycache__
```

- [ ] **Step 4: Inspect the working tree**

Run:

```bash
git status --short
```

Expected:
- only intended evaluation-semantics changes remain in addition to any already-existing local changes

## Self-Review

- Spec coverage:
  - config, loader rename, deterministic train/test split, client eval path, tests, and docs are all covered.
- Placeholder scan:
  - no `TODO`, `TBD`, or omitted code steps remain.
- Type consistency:
  - `client_test_fraction`, `ClientDataLoaders.test`, and `client-test-fraction` are named consistently throughout.
