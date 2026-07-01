# Evaluation Macro Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add macro `precision`, `recall`, and `f1` to the shared evaluation path so all baselines report them for both server and client evaluation.

**Architecture:** Extend `evaluate_model(...)` in one place and keep the current `(loss, metrics)` contract unchanged. Because both server and client evaluation already use this helper, the new metrics propagate automatically without strategy-specific changes.

**Tech Stack:** Python, PyTorch, Flower, unittest

---

### Task 1: Lock Expected Evaluation Behavior With Tests

**Files:**
- Modify: `tests/test_model_and_algorithm.py`
- Test: `tests/test_model_and_algorithm.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- `evaluate_model(...)` returns `accuracy`, `precision`, `recall`, and `f1`
- tuple-output models still work
- zero-denominator class cases do not produce errors and return finite metric values

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_evaluate_model_returns_macro_classification_metrics tests.test_model_and_algorithm.ModelAndAlgorithmTest.test_evaluate_model_handles_missing_class_predictions_in_macro_metrics -v`

Expected: FAIL because the metrics are not present yet.

- [ ] **Step 3: Write minimal implementation**

Extend `src/fl_baselines/training/evaluate.py` to accumulate predictions/targets and compute macro metrics using per-class counts from the logits class dimension.

- [ ] **Step 4: Run targeted tests to verify they pass**

Run the same targeted command and confirm PASS.

- [ ] **Step 5: Commit**

Skip commit unless explicitly requested by the user.

### Task 2: Document The New Evaluation Metrics

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/testing-and-artifacts.md`

- [ ] **Step 1: Update user-facing docs**

Document that evaluation now reports:
- `accuracy`
- `precision`
- `recall`
- `f1`

Clarify that `precision`, `recall`, and `f1` use macro averaging for multi-class classification.

- [ ] **Step 2: Run a quick docs sanity check**

Review the edited sections for terminology consistency with the existing server/client evaluation semantics.

- [ ] **Step 3: Commit**

Skip commit unless explicitly requested by the user.

### Task 3: Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Run full unit test suite**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Run compile verification**

Run: `python -m compileall src tests`

Expected: exit code 0.

- [ ] **Step 3: Remove generated caches**

Run the approved `rm -rf` command for repo `__pycache__` directories.

- [ ] **Step 4: Summarize verification evidence**

Report the exact commands run and whether they passed.
