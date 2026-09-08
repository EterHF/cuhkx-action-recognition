import numpy as np
import pytest
import torch

from yolo_r2plus1d.strict_v3.models.conditional_corrector import ConditionalCorrector
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    correction_loss,
    paired_exact_pvalue,
    prediction_delta,
    validate_provenance,
)
from yolo_r2plus1d.strict_v3.training.deployment_corrector import promotion_passed


def test_corrector_is_exact_zero_residual_but_hidden_layer_is_not_zero() -> None:
    torch.manual_seed(2026)
    model = ConditionalCorrector(visual_dim=8, classes=4, hidden_dim=6).eval()
    residual = model(torch.randn(3, 8), torch.randn(3, 4))
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.count_nonzero(model.hidden[0].weight) > 0
    assert torch.count_nonzero(model.output.weight) == 0


def test_loss_protects_only_correct_confident_baseline_rows() -> None:
    baseline = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 0.0]])
    candidate = baseline.clone().requires_grad_()
    labels = torch.tensor([0, 1, 0])
    loss, protected = correction_loss(baseline, candidate, labels, 0.8, 1.0)
    assert protected == 1
    loss.backward()
    assert candidate.grad is not None


def test_paired_metrics_reward_net_corrections() -> None:
    labels = np.array([0, 1, 0, 1])
    candidate = np.array([0, 1, 0, 0])
    users = np.array([1, 1, 2, 2])
    metrics = classification_metrics(candidate, labels, users)
    assert metrics["correct"] == 3
    assert metrics["worst_user_accuracy"] == 0.5
    assert paired_exact_pvalue(corrected=3, broken=0) == 0.25


def test_nested_provenance_fails_loud_on_outer_or_inner_leakage() -> None:
    safe = {
        "upstream_excluded_users": [1, 6],
        "cross_fitted_within_outer_train": True,
        "target_labels_used_for_upstream_selection": False,
    }
    validate_provenance(safe, {1, 6}, training=True)
    with pytest.raises(RuntimeError, match="outer-held"):
        validate_provenance({**safe, "upstream_excluded_users": [1]}, {1, 6}, True)
    with pytest.raises(RuntimeError, match="inner cross-fitted"):
        validate_provenance({**safe, "cross_fitted_within_outer_train": False}, {1, 6}, True)


def test_prediction_delta_uses_net_corrections() -> None:
    labels = np.array([0, 0, 0])
    assert prediction_delta(np.array([1, 0, 0]), np.array([0, 1, 0]), labels) == {
        "corrected": 1,
        "broken": 1,
        "net": 0,
    }


def test_deployment_promotion_gate_fails_on_one_bad_fold() -> None:
    baseline = {"subject_macro_accuracy": 0.95, "worst_user_accuracy": 0.8}
    candidate = {"subject_macro_accuracy": 0.96, "worst_user_accuracy": 0.81}
    folds = [{"net": 2}, {"net": 1}, {"net": 1}, {"net": 0}, {"net": -1}]
    assert not promotion_passed(folds, {"net": 3}, baseline, candidate)


def test_deployment_promotion_gate_requires_user_metric_non_degradation() -> None:
    baseline = {"subject_macro_accuracy": 0.95, "worst_user_accuracy": 0.8}
    candidate = {"subject_macro_accuracy": 0.96, "worst_user_accuracy": 0.79}
    folds = [{"net": 1} for _ in range(5)]
    assert not promotion_passed(folds, {"net": 5}, baseline, candidate)
