import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import (
    bits_for,
    prediction_delta,
    quantized_state,
)
from yolo_r2plus1d.strict_v3.training.public_finetune import dequantize_state


def test_mixed_policy_assigns_only_late_layers_more_bits() -> None:
    assert bits_for("late8", "encoder.layer3.0.weight") == 5
    assert bits_for("late8", "encoder.layer4.0.weight") == 8
    assert bits_for("late8", "head.1.weight") == 8
    assert bits_for("mixed6_late8", "encoder.layer3.0.weight") == 6
    assert bits_for("mixed4_late8", "encoder.layer3.0.weight") == 4


def test_quantized_state_uses_original_float_tensor_not_packed_input() -> None:
    state = {
        "encoder.layer4.weight": torch.tensor([[0.2, -0.7]]),
        "counter": torch.tensor(4),
    }
    packed = quantized_state(state, "late8")
    assert packed["encoder.layer4.weight"]["bits"] == 8
    assert torch.equal(dequantize_state(packed)["counter"], state["counter"])


def test_prediction_delta_reports_corrections_and_regressions() -> None:
    labels = np.array([0, 0, 0, 0])
    baseline = np.array([1, 0, 0, 1])
    candidate = np.array([0, 1, 0, 1])
    assert prediction_delta(baseline, candidate, labels) == {
        "corrected": 1,
        "broken": 1,
        "net": 0,
    }
