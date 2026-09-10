import numpy as np
import torch

from yolo_r2plus1d.strict_v3.models.conditional_corrector import (
    ConditionalCorrector,
    available_correction,
)
from yolo_r2plus1d.strict_v3.training.available_corrector import available_split


def test_unavailable_nan_inputs_cannot_modify_baseline():
    model = ConditionalCorrector(visual_dim=4, classes=4, hidden_dim=6).eval()
    with torch.no_grad():
        model.output.bias.fill_(2)
    visual = torch.tensor([[float("nan")] * 4, [1., 2., 3., 4.]])
    temporal = visual.clone()
    baseline = torch.randn(2, 4)
    actual = available_correction(model, visual, temporal, baseline, torch.tensor([False, True]))
    assert torch.equal(actual[0], baseline[0])
    torch.testing.assert_close(actual[1], baseline[1] + 2)
    assert torch.isfinite(actual).all()


def test_training_excludes_held_subjects_and_unavailable_rows():
    users = np.array([1, 1, 2, 2, 3, 3])
    valid = np.array([True, False, True, False, False, True])
    train, held = available_split(users, valid, [1])
    np.testing.assert_array_equal(train, [2, 5])
    np.testing.assert_array_equal(held, [0, 1])
