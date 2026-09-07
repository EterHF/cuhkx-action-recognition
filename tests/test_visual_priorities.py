import torch

from yolo_r2plus1d.strict_v3.models.visual_priorities import (
    ModalityGate,
    VisualPriorityClassifier,
)


def test_modality_gate_is_identity_at_initialization() -> None:
    inputs = torch.randn(2, 4, 4, 8, 8)
    torch.testing.assert_close(ModalityGate()(inputs), inputs)


def test_highres_variant_removes_only_layer4_temporal_stride() -> None:
    control = VisualPriorityClassifier("multiscale")
    highres = VisualPriorityClassifier("highres")
    assert control.encoder.layer4[0].conv1[0][3].stride == (2, 1, 1)
    assert control.encoder.layer4[0].downsample[0].stride == (2, 2, 2)
    assert highres.encoder.layer4[0].conv1[0][3].stride == (1, 1, 1)
    assert highres.encoder.layer4[0].downsample[0].stride == (1, 2, 2)


def test_multiscale_residual_is_zero_at_initialization() -> None:
    model = VisualPriorityClassifier("multiscale").eval()
    inputs = torch.randn(1, 16, 4, 32, 32)
    with torch.inference_mode():
        observed = model(inputs)
        model.temporal_head = None
        expected = model(inputs)
    torch.testing.assert_close(observed, expected)


def test_modality_gate_variant_preserves_shared_initialization() -> None:
    torch.manual_seed(2026)
    highres = VisualPriorityClassifier("highres")
    torch.manual_seed(2026)
    gated = VisualPriorityClassifier("modality_gate")

    gated_state = gated.state_dict()
    for key, value in highres.state_dict().items():
        assert torch.equal(value, gated_state[key]), key
