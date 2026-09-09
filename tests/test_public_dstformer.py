import torch

from yolo_r2plus1d.strict_v3.models.public_dstformer import Net


def test_forward_features_are_exact_fc2_input() -> None:
    model = Net().eval()
    inputs = torch.zeros(2, 1, 17, 3)
    with torch.inference_mode():
        features = model.forward_features(inputs)
        logits = model(inputs)
        expected = model.head["fc2"](features)
    assert features.shape == (2, 2048)
    assert torch.equal(logits, expected)
