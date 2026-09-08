from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from yolo_r2plus1d.strict_v3.data.skeleton_retarget import retarget_bone_lengths
from yolo_r2plus1d.strict_v3.data.validity import (
    has_decodable_image,
    load_validity_mask,
    masked_classification_loss,
)


def test_image_validity_requires_a_decodable_recorded_image(tmp_path: Path) -> None:
    modality = tmp_path / "IR"
    modality.mkdir()
    (modality / "broken.png").write_bytes(b"not an image")
    assert not has_decodable_image(modality)
    Image.new("L", (2, 2)).save(modality / "valid.png")
    assert has_decodable_image(modality)


def test_validity_mask_contract_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "validity.npz"
    np.savez_compressed(path, visual=np.array([True, False]))
    assert np.array_equal(load_validity_mask(path, "visual", 2), [True, False])
    with pytest.raises(KeyError, match="skeleton"):
        load_validity_mask(path, "skeleton", 2)
    with pytest.raises(ValueError, match=r"bool\[3\]"):
        load_validity_mask(path, "visual", 3)


def test_masked_loss_retains_forward_rows_but_removes_invalid_gradient() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 0])
    valid = torch.tensor([True, False])
    loss = masked_classification_loss(
        torch.nn.CrossEntropyLoss(), logits, labels, valid
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_retargeting_preserves_root_confidence_direction_and_projected_motion() -> None:
    sequence = np.zeros((2, 17, 3), dtype=np.float32)
    sequence[..., 2] = 1.0
    sequence[0, 1, :2] = [1.0, 0.0]
    sequence[1, 1, :2] = [0.0, 2.0]
    scales = np.ones(17, dtype=np.float32)
    scales[1] = 1.1
    changed = retarget_bone_lengths(sequence, scales)
    assert np.array_equal(changed[:, 0], sequence[:, 0])
    assert np.array_equal(changed[..., 2], sequence[..., 2])
    lengths = np.linalg.norm(changed[:, 1, :2] - changed[:, 0, :2], axis=1)
    assert np.allclose(lengths, [1.1, 2.2])
    assert np.allclose(changed[0, 1, 1], 0.0)
    assert np.allclose(changed[1, 1, 0], 0.0)
