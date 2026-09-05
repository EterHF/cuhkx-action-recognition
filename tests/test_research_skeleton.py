import numpy as np
import torch
from torch import nn

from yolo_r2plus1d.strict_v3.models.omnivore_layer_fusion import OmnivoreLayerFusion
from yolo_r2plus1d.strict_v3.models.omnivore_rgbd import OmnivoreRGBDClassifier
from yolo_r2plus1d.strict_v3.models.public_sensor_fusion import PublicSensorFusion
from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier
from yolo_r2plus1d.strict_v3.training.finetune_omnivore_rgbd import prepare_rgbd
from yolo_r2plus1d.strict_v3.training.research_oof import FOLDS, split


def test_padding_is_masked() -> None:
    torch.manual_seed(3)
    model = HighRateSkeletonClassifier(dropout=0.0).eval()
    skeleton = torch.randn(2, 12, 17, 3)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, :7] = True
    positions = torch.zeros(2, 12)
    positions[:, :7] = torch.linspace(0, 1, 7)
    changed = skeleton.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100
    with torch.inference_mode():
        expected = model(skeleton, mask, positions)
        actual = model(changed, mask, positions)
    torch.testing.assert_close(actual, expected)


def test_subject_folds_are_disjoint_and_complete(tmp_path) -> None:
    users = np.repeat(np.array(sorted({user for values in FOLDS.values() for user in values})), 2)
    labels = np.arange(len(users)) % 40
    metadata = tmp_path / "metadata.npz"
    np.savez(metadata, train_y=labels, train_users=users)
    held_rows = []
    for fold in FOLDS:
        _, observed_users, train_indices, held_indices = split(metadata, fold)
        assert not np.intersect1d(observed_users[train_indices], observed_users[held_indices]).size
        held_rows.extend(held_indices.tolist())
    assert sorted(held_rows) == list(range(len(users)))


def test_public_sensor_fusion_ignores_padded_skeleton() -> None:
    torch.manual_seed(7)
    model = PublicSensorFusion(depth_dim=12, ir_dim=10).eval()
    depth = torch.randn(2, 1, 12)
    infrared = torch.randn(2, 1, 10)
    skeleton = torch.randn(2, 12, 17, 3)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, :7] = True
    positions = torch.zeros(2, 12)
    positions[:, :7] = torch.linspace(0, 1, 7)
    changed = skeleton.clone()
    changed[:, 7:] = 100 * torch.randn_like(changed[:, 7:])

    with torch.inference_mode():
        expected = model(depth, infrared, skeleton, mask, positions)
        actual = model(depth, infrared, changed, mask, positions)
    torch.testing.assert_close(actual, expected)


def test_public_sensor_fusion_falls_back_when_skeleton_is_missing() -> None:
    model = PublicSensorFusion(depth_dim=12, ir_dim=10).eval()
    common = (
        torch.randn(2, 1, 10),
        torch.zeros(2, 12, 17, 3),
        torch.zeros(2, 12, dtype=torch.bool),
        torch.zeros(2, 12),
    )
    depth = torch.arange(12, dtype=torch.float32).view(1, 1, 12).repeat(2, 1, 1)
    with torch.inference_mode():
        first = model(depth, *common)
        second = model(depth.square(), *common)
    assert not torch.equal(first, second)


def test_public_sensor_fusion_uses_temporal_sensor_order() -> None:
    torch.manual_seed(11)
    model = PublicSensorFusion(depth_dim=12, ir_dim=10).eval()
    depth = torch.randn(2, 8, 12)
    infrared = torch.randn(2, 8, 10)
    skeleton = torch.randn(2, 12, 17, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)
    positions = torch.linspace(0, 1, 12).repeat(2, 1)
    with torch.inference_mode():
        ordered = model(depth, infrared, skeleton, mask, positions)
        reversed_depth = model(depth.flip(1), infrared, skeleton, mask, positions)
    assert not torch.equal(ordered, reversed_depth)


class FakeOmnivore(nn.Module):
    def forward(self, inputs: torch.Tensor, out_feat_keys: list[str]):
        assert len(out_feat_keys) == 4
        batch = len(inputs)
        return [
            torch.randn(batch, channels, 8, size, size, device=inputs.device)
            for channels, size in ((192, 8), (384, 4), (768, 2), (768, 2))
        ]


def test_omnivore_layer_fusion_returns_main_and_auxiliary_logits() -> None:
    model = OmnivoreLayerFusion(FakeOmnivore(), FakeOmnivore())
    inputs = torch.randn(2, 3, 16, 32, 32)
    main, depth, infrared = model(inputs, inputs, return_aux=True)
    assert main.shape == depth.shape == infrared.shape == (2, 40)


class FakeRGBDOmnivore(nn.Module):
    def forward(self, inputs: torch.Tensor):
        assert inputs.shape[1] == 4
        return inputs.mean(dim=(2, 3, 4)).repeat(1, 192)


def test_omnivore_rgbd_uses_single_four_channel_trunk() -> None:
    model = OmnivoreRGBDClassifier(FakeRGBDOmnivore())
    assert model(torch.randn(2, 4, 16, 32, 32)).shape == (2, 40)
    with np.testing.assert_raises_regex(ValueError, "shape"):
        model(torch.randn(2, 3, 16, 32, 32))


def test_rgbd_preparation_repeats_ir_and_retains_depth_channel() -> None:
    frames = torch.zeros(2, 3, 4, 8, 8, dtype=torch.uint8)
    frames[:, :, 3] = 255
    values = prepare_rgbd(
        frames,
        mean=torch.tensor([0.0, 0.0]),
        std=torch.tensor([1.0, 1.0]),
        augment=False,
    )
    assert values.shape == (2, 4, 3, 8, 8)
    torch.testing.assert_close(values[:, 0], values[:, 1])
    torch.testing.assert_close(values[:, 1], values[:, 2])
    assert torch.all(values[:, :3] == 1.0)
