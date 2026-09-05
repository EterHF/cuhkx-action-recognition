import numpy as np
import torch

from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier
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
