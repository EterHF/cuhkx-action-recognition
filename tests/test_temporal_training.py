from pathlib import Path

import numpy as np
import pytest
import torch

from yolo_r2plus1d.strict_v3.training.public_finetune import visual_member_state
from yolo_r2plus1d.strict_v3.training.temporal import Residual, pred, sha256


def test_temporal_prediction_supports_cpu(tmp_path: Path) -> None:
    logits_path = tmp_path / "frame_logits.npy"
    logits = np.zeros((3, 16, 40), dtype=np.float32)
    logits[:, :, 7] = 2.0
    np.save(logits_path, logits)

    model = Residual("tcn")
    output = pred(
        model,
        logits_path,
        np.arange(3),
        np.zeros(3, dtype=np.int64),
        torch.device("cpu"),
        batch=2,
        workers=0,
    )

    assert output.shape == (3, 40)
    assert np.isfinite(output).all()
    assert (output.argmax(1) == 7).all()
    assert len(sha256(logits_path)) == 64


def test_visual_member_state_supports_compact_release_schema() -> None:
    state = {"head.weight": object()}
    package = {"visual_member0": {"model_state_packed": state, "bits": 5}}

    assert visual_member_state(package, 0) is state
    with pytest.raises(IndexError, match="only visual member 0"):
        visual_member_state(package, 1)
