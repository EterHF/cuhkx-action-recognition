import json

import pytest
import torch

from yolo_r2plus1d.strict_v3.export.int8_state import pack_state, unpack_state
from yolo_r2plus1d.strict_v3.export.public_sensor_bundle import validated_state


def test_int8_state_round_trip_preserves_keys_and_integer_buffers() -> None:
    state = {
        "weight": torch.tensor([[1.0, -0.5], [0.01, -0.02]]),
        "bias": torch.tensor([0.3, -0.7]),
        "generated_index": torch.arange(4),
    }
    restored = unpack_state(pack_state(state), state)
    assert restored.keys() == state.keys()
    torch.testing.assert_close(restored["weight"], state["weight"], atol=0.004, rtol=0.01)
    torch.testing.assert_close(restored["bias"], state["bias"], atol=0.001, rtol=0.001)
    torch.testing.assert_close(restored["generated_index"], state["generated_index"])


def test_bundle_rejects_non_full_fit_or_wrong_modality(tmp_path) -> None:
    directory = tmp_path / "depth"
    directory.mkdir()
    torch.save({"weight": torch.ones(2, 2)}, directory / "model.pt")
    metrics = {
        "protocol": "full-fit-public-encoder/v1",
        "modality": "depth",
        "project_checkpoint_loaded": False,
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    assert "weight" in validated_state(directory, "depth")

    with pytest.raises(RuntimeError, match="expected ir"):
        validated_state(directory, "ir")
    metrics["protocol"] = "single-subject-fold-public-encoder-finetune/v1"
    (directory / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="not a full-fit"):
        validated_state(directory, "depth")
