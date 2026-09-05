import argparse
import json

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.training.finetune_public_omnivore import prepare_inputs
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import (
    TemporalProbe,
    combine_features,
    defm_preprocess,
    depth_color_to_inverse,
    softmax_numpy,
)


def test_depth_colour_conversion_marks_black_invalid() -> None:
    frames = torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8)
    frames[:, :, 0, 2:6, 2:6] = 255
    inverse = depth_color_to_inverse(frames)
    assert torch.count_nonzero(inverse[:, :, :2]) == 0
    assert torch.count_nonzero(inverse[:, :, 2:6, 2:6]) > 0


def test_defm_preprocessing_and_probe_shapes() -> None:
    inverse = torch.rand(6, 32, 32)
    transformed = defm_preprocess(inverse, image_size=56)
    assert transformed.shape == (6, 3, 56, 56)
    assert torch.isfinite(transformed).all()
    probe = TemporalProbe(24)
    assert probe(torch.randn(3, 16, 24)).shape == (3, 40)


def test_combine_broadcasts_clip_feature_over_frames(tmp_path) -> None:
    frame_path = tmp_path / "frame.npy"
    clip_path = tmp_path / "clip.npy"
    output = tmp_path / "combined.npy"
    np.save(frame_path, np.zeros((3, 4, 2), dtype=np.float16))
    np.save(clip_path, np.ones((3, 1, 5), dtype=np.float16))
    for path, model, modality, parameters in (
        (frame_path, "frame", "depth", 2),
        (clip_path, "clip", "ir", 5),
    ):
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "model": model,
                    "modality": modality,
                    "encoder_parameters": parameters,
                }
            )
        )
    combine_features(argparse.Namespace(features=[frame_path, clip_path], output=output))
    combined = np.load(output)
    assert combined.shape == (3, 4, 7)
    assert np.all(combined[..., 2:] == 1)


def test_softmax_numpy_is_stable_and_normalised() -> None:
    probabilities = softmax_numpy(np.array([[1_000.0, 1_001.0], [-1_000.0, -1_000.0]]))

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[1], (0.5, 0.5))


def test_omnivore_preprocessing_preserves_video_shape() -> None:
    frames = torch.randint(0, 256, (2, 4, 4, 16, 16), dtype=torch.uint8)

    for modality in ("depth", "ir"):
        inputs = prepare_inputs(frames, modality, augment=False)
        assert inputs.shape == (2, 3, 4, 16, 16)
        assert torch.isfinite(inputs).all()
