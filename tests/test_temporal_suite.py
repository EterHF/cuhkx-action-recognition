import json
import sys
from pathlib import Path

import numpy as np
import pytest

from yolo_r2plus1d.strict_v3.training import temporal_suite


def test_oof_only_skips_full_jobs_and_test_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    frame_logits = tmp_path / "train_frame_logits.npy"
    metadata = tmp_path / "metadata.npz"
    output = tmp_path / "run"
    np.save(frame_logits, np.zeros((2, 16, 40), dtype=np.float32))
    np.savez(
        metadata,
        train_y=np.array([0, 1], dtype=np.int64),
        train_users=np.array([1, 2], dtype=np.int64),
    )
    commands: list[list[str]] = []

    def record_job(command: list[str], job_output: Path, resume: bool) -> None:
        commands.append(command)

    monkeypatch.setattr(temporal_suite, "run_job", record_job)
    monkeypatch.setattr(
        temporal_suite,
        "assemble_oof",
        lambda root, seeds, metadata_path: (
            np.zeros((2, 40), dtype=np.float32),
            {
                "seed_accuracy": {"2026": 0.0},
                "seed_mean_accuracy": 0.0,
                "fold_seed_accuracy": {},
            },
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal_suite",
            "--recipe",
            "strict",
            "--frame-logits",
            str(frame_logits),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
            "--oof-only",
        ],
    )

    temporal_suite.main()

    assert len(commands) == 5
    assert all("--test-logits" not in command for command in commands)
    assert not list(output.glob("full_seed*"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["oof_only"] is True
    assert receipt["test_data_loaded"] is False
    assert receipt["inputs"]["test_frame_logits"] is None
    assert receipt["outputs"]["test_sha256"] is None


def test_oof_only_refuses_a_test_logit_argument(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    frame_logits = tmp_path / "train.npy"
    test_logits = tmp_path / "test.npy"
    metadata = tmp_path / "metadata.npz"
    np.save(frame_logits, np.zeros((1, 16, 40), dtype=np.float32))
    np.save(test_logits, np.zeros((1, 16, 40), dtype=np.float32))
    np.savez(metadata, train_y=np.array([0]), train_users=np.array([1]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal_suite",
            "--recipe",
            "strict",
            "--frame-logits",
            str(frame_logits),
            "--test-frame-logits",
            str(test_logits),
            "--metadata",
            str(metadata),
            "--output",
            str(tmp_path / "run"),
            "--oof-only",
        ],
    )

    with pytest.raises(SystemExit):
        temporal_suite.main()

    assert "--test-frame-logits must be omitted with --oof-only" in capsys.readouterr().err
