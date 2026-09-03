from pathlib import Path

import pandas as pd

from yolo_r2plus1d.strict_v3.data.indexing import discover_train, read_test_ids


def test_discover_train_uses_class_user_trial_order(tmp_path: Path) -> None:
    for relative in (
        "Depth_Color/1_Run/user2/2-1-1",
        "Depth_Color/0_Walk/user16/1-1-2",
        "Depth_Color/0_Walk/user16/1-1-1",
        "Skeleton/0_Walk/user16/1-1-1",
    ):
        (tmp_path / relative).mkdir(parents=True)
    samples = discover_train(tmp_path)
    assert [sample[:3] for sample in samples] == [
        ("0_Walk/user16/1-1-1", 0, 16),
        ("0_Walk/user16/1-1-2", 0, 16),
        ("1_Run/user2/2-1-1", 1, 2),
    ]


def test_read_test_ids_accepts_submission_paths(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"path": ["small_model_track_test/SM_test_0002/"]}).to_csv(
        csv_path, index=False
    )
    assert read_test_ids(csv_path) == ["SM_test_0002"]
