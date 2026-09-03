from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.verify import EXPECTED, digest


def test_published_release_hashes() -> None:
    roots = {"model.pt": CHECKPOINT_DIR, "yolo11n.pt": CHECKPOINT_DIR}
    for name, expected in EXPECTED.items():
        path = roots.get(name, RESULT_DIR) / name
        assert path.is_file()
        assert digest(path) == expected


def test_release_fits_track_budget() -> None:
    total = (CHECKPOINT_DIR / "model.pt").stat().st_size + (
        CHECKPOINT_DIR / "yolo11n.pt"
    ).stat().st_size
    assert total <= 100_000_000
