from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.bundle import (
    DEFAULT_BUNDLE,
    MODEL_LIMIT_BYTES,
    load_bundle,
)
from yolo_r2plus1d.strict_v3.release.verify import EXPECTED, EXPECTED_BUNDLE, digest


def test_published_release_hashes() -> None:
    roots = {"model.pt": CHECKPOINT_DIR, "yolo11n.pt": CHECKPOINT_DIR}
    for name, expected in EXPECTED.items():
        path = roots.get(name, RESULT_DIR) / name
        assert path.is_file()
        assert digest(path) == expected


def test_release_fits_track_budget() -> None:
    assert DEFAULT_BUNDLE.stat().st_size < MODEL_LIMIT_BYTES
    assert digest(DEFAULT_BUNDLE) == EXPECTED_BUNDLE
    bundle = load_bundle(DEFAULT_BUNDLE)
    assert bundle["metadata"]["model_sha256"] == EXPECTED["model.pt"]
    assert bundle["metadata"]["detector_sha256"] == EXPECTED["yolo11n.pt"]
