import numpy as np

from yolo_r2plus1d.strict_v3.evaluation.current_error_audit import true_class_rank


def test_true_class_rank_is_one_based_and_handles_ties() -> None:
    logits = np.asarray([[3.0, 2.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    labels = np.asarray([1, 1])

    assert np.array_equal(true_class_rank(logits, labels), np.asarray([2, 1]))
