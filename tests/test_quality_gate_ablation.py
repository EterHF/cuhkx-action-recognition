import numpy as np

from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    exact_mcnemar_p,
    fixed_quality_logits,
)


def test_fixed_quality_preserves_base_weight_ratio() -> None:
    visual = np.asarray([[2.0, 0.0]], dtype=np.float32)
    temporal = np.asarray([[0.0, 4.0]], dtype=np.float32)
    output = fixed_quality_logits(
        visual,
        temporal,
        {"visual": 2.0, "temporal": 4.0},
        {"visual": 1.0, "temporal": 3.0},
    )
    assert np.allclose(output, [[0.25, 0.75]])


def test_exact_mcnemar_p_handles_equal_and_one_sided_changes() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == 1.0
    assert exact_mcnemar_p(5, 0) == 0.0625
