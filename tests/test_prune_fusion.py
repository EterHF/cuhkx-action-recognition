import copy

import pytest

from yolo_r2plus1d.strict_v3.release.prune_fusion import temporal_visual_package


def package() -> dict:
    return {
        "fusion_4bit": {"weights": "fusion"},
        "visual_member0": {"weights": "visual"},
        "dstformer": {"weights": "temporal"},
        "thermal_mobilenet": {"weights": "dead"},
        "blend": {
            "fusion_weight": 0.11,
            "visual_weight": 0.22,
            "temporal_weight": 0.67,
            "presence": {"fusion": "fixed"},
        },
        "release_contract": {
            "weights": {"fusion": 0.11, "visual": 0.22, "temporal": 0.67},
            "temperatures": {"fusion": 1.1, "visual": 1.5, "temporal": 1.2},
            "presence": {"fusion": "fixed"},
            "thermal_weight": 0.0,
            "visual_package_output_scale": 0.5,
        },
    }


def test_pruning_removes_weights_without_changing_effective_tv_contract() -> None:
    source = package()
    original = copy.deepcopy(source)
    result = temporal_visual_package(source)

    assert source == original
    assert "fusion_4bit" not in result
    assert "thermal_mobilenet" not in result
    assert result["release_contract"]["weights"] == {
        "fusion": 0.0,
        "visual": 0.22,
        "temporal": 0.67,
    }
    assert result["release_contract"]["temperatures"] == original[
        "release_contract"
    ]["temperatures"]
    assert result["release_contract"]["visual_package_output_scale"] == 0.5


def test_pruning_rejects_active_thermal_or_already_inactive_fusion() -> None:
    active_thermal = package()
    active_thermal["release_contract"]["thermal_weight"] = 0.1
    with pytest.raises(RuntimeError, match="active thermal"):
        temporal_visual_package(active_thermal)

    inactive_fusion = package()
    inactive_fusion["release_contract"]["weights"]["fusion"] = 0.0
    with pytest.raises(RuntimeError, match="already inactive"):
        temporal_visual_package(inactive_fusion)
