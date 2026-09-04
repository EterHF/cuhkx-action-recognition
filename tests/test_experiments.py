from yolo_r2plus1d.experiments.audit import audit_candidate


def test_sched30_consensus_contract() -> None:
    result = audit_candidate("sched30_consensus")
    assert result["ok"]
    assert result["oof_delta"] > 0.004
    assert result["single_checkpoint_bytes"] < 100_000_000
    assert result["single_checkpoint_margin_bytes"] > 0
    assert result["kaggle"] == {
        "submitted": True,
        "submission_ref": 56016293,
        "submitted_at_utc": "2026-09-04T16:25:01.040000",
        "public_score": 0.97014,
    }


def test_temporal_pool_consensus_contract() -> None:
    result = audit_candidate("temporal_pool_consensus")
    assert result["ok"]
    assert result["oof_delta"] > 0.003
    assert result["single_checkpoint_bytes"] < 100_000_000
    assert result["single_checkpoint_margin_bytes"] > 0
    assert result["kaggle"] == {"submitted": False}
