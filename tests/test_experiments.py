from yolo_r2plus1d.experiments.audit import audit_candidate


def test_sched30_consensus_contract() -> None:
    result = audit_candidate("sched30_consensus")
    assert result["ok"]
    assert result["oof_delta"] > 0.004
    assert result["combined_bytes"] <= 100_000_000


def test_temporal_pool_consensus_contract() -> None:
    result = audit_candidate("temporal_pool_consensus")
    assert result["ok"]
    assert result["oof_delta"] > 0.003
    assert result["combined_bytes"] <= 100_000_000
