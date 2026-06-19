from __future__ import annotations

import numpy as np

from task2_act.metrics import action_l1_metrics


def test_action_l1_metrics_with_mask() -> None:
    pred = np.zeros((2, 3, 7), dtype=np.float32)
    target = np.ones((2, 3, 7), dtype=np.float32)
    mask = np.array([[True, True, False], [True, False, False]])

    metrics = action_l1_metrics(pred, target, mask)

    assert metrics["overall_l1"] == 1.0
    assert metrics["num_valid_action_vectors"] == 3
    assert len(metrics["per_dim_l1"]) == 7
    assert len(metrics["per_horizon_l1"]) == 3
