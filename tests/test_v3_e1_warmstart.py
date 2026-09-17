import pytest
from scripts.run_v3_e1_warmstart import phase_lr


def test_phase_schedule_restarts_and_remains_at_reduced_peak():
    assert phase_lr(0) == pytest.approx(5e-5/300)
    assert phase_lr(299) == pytest.approx(5e-5)
    assert phase_lr(300) == pytest.approx(5e-5)
    assert phase_lr(731) == pytest.approx(5e-5)
    assert max(phase_lr(i) for i in range(732)) <= 5e-5
