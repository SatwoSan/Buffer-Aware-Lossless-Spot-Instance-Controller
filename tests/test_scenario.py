"""
Tests for scenario.py.

Verifies that the generated traffic and evictions adhere strictly to the leaky 
bucket math and the requested timing modes.
"""

import pytest
import numpy as np
from scenario import make_conforming_trace, make_violating_trace, schedule_evictions
from bucket import BucketMeter

class TestConformingTrace:
    @pytest.mark.parametrize("mode", ["greedy", "onoff", "smooth"])
    def test_trace_strictly_conforms(self, mode):
        """
        No matter the mode, a conforming trace must NEVER violate any bucket.
        We police the output array using standalone BucketMeters to prove it.
        """
        buckets = [(200.0, 50.0), (500.0, 30.0)]
        arrivals = make_conforming_trace(seed=42, n_ticks=1000, buckets=buckets, mode=mode)
        
        meters = [BucketMeter(sig, rho) for sig, rho in buckets]
        
        for a_t in arrivals:
            for m in meters:
                m.update(a_t)
                assert m.conforms() is True, f"Bucket {m.sigma, m.rho} violated in {mode} mode!"

class TestViolatingTrace:
    def test_trace_violates_at_exact_ticks(self):
        """
        Confirm make_violating_trace breaks the rules only when told to.
        """
        buckets = [(200.0, 50.0)]
        violation_ticks = [50, 80]
        arrivals = make_violating_trace(seed=42, n_ticks=100, buckets=buckets, 
                                        violation_ticks=violation_ticks, violation_size=10.0)
        
        meter = BucketMeter(200.0, 50.0)
        
        for t, a_t in enumerate(arrivals):
            meter.update(a_t)
            if t in violation_ticks:
                assert meter.conforms() is False, f"Expected violation missed at tick {t}"
            else:
                # Need to allow it to recover, but immediately before/after it might still fail 
                # depending on the leak rate. However, at t=49, it should definitely conform.
                if t < violation_ticks[0]:
                    assert meter.conforms() is True

class TestEvictions:
    def test_stale_mode_maximizes_blind_spot(self):
        """
        Confirm 'stale' mode places evictions exactly 1 tick after a control interval.
        """
        delta = 5
        sched = schedule_evictions(seed=1, n_ticks=100, mode="stale", delta=delta)
        evictions = sched["evictions"]
        
        assert len(evictions) > 0
        for ev in evictions:
            # If delta=5, control happens at 0, 5, 10...
            # Eviction should happen at 1, 6, 11... -> remainder is 1
            assert ev % delta == 1

    def test_notices_are_properly_offset(self):
        """
        Confirm notice_ticks properly offsets the notice array.
        """
        sched = schedule_evictions(seed=1, n_ticks=100, mode="worst_case", notice_ticks=30)
        evictions = sched["evictions"]
        notices = sched["notices"]
        
        assert evictions[0] == 50
        assert notices[0] == 20  # 50 - 30