"""
Tests for bucket.py — the stateful leaky-bucket meter.

Test structure follows Doc 09 Phase 2 test requirements:
  1. Reproduce the exact worked example from Doc 08 §3.3.
  2. Confirm steady stream behavior at exactly rho.
  3. Confirm boundary behavior with a single burst of exactly sigma.
"""

import pytest
import sys
import os

# Add the project root to the path so we can import the bucket module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bucket import BucketMeter

class TestBucketMeterHandComputed:
    """Verify the exact worked example from the BALSIC documentation."""

    def test_doc08_worked_example(self):
        """
        Reproduce the worked example from Doc 08 §3.3.
        
        Parameters:
            rho = 50, sigma = 200
            Arrivals = [50, 50, 300, 50]
            
        Expected b(t) states:
            Tick 0: 0 (initial)
            Tick 1: max(0, 0 + 50 - 50) = 0
            Tick 2: max(0, 0 + 50 - 50) = 0
            Tick 3: max(0, 0 + 300 - 50) = 250 -> Violated (250 > 200)
            Tick 4: max(0, 250 + 50 - 50) = 250 -> Still violated
        """
        meter = BucketMeter(sigma=200.0, rho=50.0)
        assert meter.state() == 0.0

        # Tick 1: arrival = 50
        b = meter.update(50.0)
        assert b == 0.0
        assert meter.conforms() is True

        # Tick 2: arrival = 50
        b = meter.update(50.0)
        assert b == 0.0
        assert meter.conforms() is True

        # Tick 3: arrival = 300 (Massive burst)
        b = meter.update(300.0)
        assert b == 250.0
        assert meter.conforms() is False

        # Tick 4: arrival = 50
        b = meter.update(50.0)
        assert b == 250.0
        assert meter.conforms() is False


class TestBucketMeterSteadyState:
    """Verify behavior under sustained, non-bursty loads."""

    def test_steady_stream_at_rho_never_grows(self):
        """
        Confirm a steady stream at exactly rho never grows b past its starting value.
        If arrival == rho, then (arrival - rho) == 0, so b(t) = max(0, b(t-1) + 0).
        """
        meter = BucketMeter(sigma=100.0, rho=50.0)
        
        # Stream at exactly rho for 100 ticks
        for _ in range(100):
            b = meter.update(50.0)
            assert b == 0.0
            assert meter.conforms() is True

    def test_steady_stream_maintains_existing_backlog(self):
        """
        If the bucket already has some burst tracked, a steady stream at exactly 
        rho should keep the tracker completely flat, neither draining nor growing.
        """
        meter = BucketMeter(sigma=100.0, rho=50.0)
        
        # Inject a small initial burst
        meter.update(80.0)
        assert meter.state() == 30.0  # max(0, 0 + 80 - 50)
        
        # Stream at exactly rho should hold the state at 30.0
        for _ in range(10):
            b = meter.update(50.0)
            assert b == 30.0
            assert meter.conforms() is True


class TestBucketMeterBoundary:
    """Verify behavior exactly at the sigma threshold."""

    def test_burst_exactly_at_sigma_boundary(self):
        """
        Confirm a single burst of exactly sigma (from b=0) leaves conforms()==True 
        at the boundary.
        
        To push b to exactly sigma in one tick, the arrival must be (sigma + rho).
        """
        sigma = 200.0
        rho = 50.0
        meter = BucketMeter(sigma=sigma, rho=rho)
        
        # Arrival of 250 -> b = max(0, 0 + 250 - 50) = 200
        exact_boundary_arrival = sigma + rho
        b = meter.update(exact_boundary_arrival)
        
        assert b == sigma
        assert meter.conforms() is True

    def test_burst_slightly_over_sigma_fails(self):
        """
        Confirm that an arrival of sigma + epsilon makes the bucket fail.
        """
        sigma = 200.0
        rho = 50.0
        meter = BucketMeter(sigma=sigma, rho=rho)
        
        # Arrival of 250.001 -> b = 200.001
        failing_arrival = sigma + rho + 0.001
        b = meter.update(failing_arrival)
        
        assert b > sigma
        assert meter.conforms() is False


class TestBucketMeterDraining:
    """Verify the natural decay of the burst tracker."""

    def test_bucket_drains_during_silence(self):
        """
        If arrivals < rho, the bucket should drain by (rho - arrival) each tick, 
        and flatten perfectly at zero without going negative.
        """
        meter = BucketMeter(sigma=200.0, rho=50.0)
        
        # Spike it to 100
        meter.update(150.0)
        assert meter.state() == 100.0
        
        # Tick with no arrivals (drains by 50)
        meter.update(0.0)
        assert meter.state() == 50.0
        
        # Tick with no arrivals (drains by 50)
        meter.update(0.0)
        assert meter.state() == 0.0
        
        # Tick with no arrivals (should stay exactly 0, not -50)
        meter.update(0.0)
        assert meter.state() == 0.0


class TestBucketMeterValidation:
    """Verify input validation stops invalid states."""

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            BucketMeter(sigma=-10.0, rho=50.0)
            
    def test_zero_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            BucketMeter(sigma=200.0, rho=0.0)

    def test_negative_arrival_raises(self):
        meter = BucketMeter(sigma=200.0, rho=50.0)
        with pytest.raises(ValueError, match="Arrivals cannot be negative"):
            meter.update(-10.0)