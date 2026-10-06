"""
Tests for baselines.py.

Verifies state transitions, initialization parameters, and expected behaviors 
for the reference policies prior to full simulation integration.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines import policy_all_on_demand, make_static, ReactiveBaseline, NoticeAwareBaseline

def test_all_on_demand():
    """All-OD should always return 0.0 regardless of inputs."""
    assert policy_all_on_demand(0, 1000.0, [50.0], False) == 0.0
    assert policy_all_on_demand(10, 0.0, [0.0], True) == 0.0

def test_static_policy():
    """Static policy should consistently return its initialized value."""
    policy = make_static(0.65)
    assert policy(0, 500.0, [10.0], False) == 0.65
    assert policy(5, 1500.0, [200.0], True) == 0.65

def test_static_policy_validation():
    with pytest.raises(ValueError, match="must be in"):
        make_static(1.5)
    with pytest.raises(ValueError, match="must be in"):
        make_static(-0.1)

def test_reactive_baseline_state_transitions():
    """
    Reactive baseline should return alpha_hi until an eviction strikes, 
    then drop to 0.0 for the duration of the cooldown.
    """
    policy = ReactiveBaseline(alpha_hi=0.8, cooldown_ticks=3)
    
    # Normal operation
    assert policy(0, 0.0, [0.0], False) == 0.8
    assert policy(1, 0.0, [0.0], False) == 0.8
    
    # Eviction strikes
    policy.on_eviction()
    
    # Cooldown period (3 ticks)
    assert policy(2, 0.0, [0.0], False) == 0.0  # Remaining: 2
    assert policy(3, 0.0, [0.0], False) == 0.0  # Remaining: 1
    assert policy(4, 0.0, [0.0], False) == 0.0  # Remaining: 0
    
    # Recovered
    assert policy(5, 0.0, [0.0], False) == 0.8

def test_notice_aware_baseline_preemptive_reaction():
    """
    Notice-aware baseline should react immediately upon seeing `notice_active=True`,
    even before the actual eviction hook is called.
    """
    policy = NoticeAwareBaseline(alpha_hi=0.75, cooldown_ticks=5)
    
    # Normal operation
    assert policy(0, 0.0, [0.0], False) == 0.75
    
    # Advance notice arrives (no eviction hook called yet)
    assert policy(1, 0.0, [0.0], True) == 0.0
    
    # Even if notice drops, cooldown was triggered
    assert policy(2, 0.0, [0.0], False) == 0.0