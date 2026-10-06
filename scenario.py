"""
Scenario generation for the Buffer-Aware Lossless Spot Instance Controller (BALSIC).

Generates synthetic, mathematically guaranteed arrival traces (Doc 08 §3) 
and eviction schedules, allowing the controller to be tested against both 
"nice" and strictly adversarial conditions.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from bucket import BucketMeter


def make_conforming_trace(
    seed: int, 
    n_ticks: int, 
    buckets: List[Tuple[float, float]],
    mode: str = "greedy"
) -> np.ndarray:
    """
    Generate an arrival sequence guaranteed to conform to ALL provided buckets.

    This function internally uses BucketMeters to police the generated traffic,
    ensuring the foundational assumption of the leaky-bucket model is never broken.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_ticks : int
        Length of the trace in ticks.
    buckets : List[Tuple[float, float]]
        List of (sigma, rho) pairs defining the arrival envelope.
    mode : str, default "greedy"
        - 'greedy': Always sends the absolute maximum allowed by the tightest bucket 
          (adversarial worst-case).
        - 'onoff': Alternates between greedy bursts and total silence.
        - 'smooth': Smoothly varying traffic capped safely below the limit.

    Returns
    -------
    np.ndarray
        Array of length n_ticks containing arrival volumes.
    """
    if n_ticks <= 0:
        raise ValueError(f"n_ticks must be > 0, got {n_ticks}")
    if not buckets:
        raise ValueError("Must provide at least one bucket (sigma, rho).")
    
    rng = np.random.default_rng(seed)
    arrivals = np.zeros(n_ticks, dtype=float)
    meters = [BucketMeter(sigma, rho) for sigma, rho in buckets]

    is_on = True
    toggle_countdown = rng.integers(10, 50)

    for t in range(n_ticks):
        # Calculate the absolute maximum we can send this tick without violating ANY bucket.
        # From: b(t-1) + a(t) - rho <= sigma  =>  a(t) <= sigma + rho - b(t-1)
        max_allowed = min(m.sigma + m.rho - m.state() for m in meters)
        
        if mode == "greedy":
            a_t = max_allowed
        
        elif mode == "onoff":
            toggle_countdown -= 1
            if toggle_countdown <= 0:
                is_on = not is_on
                toggle_countdown = rng.integers(10, 50)
            a_t = max_allowed if is_on else 0.0
            
        elif mode == "smooth":
            # Aim for 80% of the tightest sustained rate, add noise, clamp to safe limit.
            base_rate = min(rho for _, rho in buckets) * 0.8
            noise = rng.normal(0, base_rate * 0.2)
            a_t = np.clip(base_rate + noise, 0.0, max_allowed)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        arrivals[t] = a_t
        
        # Advance meters
        for m in meters:
            m.update(a_t)

    return arrivals


def make_violating_trace(
    seed: int, 
    n_ticks: int, 
    buckets: List[Tuple[float, float]], 
    violation_ticks: List[int],
    violation_size: float
) -> np.ndarray:
    """
    Deliberately generate a trace that violates the bucket envelope.
    
    Used to demonstrate data loss when foundational assumptions are broken.

    Parameters
    ----------
    violation_ticks : List[int]
        Specific ticks where the bounds should be breached.
    violation_size : float
        The extra volume to inject *above* the maximum allowed boundary.
    """
    if violation_size <= 0:
        raise ValueError("violation_size must be positive to force a violation.")
        
    arrivals = np.zeros(n_ticks, dtype=float)
    meters = [BucketMeter(sigma, rho) for sigma, rho in buckets]
    
    for t in range(n_ticks):
        max_allowed = min(m.sigma + m.rho - m.state() for m in meters)
        
        if t in violation_ticks:
            # Force the violation
            a_t = max_allowed + violation_size
        else:
            # Just send the sustained rate (or max allowed if sustained is too high)
            base_rho = min(rho for _, rho in buckets)
            a_t = min(base_rho, max_allowed)
            
        arrivals[t] = a_t
        for m in meters:
            m.update(a_t)
            
    return arrivals


def schedule_evictions(
    seed: int, 
    n_ticks: int, 
    mode: str, 
    hazard: float = 1/300,
    notice_ticks: Optional[int] = None,
    delta: Optional[int] = None
) -> Dict[str, Optional[np.ndarray]]:
    """
    Generate an eviction schedule and optional advance notices.

    Parameters
    ----------
    mode : str
        - 'random': Bernoulli trials with probability = hazard.
        - 'worst_case': Hand-picked strikes (e.g., middle of the simulation).
        - 'stale': Strikes exactly 1 tick after a control decision to maximize 
          the (delta - 1) blind spot (Doc 08 §5.2). Requires `delta`.
    notice_ticks : int, optional
        If provided, schedules a notice exactly this many ticks before the eviction.
    """
    rng = np.random.default_rng(seed)
    evictions = []

    if mode == "random":
        evictions = np.where(rng.random(n_ticks) < hazard)[0].tolist()
        
    elif mode == "worst_case":
        # Strike right in the middle
        evictions = [n_ticks // 2]
        
    elif mode == "stale":
        if delta is None or delta < 1:
            raise ValueError("Must provide delta >= 1 for 'stale' mode.")
        # If control happens at t=0, 5, 10... maximum staleness is at t=1, 6, 11...
        evictions = [t + 1 for t in range(0, n_ticks, delta * 10) if t + 1 < n_ticks][1:]
        
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    evictions_arr = np.array(evictions, dtype=int)
    
    notices_arr = None
    if notice_ticks is not None and notice_ticks > 0:
        notices_arr = np.maximum(0, evictions_arr - notice_ticks)

    return {
        "evictions": evictions_arr,
        "notices": notices_arr
    }