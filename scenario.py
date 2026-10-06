"""
Scenario generation for the Buffer-Aware Lossless Spot Instance Controller (BALSIC).

Generates synthetic, mathematically guaranteed arrival traces and 
eviction schedules, allowing the controller to be tested against both 
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
            base_rate = min(rho for _, rho in buckets) * 0.8
            noise = rng.normal(0, base_rate * 0.2)
            a_t = np.clip(base_rate + noise, 0.0, max_allowed)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        arrivals[t] = a_t
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
    """
    if violation_size <= 0:
        raise ValueError("violation_size must be positive to force a violation.")
        
    arrivals = np.zeros(n_ticks, dtype=float)
    meters = [BucketMeter(sigma, rho) for sigma, rho in buckets]
    
    for t in range(n_ticks):
        max_allowed = min(m.sigma + m.rho - m.state() for m in meters)
        
        if t in violation_ticks:
            a_t = max_allowed + violation_size
        else:
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
        
        # Place a single eviction around the middle of the simulation, 
        # exactly 1 tick after a control decision to maximize staleness[cite: 6].
        mid_point = n_ticks // 2
        control_tick = (mid_point // delta) * delta
        stale_tick = control_tick + 1
        
        if stale_tick < n_ticks:
            evictions = [stale_tick]
        else:
            evictions = [1]
        
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