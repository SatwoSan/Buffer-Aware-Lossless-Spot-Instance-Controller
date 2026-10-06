"""
Tick-by-tick fluid-buffer simulator for BALSIC.

This is the integration point where the traffic envelope, the control law, 
and the eviction mechanics meet. It is designed as a single, unified loop 
capable of running both the EnvelopeController and Phase 5 baseline policies.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any
import numpy as np
import inspect
from bucket import BucketMeter

@dataclass
class SimParams:
    mu_od: float
    mu_sp: float
    B: float
    t_mig: float
    delta: int
    buckets: List[Tuple[float, float]]
    f: float = 1.0
    t_up: float = 0.0
    price_od: float = 1.0
    price_sp: float = 0.3

    def __post_init__(self):
        if self.mu_od <= 0 or self.mu_sp <= 0 or self.B <= 0:
            raise ValueError("Capacities and buffer must be positive.")
        if self.delta < 1:
            raise ValueError("delta must be >= 1.")
        if not (0 < self.f <= 1):
            raise ValueError("f must be in (0, 1].")

@dataclass
class SimResult:
    drop_total: float
    drop_pct: float
    cost_total: float
    cost_pct_of_all_od: float
    backlog_trace: np.ndarray
    alpha_trace: np.ndarray
    eviction_count: int
    infeasible_count: int
    p99_backlog: float
    mean_backlog: float
    decision_log: List[Tuple[int, float, bool]]


def run_simulation(
    policy: Callable, 
    arrivals: np.ndarray, 
    evictions: np.ndarray,
    notices: Optional[np.ndarray], 
    params: SimParams
) -> SimResult:
    """
    Run the unified tick-by-tick simulation.

    Parameters
    ----------
    policy : Callable
        Signature: policy(tick, x, buckets_state, notice_active) -> float or Result.
        Can be the EnvelopeController or a simple baseline.
    arrivals : np.ndarray
        Array of arrivals per tick (from scenario.py).
    evictions : np.ndarray
        Ticks on which Spot evictions strike.
    notices : np.ndarray, optional
        Ticks on which advance interruption notices arrive.
    params : SimParams
        The physical system constants.

    Returns
    -------
    SimResult
        Comprehensive metrics for Phase 7 analysis.
    """
    n_ticks = len(arrivals)
    
    # State tracking
    x = 0.0
    meters = [BucketMeter(sig, rho) for sig, rho in params.buckets]
    alpha_current = 0.0
    alpha_ev = 0.0
    
    mig_countdown = 0
    notice_active = False
    
    # Actuation lag queue: list of (activation_tick, target_alpha)
    pending_alphas = []
    
    # Fast lookups
    evict_set = set(evictions) if evictions is not None else set()
    notice_set = set(notices) if notices is not None else set()

    # Check policy signature to seamlessly support baselines (4 params) and alpha_star (2 params)
    sig = inspect.signature(policy)
    policy_takes_full_args = len(sig.parameters) >= 4

    # Telemetry
    backlog_trace = np.zeros(n_ticks, dtype=float)
    alpha_trace = np.zeros(n_ticks, dtype=float)
    decision_log = []
    
    drop_total = 0.0
    total_processed = 0.0
    cost_total = 0.0
    eviction_count = 0
    infeasible_count = 0

    for t in range(n_ticks):
        arr = arrivals[t]

        # 1. Check for notices
        if t in notice_set:
            notice_active = True

        # 2. Control Decision (only outside of an active migration recovery)
        if t % params.delta == 0 and mig_countdown <= 0:
            b_states = [m.state() for m in meters]
            
            # API Adaptation: Route parameters based on policy signature
            if policy_takes_full_args:
                res = policy(t, x, b_states, notice_active)
            else:
                res = policy(x, b_states)
            
            # Unify API: accept both EnvelopeController Result objects and raw floats (baselines)
            if hasattr(res, 'alpha'):
                new_alpha = float(res.alpha)
                is_feasible = bool(getattr(res, 'feasible', True))
            else:
                new_alpha = float(res)
                is_feasible = True
                
            if not is_feasible:
                infeasible_count += 1
                
            decision_log.append((t, new_alpha, is_feasible))
            # Enqueue the decision to account for actuation lag (t_up)
            pending_alphas.append((t + int(params.t_up), new_alpha))

        # Apply any decisions that have finished their actuation lag
        while pending_alphas and pending_alphas[0][0] <= t:
            alpha_current = pending_alphas.pop(0)[1]

        # 3. Evictions
        # We only start a new migration if we aren't already in one, and if we actually have Spot.
        if t in evict_set and alpha_current > 0.0 and mig_countdown <= 0:
            alpha_ev = alpha_current
            mig_countdown = int(params.t_mig)
            eviction_count += 1
            # The notice is consumed by the actual eviction
            notice_active = False 

        # 4. Processing Capacity
        if mig_countdown > 0:
            # During migration: OD capacity + Surviving Spot capacity (1 - f)
            capacity = (1.0 - alpha_ev) * params.mu_od + alpha_ev * (1.0 - params.f) * params.mu_sp
            mig_countdown -= 1
        else:
            # Normal operation
            capacity = (1.0 - alpha_current) * params.mu_od + alpha_current * params.mu_sp

        # 5. Advance meters
        # The meters update AFTER the policy reads them, representing the arrival of a(t)
        for m in meters:
            m.update(arr)

        # 6. Buffer Physics
        avail = x + arr
        processed = min(capacity, avail)
        new_x = avail - processed

        dropped = 0.0
        if new_x > params.B:
            dropped = new_x - params.B
            new_x = params.B

        x = new_x
        
        # 7. Accumulate metrics
        drop_total += dropped
        total_processed += processed
        cost_total += ((1.0 - alpha_current) * params.price_od) + (alpha_current * params.price_sp)
        
        backlog_trace[t] = x
        alpha_trace[t] = alpha_current

    # Final Summary calculations
    total_arrived = np.sum(arrivals)
    drop_pct = (drop_total / total_arrived) * 100.0 if total_arrived > 0 else 0.0
    all_od_cost = n_ticks * params.price_od
    cost_pct = (cost_total / all_od_cost) * 100.0 if all_od_cost > 0 else 0.0

    return SimResult(
        drop_total=drop_total,
        drop_pct=drop_pct,
        cost_total=cost_total,
        cost_pct_of_all_od=cost_pct,
        backlog_trace=backlog_trace,
        alpha_trace=alpha_trace,
        eviction_count=eviction_count,
        infeasible_count=infeasible_count,
        p99_backlog=float(np.percentile(backlog_trace, 99)),
        mean_backlog=float(np.mean(backlog_trace)),
        decision_log=decision_log
    )