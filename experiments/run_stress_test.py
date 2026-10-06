"""
Phase 7a: Adversarial Proof-Verification Sweep.

Iterates over a strict parameter grid of adversarial scenarios (greedy traces 
and worst-case/stale eviction timings). Will raise an AssertionError and fail loudly 
if a single data drop occurs, verifying the mathematical envelope law[cite: 6, 8].
"""

import sys
import os
import itertools

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from scenario import make_conforming_trace, schedule_evictions
from simulator import SimParams, run_simulation
from controller import EnvelopeController

def main():
    print("Starting BALSIC Adversarial Stress Test...")
    
    # Parameter Grid
    t_migs = [10.0, 30.0, 60.0]
    deltas = [1, 5, 10]
    fs = [0.5, 1.0]
    Bs = [1500.0, 5000.0]
    t_ups = [0.0, 2.0]
    
    # Bucket setups (Single vs Dual)
    bucket_setups = [
        [(200.0, 80.0)],                            # Single bucket
        [(200.0, 80.0), (500.0, 60.0)]              # Dual bucket[cite: 8]
    ]
    
    total_runs = len(t_migs) * len(deltas) * len(fs) * len(Bs) * len(t_ups) * len(bucket_setups)
    print(f"Total grid combinations to test: {total_runs}")
    
    passed = 0
    n_ticks = 500
    
    for t_mig, delta, f, B, t_up, buckets in itertools.product(t_migs, deltas, fs, Bs, t_ups, bucket_setups):
        params = SimParams(
            mu_od=100.0, mu_sp=75.0, B=B, t_mig=t_mig, 
            delta=delta, buckets=buckets, f=f, t_up=t_up
        )
        
        ctrl = EnvelopeController(
            mu_od=params.mu_od, mu_sp=params.mu_sp, B=params.B, 
            t_mig=params.t_mig, delta=params.delta, buckets=params.buckets, 
            f=params.f, t_up=params.t_up
        )
        
        # Adversarial greedy trace[cite: 6]
        arrivals = make_conforming_trace(seed=42, n_ticks=n_ticks, buckets=buckets, mode="greedy")
        
        # Stale eviction (worst-case timing relative to delta)[cite: 6]
        sched = schedule_evictions(seed=42, n_ticks=n_ticks, mode="stale", delta=delta)
        
        res = run_simulation(
            policy=ctrl.alpha_star, 
            arrivals=arrivals, 
            evictions=sched["evictions"], 
            notices=sched["notices"], 
            params=params
        )
        
        # THE CORE ASSERTION[cite: 6]
        assert res.drop_total == 0.0, (
            f"STRESS TEST FAILED! Data loss detected.\n"
            f"Params: t_mig={t_mig}, delta={delta}, f={f}, B={B}, t_up={t_up}\n"
            f"Drops: {res.drop_total}"
        )
        passed += 1

    print(f"Stress test complete. {passed}/{total_runs} scenarios passed with ZERO drops.")

if __name__ == "__main__":
    main() 