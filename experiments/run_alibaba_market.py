"""
Phase 7e: Market Stress Extension (Phi) Evaluation.

Evaluates the EnvelopeController against an Alibaba cluster trace.
Compares the base controller (Phi=0) against the Market-Aware controller (Phi>0)
and the standard baselines to prove how predictive price tracking prevents evictions.
"""

import sys
import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

from scenario import make_conforming_trace
from simulator import SimParams, run_simulation
from controller import EnvelopeController
from baselines import policy_all_on_demand, make_static, ReactiveBaseline, NoticeAwareBaseline

def wrap_policy_for_simulator(policy_obj):
    def wrapped(tick, x, b, notice_active):
        return policy_obj(tick, x, b, notice_active)
    if hasattr(policy_obj, 'on_eviction'):
        wrapped.on_eviction = policy_obj.on_eviction
    return wrapped

def make_market_aware_policy(controller: EnvelopeController, prices: np.ndarray, k: int, m: float):
    """
    Creates a wrapper policy that dynamically calculates the price acceleration (Phi)
    at every tick, and feeds it into the Envelope Controller.
    """
    def policy(tick, x, b, notice_active):
        if tick >= 2 * k:
            # P(t) - 2P(t-k) + P(t-2k) / k^2
            p_t = prices[tick]
            p_tk = prices[tick - k]
            p_t2k = prices[tick - 2 * k]
            
            accel = max(0.0, (p_t - 2 * p_tk + p_t2k) / (k ** 2))
            phi = m * accel
        else:
            phi = 0.0
            
        return controller.alpha_star(x, b, phi=phi)
    
    return policy

def make_uncoupled_policy(controller: EnvelopeController):
    """A simple wrapper that ensures Phi=0 (the base law)."""
    def policy(tick, x, b, notice_active):
        return controller.alpha_star(x, b, phi=0.0)
    return policy

def main():
    print("Loading Alibaba Environment Trace...")
    csv_file = os.path.join(ROOT_DIR, "alibaba_env_ready.csv")
    
    if not os.path.exists(csv_file):
        print(f"ERROR: '{csv_file}' not found. Please run prepare_alibaba.py first.")
        return
        
    df = pd.read_csv(csv_file)
    n_ticks = len(df)
    
    # Define bucket and capacities for the simulation
    sigma = 500.0
    rho = 100.0
    mu_od = 120.0
    mu_sp = 100.0
    B = 2500.0
    
    params = SimParams(
        mu_od=mu_od, mu_sp=mu_sp, B=B, t_mig=2.0, 
        delta=1, buckets=[(sigma, rho)], f=1.0
    )
    
    # Generate smooth, conforming traffic for the duration of the trace
    arrivals = make_conforming_trace(seed=42, n_ticks=n_ticks, buckets=params.buckets, mode="smooth")
    
    # Generate dynamic evictions based on the EXACT Alibaba cluster stress!
    rng = np.random.default_rng(99)
    eviction_hazards = df['eps'].values
    evictions = np.where(rng.random(n_ticks) < eviction_hazards)[0]
    prices = df['c_sp'].values
    
    # Give a 1-tick advance notice for the NoticeAware baseline
    notices = np.maximum(0, evictions - 1)
    
    print(f"Generated {len(evictions)} dynamic evictions based on Alibaba cluster stress.")
    
    ctrl = EnvelopeController(
        mu_od=params.mu_od, mu_sp=params.mu_sp, B=params.B, 
        t_mig=params.t_mig, delta=params.delta, buckets=params.buckets, f=params.f
    )
    
    # --- PROPER CALIBRATION FOR PHI ---
    k = 3
    accels = []
    for t in range(2 * k, len(prices)):
        accels.append(max(0.0, (prices[t] - 2 * prices[t-k] + prices[t-2*k]) / (k ** 2)))
    
    # Filter out dead zones where price isn't moving
    active_accels = [a for a in accels if a > 1e-6]
    
    # We calibrate the trigger to the 75th percentile of active price jumps.
    # This ensures the controller is sensitive enough to dodge the vast majority of danger.
    trigger_accel = np.percentile(active_accels, 75) if active_accels else 1.0
    
    target_phi = ctrl.phi_critical(x=0.0, b=[0.0])
    
    # Calculate 'm' so that when the 75th percentile jump hits, Phi forces Spot to 0%
    m = (target_phi * 1.10) / trigger_accel if trigger_accel > 0 else 0.0
    
    policies = [
        ("All-On-Demand", policy_all_on_demand),
        ("Static-50", make_static(0.50)),
        ("Reactive-80", wrap_policy_for_simulator(ReactiveBaseline(0.80, 5))),
        ("NoticeAware-80", wrap_policy_for_simulator(NoticeAwareBaseline(0.80, 5))),
        ("Base Controller (Phi=0)", make_uncoupled_policy(ctrl)),
        ("Market-Aware Controller (Phi>0)", make_market_aware_policy(ctrl, prices, k, m))
    ]
    
    results = []
    
    print("Running simulations against volatile cloud market...")
    for name, pol in policies:
        res = run_simulation(pol, arrivals, evictions, notices, params)
        results.append({
            "Policy": name,
            "Drop_Pct": round(res.drop_pct, 2),
            "Cost_Pct_OD": round(res.cost_pct_of_all_od, 2),
            "P99_Backlog": round(res.p99_backlog, 2),
            "Hard_Evictions": res.eviction_count
        })
        
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "alibaba_market_stress_comparison.csv")
    res_df.to_csv(out_csv, index=False)
    
    print("\n=== Alibaba Market Stress & Baseline Evaluation ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()