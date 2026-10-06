"""
Phase 7b: Main Results Table Generation.

Evaluates the EnvelopeController against all baselines over >=20 random seeds
and extracts Drop %, Cost %, and Backlog metrics to a CSV.
"""

import sys
import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

from scenario import make_conforming_trace, schedule_evictions
from simulator import SimParams, run_simulation
from controller import EnvelopeController
from baselines import policy_all_on_demand, make_static, ReactiveBaseline, NoticeAwareBaseline

def wrap_policy_for_simulator(policy_obj):
    """Helper to route the simulator's on_eviction hook to baseline classes."""
    def wrapped(tick, x, b, notice_active):
        return policy_obj(tick, x, b, notice_active)
    if hasattr(policy_obj, 'on_eviction'):
        wrapped.on_eviction = policy_obj.on_eviction
    return wrapped

def main():
    print("Running baseline comparisons...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    seeds = range(30)
    results = []
    
    regimes = [
        {"name": "Short T_mig", "t_mig": 10.0, "notice": 30},
        {"name": "Long T_mig", "t_mig": 60.0, "notice": 30}
    ]
    
    for regime in regimes:
        for seed in seeds:
            params = SimParams(
                mu_od=100.0, mu_sp=75.0, B=2000.0, t_mig=regime["t_mig"], 
                delta=5, buckets=[(200.0, 80.0)], f=1.0
            )
            
            n_ticks = 1000
            arrivals = make_conforming_trace(seed=seed, n_ticks=n_ticks, buckets=params.buckets, mode="smooth")
            sched = schedule_evictions(seed=seed, n_ticks=n_ticks, mode="random", hazard=0.01, notice_ticks=regime["notice"])
            
            # FIXED: Explicitly pass only the parameters the controller expects
            ctrl = EnvelopeController(
                mu_od=params.mu_od, mu_sp=params.mu_sp, B=params.B, 
                t_mig=params.t_mig, delta=params.delta, buckets=params.buckets, 
                f=params.f, t_up=params.t_up
            )
            res_env = run_simulation(ctrl.alpha_star, arrivals, sched["evictions"], sched["notices"], params)
            
            res_od = run_simulation(policy_all_on_demand, arrivals, sched["evictions"], sched["notices"], params)
            
            static_pol = make_static(0.5)
            res_stat = run_simulation(static_pol, arrivals, sched["evictions"], sched["notices"], params)
            
            react_pol = ReactiveBaseline(0.8, 100)
            res_react = run_simulation(wrap_policy_for_simulator(react_pol), arrivals, sched["evictions"], sched["notices"], params)
            
            notice_pol = NoticeAwareBaseline(0.8, 100)
            res_notice = run_simulation(wrap_policy_for_simulator(notice_pol), arrivals, sched["evictions"], sched["notices"], params)
            
            policies = [
                ("EnvelopeController", res_env),
                ("All-On-Demand", res_od),
                ("Static-50", res_stat),
                ("Reactive-80", res_react),
                ("NoticeAware-80", res_notice)
            ]
            
            for pol_name, res in policies:
                results.append({
                    "Regime": regime["name"],
                    "Seed": seed,
                    "Policy": pol_name,
                    "Drop_Pct": res.drop_pct,
                    "Cost_Pct_OD": res.cost_pct_of_all_od,
                    "P99_Backlog": res.p99_backlog,
                    "Had_Loss": res.drop_pct > 0.0
                })
                
    df = pd.DataFrame(results)
    
    summary = df.groupby(["Regime", "Policy"]).agg({
        "Drop_Pct": "mean",
        "Cost_Pct_OD": "mean",
        "P99_Backlog": "mean",
        "Had_Loss": "sum"
    }).reset_index()
    
    out_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Comparison complete. Saved to {out_path}")
    print(summary)

if __name__ == "__main__":
    main()