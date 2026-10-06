"""
Phase 7c: Controller Ablation Studies.

Isolates and evaluates the impact of Bucket Count (1 vs 2) and control period 
Delta on cost efficiency and performance.
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

def main():
    print("Running Ablation Studies...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    
    B_val = 3000.0
    t_mig = 40.0
    n_ticks = 800
    seed = 123
    sched = schedule_evictions(seed=seed, n_ticks=n_ticks, mode="random", hazard=0.02)
    
    single_bucket = [(500.0, 70.0)]
    dual_bucket = [(200.0, 90.0), (800.0, 50.0)]
    
    for name, buckets in [("Single Bucket", single_bucket), ("Dual Bucket", dual_bucket)]:
        arrivals = make_conforming_trace(seed=seed, n_ticks=n_ticks, buckets=buckets, mode="onoff")
        
        params = SimParams(
            mu_od=100.0, mu_sp=75.0, B=B_val, t_mig=t_mig, delta=2, buckets=buckets
        )
        
        # FIXED: Explicit instantiation
        ctrl = EnvelopeController(
            mu_od=params.mu_od, mu_sp=params.mu_sp, B=params.B, 
            t_mig=params.t_mig, delta=params.delta, buckets=params.buckets, 
            f=params.f, t_up=params.t_up
        )
        res = run_simulation(ctrl.alpha_star, arrivals, sched["evictions"], None, params)
        
        results.append({
            "Experiment": "Bucket Count",
            "Variant": name,
            "Cost_Pct_OD": res.cost_pct_of_all_od,
            "Drop_Pct": res.drop_pct
        })

    deltas_to_test = [1, 5, 15, 30]
    arrivals = make_conforming_trace(seed=seed, n_ticks=n_ticks, buckets=single_bucket, mode="smooth")
    
    for d in deltas_to_test:
        params = SimParams(
            mu_od=100.0, mu_sp=75.0, B=B_val, t_mig=t_mig, delta=d, buckets=single_bucket
        )
        
        # FIXED: Explicit instantiation
        ctrl = EnvelopeController(
            mu_od=params.mu_od, mu_sp=params.mu_sp, B=params.B, 
            t_mig=params.t_mig, delta=params.delta, buckets=params.buckets, 
            f=params.f, t_up=params.t_up
        )
        res = run_simulation(ctrl.alpha_star, arrivals, sched["evictions"], None, params)
        
        results.append({
            "Experiment": "Delta Sensitivity",
            "Variant": f"Delta={d}",
            "Cost_Pct_OD": res.cost_pct_of_all_od,
            "Drop_Pct": res.drop_pct
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Ablation complete. Saved to {out_path}")
    print(df)

if __name__ == "__main__":
    main()