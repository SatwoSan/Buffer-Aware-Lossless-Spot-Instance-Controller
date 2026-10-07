"""
experiments/run_azure_experiment.py — Evaluate BALSIC and baselines against 
real-world Azure Functions invocation traces.
"""

import sys
import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

from azure_parser import extract_azure_trace_and_bucket
from simulator import SimParams, run_simulation
from controller import EnvelopeController
from baselines import policy_all_on_demand, make_static, ReactiveBaseline, NoticeAwareBaseline
from scenario import schedule_evictions

def wrap_policy_for_simulator(policy_obj):
    def wrapped(tick, x, b, notice_active):
        return policy_obj(tick, x, b, notice_active)
    if hasattr(policy_obj, 'on_eviction'):
        wrapped.on_eviction = policy_obj.on_eviction
    return wrapped

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Update this path to where your extracted CSV lives!
    csv_file = "invocations_per_function_md.anon.d01.csv"
    
    if not os.path.exists(csv_file):
        # Look in current directory or prompt user
        print(f"File '{csv_file}' not found in root. Checking subdirectories...")
        found = False
        for root, _, files in os.walk(ROOT_DIR):
            if csv_file in files:
                csv_file = os.path.join(root, csv_file)
                found = True
                break
        if not found:
            print(f"ERROR: Could not find '{csv_file}'. Please place it in your root directory.")
            return

    # 2. Extract real trace and its conforming bucket
    trace, sigma, rho = extract_azure_trace_and_bucket(csv_file)
    n_ticks = len(trace)  # 1440 minutes = 24 hours

    # 3. Size capacities to match the real workload's scale:
    # 1 tick = 1 minute.
    # On-demand capacity mu_od is provisioned to comfortably drain the sustained rate rho.
    mu_od = float(np.ceil(rho * 1.25))
    mu_sp = float(np.ceil(rho * 1.10))
    B = float(np.ceil(sigma * 2.0))      # Buffer sized to hold burst headroom
    t_mig = 2.0                          # 2 minutes to boot an EC2 replacement
    delta = 1                            # Re-evaluate every 1 minute

    params = SimParams(
        mu_od=mu_od,
        mu_sp=mu_sp,
        B=B,
        t_mig=t_mig,
        delta=delta,
        buckets=[(sigma, rho)],
        f=1.0,
        t_up=0.0
    )

    print("\nSimulation Sizing Parameters (1 tick = 1 minute):")
    print(f"  Buffer Capacity (B) : {B:.1f}")
    print(f"  On-Demand Fleet (mu): {mu_od:.1f} req/min")
    print(f"  Spot Fleet Capacity : {mu_sp:.1f} req/min")
    print(f"  Migration Time      : {t_mig:.1f} min\n")

    # 4. Schedule realistic eviction events across the 24 hours
    # Average 1 eviction every 300 minutes (~5 evictions a day), with 2-minute notice
    sched = schedule_evictions(seed=42, n_ticks=n_ticks, mode="random", hazard=1/300, notice_ticks=2)

    # 5. Initialize Controller and Baselines
    ctrl = EnvelopeController(
        mu_od=params.mu_od,
        mu_sp=params.mu_sp,
        B=params.B,
        t_mig=params.t_mig,
        delta=params.delta,
        buckets=params.buckets,
        f=params.f,
        t_up=params.t_up
    )

    policies = [
        ("All-On-Demand", policy_all_on_demand),
        ("EnvelopeController", ctrl.alpha_star),
        ("Static-50", make_static(0.50)),
        ("Reactive-80", wrap_policy_for_simulator(ReactiveBaseline(alpha_hi=0.80, cooldown_ticks=10))),
        ("NoticeAware-80", wrap_policy_for_simulator(NoticeAwareBaseline(alpha_hi=0.80, cooldown_ticks=10)))
    ]

    results = []
    print("Running simulations on real Azure trace...")
    for name, pol in policies:
        res = run_simulation(pol, trace, sched["evictions"], sched["notices"], params)
        results.append({
            "Policy": name,
            "Drop_Total_Reqs": res.drop_total,
            "Drop_Pct": res.drop_pct,
            "Cost_Pct_OD": res.cost_pct_of_all_od,
            "P99_Backlog": res.p99_backlog,
            "Evictions": res.eviction_count
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "azure_trace_comparison.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Real Azure Functions Evaluation Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()