"""
Phase 7d: Figure Generation.

Reads the output CSVs from Phase 7b and 7c and plots them for the final write-up[cite: 6].
Requires matplotlib.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

def main():
    print("Generating figures...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Comparison Bar Chart (Cost vs Loss)
    comp_file = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    if os.path.exists(comp_file):
        df = pd.read_csv(comp_file)
        
        # Filter for Long T_mig to show clear baseline failures[cite: 8]
        df_long = df[df["Regime"] == "Long T_mig"]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Policy')
        ax1.set_ylabel('Cost % of All-OD', color=color)
        ax1.bar(df_long["Policy"], df_long["Cost_Pct_OD"], color=color, alpha=0.6, label="Cost")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Scenarios with Data Loss (Count)', color=color)
        ax2.plot(df_long["Policy"], df_long["Had_Loss"], color=color, marker='o', linewidth=2, label="Loss Count")
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Cost vs Reliability (Long T_mig)')
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, "cost_vs_loss.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Generated {out_path}")
    else:
        print(f"Warning: {comp_file} not found. Run run_comparison.py first.")
        
    # 2. Ablation Chart (Delta Sensitivity)
    ab_file = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    if os.path.exists(ab_file):
        df = pd.read_csv(ab_file)
        df_delta = df[df["Experiment"] == "Delta Sensitivity"]
        
        plt.figure(figsize=(8, 5))
        plt.plot(df_delta["Variant"], df_delta["Cost_Pct_OD"], marker='s', color='green', linewidth=2)
        plt.title('Impact of Control Staleness ($\Delta$) on Cost')
        plt.xlabel('Control Period ($\Delta$)')
        plt.ylabel('Cost % of All-OD')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, "delta_sensitivity.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Generated {out_path}")
    else:
         print(f"Warning: {ab_file} not found. Run run_ablation.py first.")

if __name__ == "__main__":
    main()