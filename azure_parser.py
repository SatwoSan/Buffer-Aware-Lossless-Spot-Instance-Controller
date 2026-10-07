"""
azure_parser.py — Extract real-world bursty traces from the Azure Functions 2019 dataset
and empirically characterize their leaky-bucket parameters (sigma, rho).
"""

import pandas as pd
import numpy as np
from typing import Tuple

def extract_azure_trace_and_bucket(csv_path: str) -> Tuple[np.ndarray, float, float]:
    """
    Extracts an active, bursty 24-hour trace from an Azure Functions invocations CSV
    and computes the tightest conforming (sigma, rho) envelope for it.

    Parameters
    ----------
    csv_path : str
        Path to an invocations_per_function_md.anon.d[01-14].csv file.

    Returns
    -------
    trace : np.ndarray
        Array of length 1440 (1 tick = 1 minute).
    sigma : float
        Empirical burst allowance.
    rho : float
        Empirical sustained rate.
    """
    print(f"Loading Azure invocations from: {csv_path} ...")
    # Read CSV (first 4 columns are HashOwner, HashApp, HashFunction, Trigger)
    df = pd.read_csv(csv_path)

    # Invocations per minute are in columns 4 through 1443 (1440 minutes in a day)
    ts_matrix = df.iloc[:, 4:].fillna(0.0).values.astype(float)

    total_traffic = ts_matrix.sum(axis=1)
    max_spikes = ts_matrix.max(axis=1)

    # Filter for active functions (at least 20,000 requests over the 24h day)
    active_indices = np.where(total_traffic > 20000)[0]
    if len(active_indices) == 0:
        raise ValueError("No functions found with > 20,000 daily invocations.")

    # Calculate burst ratio: Peak Minute / Average Minute
    avg_per_minute = total_traffic[active_indices] / 1440.0
    burst_ratios = max_spikes[active_indices] / np.maximum(avg_per_minute, 1.0)

    # Pick the function with the highest burst ratio among active workloads
    best_relative_idx = np.argmax(burst_ratios)
    chosen_row_idx = active_indices[best_relative_idx]
    trace = ts_matrix[chosen_row_idx]

    mean_rate = float(np.mean(trace))
    peak_rate = float(np.max(trace))

    # Empirical bucket fitting:
    # Set rho with a small 10% headroom above mean to ensure long-term stability
    rho = float(np.ceil(mean_rate * 1.10))

    # Calculate exact sigma required so the trace conforns: b(t) <= sigma for all t
    b = 0.0
    max_b = 0.0
    for a_t in trace:
        b = max(0.0, b + a_t - rho)
        if b > max_b:
            max_b = b

    sigma = float(np.ceil(max_b * 1.05))  # 5% buffer on peak burst

    print(f"Selected Function Row #{chosen_row_idx}:")
    print(f"  Total Invocations: {int(total_traffic[chosen_row_idx]):,}")
    print(f"  Mean Arrival Rate: {mean_rate:.2f} req/min")
    print(f"  Peak 1-Min Burst : {peak_rate:.2f} req/min")
    print(f"  Fitted Envelope  : rho = {rho:.1f} req/min, sigma = {sigma:.1f}")

    return trace, sigma, rho