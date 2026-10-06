"""
Leaky-bucket arrival meter for the Buffer-Aware Lossless Spot Instance Controller (BALSIC).

Implements the stateful tracker from Doc 08 §3.3.
Tracks cumulative arrivals over a continuous window to guarantee they do not 
exceed the sustained rate (rho) plus the maximum allowed single-tick burst (sigma).
"""

class BucketMeter:
    """
    A pure state tracker for a single leaky-bucket arrival curve.
    
    This object maintains the running burst tracker, `b(t)`. It does NOT make 
    admission control decisions (like dropping or shaping traffic) — it only 
    measures whether the traffic stream has violated the theoretical envelope.
    """

    def __init__(self, sigma: float, rho: float):
        """
        Initialise the bucket meter with its theoretical limits.
        
        Parameters
        ----------
        sigma : float
            The burst allowance — maximum extra traffic allowed in a single instant
            above the sustained rate.
        rho : float
            The sustained processing rate — the long-run average traffic limit.
        """
        if sigma <= 0:
            raise ValueError(f"Burst allowance (sigma) must be > 0, got {sigma}")
        if rho <= 0:
            raise ValueError(f"Sustained rate (rho) must be > 0, got {rho}")

        self.sigma = float(sigma)
        self.rho = float(rho)
        
        # Internal state b(t): How much of the burst allowance is currently used up.
        # Starts at 0, representing a completely quiet system.
        self.b = 0.0

    def update(self, arrival: float) -> float:
        """
        Advance the meter by one tick and update the internal burst state.
        
        Formula: b(t) = max(0, b(t-1) + a(t) - rho)
        
        This is a continuous tracker that naturally "drains" by rho each tick 
        and flattens at 0 during quiet periods. It does NOT reset on a fixed window.
        
        Parameters
        ----------
        arrival : float
            The total volume of traffic (a(t)) that arrived during this tick.
            
        Returns
        -------
        float
            The new internal state b(t).
        """
        if arrival < 0:
            raise ValueError(f"Arrivals cannot be negative, got {arrival}")
            
        self.b = max(0.0, self.b + arrival - self.rho)
        return self.b

    def conforms(self) -> bool:
        """
        Check if the traffic trace is currently obeying the bucket's limits.
        
        Returns
        -------
        bool
            True if b(t) <= sigma (the burst allowance has not been exceeded).
            False if the trace has violated the theoretical arrival envelope.
        """
        # Using a tiny epsilon for floating-point safety at the exact boundary
        return self.b <= self.sigma + 1e-9

    def state(self) -> float:
        """
        Return the current burst tracker value for logging or plotting.
        
        Returns
        -------
        float
            The current value of b(t).
        """
        return self.b