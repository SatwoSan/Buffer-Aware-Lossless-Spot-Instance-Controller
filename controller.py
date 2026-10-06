"""
Envelope-aware, state-feedback Spot/On-Demand allocation law.

Core control law for the Buffer-Aware Lossless Spot Instance Controller.
Implements the formula from Doc 08 (Understanding Standalone v2):

  - Leaky-bucket arrival model: cumulative arrivals over any window
    are bounded by sigma + rho * (window length).
  - State-feedback headroom: H = B_max - sigma - max(0, x - b) - phi.
  - Per-bucket alpha: alpha_i = (H_i / T_eff + mu_od - rho_i) / kappa.
  - Final alpha: clip(min over all buckets of alpha_i, 0, 1).

References:
  Doc 08 — Understanding the Project (Standalone, v2 Only)
  Doc 09 — Implementation Guide: Research/Simulation Code (v2)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Result:
    """Output of a single alpha_star() call.

    alpha:     the clipped decision in [0, 1] — fraction of fleet on Spot.
    feasible:  whether the safety constraint is satisfiable at all right now
               (see Doc 08 §5.4). When False, alpha=0 is the least-bad choice
               and an alarm should be raised externally.
    alpha_raw: the value *before* clipping to [0, 1]. Kept for debugging and
               for phi_critical / feasibility round-trip checks.
    """
    alpha: float
    feasible: bool
    alpha_raw: float


class EnvelopeController:
    """The core envelope control law.

    Given the current buffer backlog and bucket meter states, computes the
    maximum safe Spot fraction alpha* such that the buffer is guaranteed
    never to overflow — even under worst-case bucket-conforming arrivals,
    a simultaneous eviction, stale-decision slack, and actuation lag.
    """

    def __init__(
        self,
        mu_od: float,
        mu_sp: float,
        B: float,
        t_mig: float,
        delta: int,
        buckets: List[Tuple[float, float]],
        f: float = 1.0,
        t_up: float = 0.0,
        t_notice: float = 0.0,
    ):
        """Initialise the controller with system parameters."""
        # ---- Input validation ------------------------------------------------
        # Validate every parameter upfront so bugs surface immediately, not deep
        # inside the simulator or experiment scripts.
        if mu_od <= 0:
            raise ValueError(f"mu_od must be > 0, got {mu_od}")
        if mu_sp <= 0:
            raise ValueError(f"mu_sp must be > 0, got {mu_sp}")
        if B <= 0:
            raise ValueError(f"B (buffer capacity) must be > 0, got {B}")
        if t_mig < 0:
            raise ValueError(f"t_mig must be >= 0, got {t_mig}")
        if not isinstance(delta, int) or delta < 1:
            raise ValueError(f"delta must be an int >= 1, got {delta}")
        if not (0 < f <= 1):
            raise ValueError(f"f must be in (0, 1], got {f}")
        if t_up < 0:
            raise ValueError(f"t_up must be >= 0, got {t_up}")
        if t_notice < 0:
            raise ValueError(f"t_notice must be >= 0, got {t_notice}")
        if not buckets:
            raise ValueError("buckets list must contain at least one (sigma, rho) pair")
        for i, (sigma, rho) in enumerate(buckets):
            if sigma <= 0:
                raise ValueError(f"sigma must be > 0 for bucket {i}, got {sigma}")
            if rho <= 0:
                raise ValueError(f"rho must be > 0 for bucket {i}, got {rho}")

        # ---- Store parameters ------------------------------------------------
        self.mu_od = mu_od      # On-Demand fleet capacity
        self.mu_sp = mu_sp      # Spot fleet capacity
        self.B = B              # Buffer capacity (B_max)
        self.f = f              # Fraction of Spot capacity lost on eviction
        self.t_mig = t_mig      # Migration time in ticks
        self.t_notice = t_notice  # Advance eviction notice in ticks
        self.buckets = list(buckets)  # List of (sigma, rho) arrival curves

        # ---- Precompute derived constants ------------------------------------
        # T_eff: worst-case time window the system is exposed to reduced
        # capacity after an eviction. It accounts for:
        #   1. max(0, t_mig - t_notice): migration time minus any advance notice
        #   2. (delta - 1): staleness of the last control decision
        #   3. t_up: actuation lag for new decisions to take effect
        # (Doc 08 §5.2)
        self.t_eff = max(0.0, t_mig - t_notice) + (delta - 1) + t_up

        # kappa: net capacity lost per unit of alpha when an eviction hits.
        # At alpha=1, you lose mu_od worth of capacity (you switch to OD)
        # but retain (1-f)*mu_sp of surviving Spot capacity.
        self.kappa = mu_od - (1.0 - f) * mu_sp

    def surviving_capacity(self, alpha: float) -> float:
        # Capacity available DURING a migration window.

        # This is used by the simulator to compute service rate during recovery,
        # NOT by alpha_star itself.
        # On-Demand part: (1 - alpha) of the fleet processes at mu_od rate
        # Surviving Spot part: alpha of the fleet, but only (1-f) survives
        return (1.0 - alpha) * self.mu_od + alpha * (1.0 - self.f) * self.mu_sp

    def alpha_star(self, x: float, b: List[float], phi: float = 0.0) -> Result:
        """Compute the optimal safe Spot fraction alpha* for this tick.

        This is the heart of the controller. For each leaky bucket, it computes
        headroom (how much buffer space is available for absorbing a worst-case
        eviction), then derives the maximum safe alpha from that headroom. The
        tightest (minimum) alpha across all buckets governs.

        Parameters
        ----------
        x : float
            Current backlog (buffer occupancy at start of tick).
        b : list of float
            Current meter state, one value per bucket, SAME ORDER as
            self.buckets. Each b[i] tracks how much burst allowance is
            currently consumed for bucket i.
        phi : float, default 0.0
            Optional market-stress safety margin in backlog units (Doc 08 §7).
            Phi=0 recovers the base control law; positive values make the
            controller more conservative. Does NOT affect feasibility checks
            (feasibility is about the true physical safety limit, not an
            optional heuristic tightening).

        Returns
        -------
        Result
            alpha: clipped to [0, 1].
            feasible: True if the safety constraint can be satisfied.
            alpha_raw: unclipped value for debugging / phi_critical checks.

        Design decision — feasibility rule
        -----------------------------------
        Feasibility is True if ANY bucket is individually feasible. Rationale:
        each bucket represents a different view of arrival behavior (tight
        short-burst vs. loose long-average). If at least one view says the
        system can be kept safe, the overall constraint is satisfiable (the
        controller will use the tightest alpha, which respects all buckets).
        This matches the convention used in the reference implementation.
        """
        # ---- Special case: kappa <= 0 ----------------------------------------
        # kappa = mu_od - (1-f)*mu_sp. When kappa <= 0, surviving Spot capacity
        # (after eviction) is >= On-Demand capacity. This means MORE Spot is
        # strictly better (or neutral) — there is no eviction risk to hedge.
        # Any alpha in [0,1] is safe; return 1.0 (cheapest) after checking
        # feasibility.
        if self.kappa <= 1e-9:
            any_feasible = False
            for (sigma, rho), bi in zip(self.buckets, b):
                H0 = self.B - sigma - max(0.0, x - bi)
                if H0 >= max(0.0, rho - self.mu_od) * self.t_eff - 1e-9:
                    any_feasible = True
                    break
            return Result(alpha=1.0, feasible=any_feasible, alpha_raw=float('inf'))

        # Track the minimum alpha across all buckets (tightest governs)
        # and whether any bucket is individually feasible.
        alpha_min = float('inf')   # will take the min; start at +inf
        any_feasible = False       # True if ANY bucket allows feasibility
        any_valid = False          # True if ANY bucket produced a valid alpha

        for (sigma, rho), bi in zip(self.buckets, b):
            # ---- Headroom with phi (for alpha computation) ----
            # H_i = B - sigma_i - max(0, x - b_i) - phi
            # "Buffer capacity, minus burst room promised, minus current
            #  backlog not already accounted for by the bucket, minus
            #  optional extra safety margin."  (Doc 08 §5.1)
            H = self.B - sigma - max(0.0, x - bi) - phi

            # ---- Headroom without phi (for feasibility check) ----
            # Feasibility is about the true physical safety limit. The phi
            # term is a voluntary, heuristic tightening; it should not make
            # a physically feasible situation report as infeasible.
            H0 = self.B - sigma - max(0.0, x - bi)

            # ---- Per-bucket feasibility (Doc 08 §5.4) ----
            # A bucket is individually feasible iff:
            #   H0_i >= max(0, rho_i - mu_od) * t_eff
            # i.e. there is enough headroom to cover the worst-case surplus
            # arrivals (above On-Demand capacity) over the full exposure window.
            # Small epsilon (1e-9) for floating-point tolerance.
            if H0 >= max(0.0, rho - self.mu_od) * self.t_eff - 1e-9:
                any_feasible = True

            # ---- Per-bucket alpha_i (Doc 08 §5.3) ----
            # alpha_i = (H_i / T_eff + mu_od - rho_i) / kappa
            # Computed for every bucket regardless of H's sign — the formula
            # naturally produces negative values when H < 0, and the
            # subsequent min + clip handles it. Always computing the exact
            # value (rather than forcing -1.0) preserves algebraic consistency
            # with phi_critical round-trips.

            # Avoid division by zero if t_eff is 0 (instantaneous migration).
            t_eff_safe = max(self.t_eff, 1e-9)

            alpha_i = (H / t_eff_safe + self.mu_od - rho) / self.kappa
            alpha_min = min(alpha_min, alpha_i)
            any_valid = True

        # ---- Determine raw and clipped alpha ----
        if not any_valid:
            # No bucket produced any alpha (shouldn't happen with valid inputs,
            # but handle defensively).
            raw = -1.0
        elif alpha_min == float('inf'):
            # All buckets were skipped — treat as infeasible.
            raw = -1.0
        else:
            raw = alpha_min

        # Clip to [0, 1] for the actual decision; keep raw for diagnostics.
        alpha_clipped = max(0.0, min(1.0, raw))

        return Result(
            alpha=float(alpha_clipped),
            feasible=any_feasible,
            alpha_raw=float(raw),
        )

    def phi_critical(self, x: float, b: List[float]) -> float:
        """Smallest phi that forces alpha_raw <= 0 (i.e. alpha* = 0).

        This tells you "how much market-stress margin would it take to make
        the controller go fully On-Demand?" Useful for calibrating the
        optional phi market-stress extension (Doc 08 §7).

        Takes the MAX across buckets — the bucket that is *most tolerant*
        (i.e. needs the largest phi to force alpha down to zero) is the
        binding constraint, because alpha_star takes the minimum across
        buckets so the tightest bucket already governs.

        Parameters
        ----------
        x : float
            Current backlog.
        b : list of float
            Current meter state, one per bucket.

        Returns
        -------
        float
            The phi value at which alpha_raw would become <= 0.
        """
        vals = []
        for (sigma, rho), bi in zip(self.buckets, b):
            # From alpha_i = (H_i/t_eff + mu_od - rho_i) / kappa = 0,
            # solving for phi:
            # phi_crit_i = B - sigma_i - max(0, x - b_i) - (rho_i - mu_od) * t_eff
            phi_i = (
                self.B
                - sigma
                - max(0.0, x - bi)
                - (rho - self.mu_od) * self.t_eff
            )
            vals.append(phi_i)

        # Max across buckets: the bucket that tolerates the most phi before
        # its alpha_i hits zero is the one that determines phi_critical.
        return max(vals)

    def min_buffer_for_alpha(self, alpha: float, bucket_index: int = 0) -> float:
        """Design-time buffer sizing formula (Doc 08 §5.5).

        Computes the minimum buffer capacity B_max needed to comfortably
        support a target Spot fraction `alpha` during normal operation,
        for one named bucket.

        Formula:
            B_min = sigma + max(0, rho - surviving_capacity(alpha)) * t_eff

        The surviving capacity uses (1-f)*mu_sp for the Spot fraction —
        the portion that survives an eviction. This correctly accounts for
        the reduced capacity during migration.

        Parameters
        ----------
        alpha : float
            Target Spot fraction to support.
        bucket_index : int, default 0
            Which bucket's (sigma, rho) to use.

        Returns
        -------
        float
            Minimum buffer size needed.
        """
        sigma, rho = self.buckets[bucket_index]

        # surviving_capacity gives the processing rate during migration:
        # (1-alpha)*mu_od + alpha*(1-f)*mu_sp
        # If rho exceeds this, surplus arrivals accumulate over t_eff ticks.
        surplus_rate = max(0.0, rho - self.surviving_capacity(alpha))

        return sigma + surplus_rate * self.t_eff