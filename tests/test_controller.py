"""
Tests for controller.py — the core envelope control law.

Test structure follows Doc 09 Phase 1 test requirements:
  1. Hand-computed example from Doc 08 §5.3 (two cases).
  2. Brute-force cross-check over random parameter sets.
  3. Feasibility edge cases.
  4. phi_critical round-trip.
  5. Monotonicity sanity checks (property tests).
  + Input validation tests for bad parameters.

Each test is commented to explain what it verifies and why.
"""

import pytest
import random
import math
import sys
import os

# Add the project root to the path so we can import controller
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controller import EnvelopeController, Result


# ===========================================================================
# Test 1: Hand-computed example from Doc 08 §5.3
# ===========================================================================
# These two cases are the "gold standard" from the documentation. If our
# controller doesn't reproduce these exact numbers, the formula is wrong.
# Parameters: mu_od=100, mu_sp=75, f=1, T_mig=30, delta=5, t_up=0
# One bucket: sigma=200, rho=80.  B_max=1500.
# T_eff = max(0, 30-0) + (5-1) + 0 = 34
# kappa = 100 - (1-1)*75 = 100

class TestHandComputedExamples:
    """Reproduce the exact worked examples from Doc 08 §5.3."""

    @pytest.fixture
    def controller(self):
        """Create the controller matching the Doc 08 §5.3 example."""
        # mu_od=100, mu_sp=75, B=1500, t_mig=30, delta=5
        # Single bucket: (sigma=200, rho=80), f=1.0, t_up=0
        return EnvelopeController(
            mu_od=100.0,
            mu_sp=75.0,
            B=1500.0,
            t_mig=30.0,
            delta=5,
            buckets=[(200.0, 80.0)],
            f=1.0,
            t_up=0.0,
        )

    def test_case1_x100(self, controller):
        """Case 1: x=100, b=50, phi=0.

        Expected calculation:
          H = 1500 - 200 - max(0, 100-50) - 0 = 1250
          kappa = 100
          alpha_raw = (1250/34 + 100 - 80) / 100 = (36.76... + 20) / 100 = 0.5676...
          alpha = clip(0.5676, 0, 1) = 0.5676...

        The buffer is relatively empty, so the controller allows ~57% Spot.
        """
        result = controller.alpha_star(x=100.0, b=[50.0], phi=0.0)

        # Verify the hand-computed alpha
        expected_H = 1500 - 200 - max(0, 100 - 50) - 0  # = 1250
        expected_alpha_raw = (expected_H / 34.0 + 100 - 80) / 100.0  # ≈ 0.5676
        assert result.alpha == pytest.approx(expected_alpha_raw, abs=1e-4), (
            f"Case 1 alpha mismatch: got {result.alpha}, expected ~{expected_alpha_raw:.4f}"
        )
        # Should be feasible (buffer is far from full)
        assert result.feasible is True
        # alpha_raw should equal alpha since it's in [0,1]
        assert result.alpha_raw == pytest.approx(expected_alpha_raw, abs=1e-4)

    def test_case2_x1200(self, controller):
        """Case 2: x=1200, b=50, phi=0.

        Expected calculation:
          H = 1500 - 200 - max(0, 1200-50) - 0 = 150
          alpha_raw = (150/34 + 100 - 80) / 100 = (4.41... + 20) / 100 = 0.2441...
          alpha = clip(0.2441, 0, 1) = 0.2441...

        The buffer is much fuller, so the controller pulls back to ~24% Spot.
        This demonstrates the state-feedback behavior: higher backlog → lower alpha.
        """
        result = controller.alpha_star(x=1200.0, b=[50.0], phi=0.0)

        expected_H = 1500 - 200 - max(0, 1200 - 50) - 0  # = 150
        expected_alpha_raw = (expected_H / 34.0 + 100 - 80) / 100.0  # ≈ 0.2441
        assert result.alpha == pytest.approx(expected_alpha_raw, abs=1e-4), (
            f"Case 2 alpha mismatch: got {result.alpha}, expected ~{expected_alpha_raw:.4f}"
        )
        assert result.feasible is True
        assert result.alpha_raw == pytest.approx(expected_alpha_raw, abs=1e-4)


# ===========================================================================
# Test 2: Brute-force cross-check
# ===========================================================================
# For random parameter sets, grid-search alpha in [0,1] and find the largest
# alpha satisfying every bucket's constraint directly (no formula — just
# checking the inequality on a fine grid). Assert it matches alpha_star's
# output within a small tolerance.
#
# This is "the single most valuable test in the whole project" (Doc 09) —
# it's independent evidence the algebra is right, because the grid search
# uses a completely different method to find the answer.

class TestBruteForceGridSearch:
    """Cross-validate alpha_star against an exhaustive grid search."""

    @staticmethod
    def _grid_search_alpha(mu_od, mu_sp, B, t_eff, kappa, buckets, x, b_vals, phi=0.0):
        """Find the largest safe alpha by brute-force grid search.

        For each alpha in a fine grid [0, 1], check whether ALL buckets'
        safety constraints are satisfied:
            H_i / t_eff + mu_od - rho_i >= alpha * kappa
        (which is equivalent to alpha_i >= alpha for bucket i).

        Returns the largest alpha that passes all buckets.
        """
        # When kappa <= 0, more Spot is strictly better (or neutral) even
        # during eviction — every alpha satisfies the constraint.
        if kappa <= 1e-9:
            return 1.0

        best_alpha = 0.0
        grid_size = 2000  # fine enough for < 0.001 tolerance

        for step in range(grid_size + 1):
            alpha = step / grid_size
            all_safe = True

            for (sigma, rho), bi in zip(buckets, b_vals):
                # Headroom for this bucket at this alpha
                H = B - sigma - max(0.0, x - bi) - phi

                # The constraint: the buffer must not overflow even in the
                # worst case over the full exposure window t_eff.
                # Rearranged: H >= alpha * kappa * t_eff - (mu_od - rho) * t_eff
                # → H / t_eff + (mu_od - rho) >= alpha * kappa
                if t_eff > 1e-9:
                    lhs = H / t_eff + mu_od - rho
                else:
                    # If t_eff ≈ 0, the constraint is trivially satisfied
                    # for any alpha (instantaneous migration).
                    lhs = float('inf')

                if lhs < alpha * kappa - 1e-9:
                    all_safe = False
                    break

            if all_safe:
                best_alpha = alpha

        return best_alpha

    @pytest.mark.parametrize("seed", range(200))
    def test_grid_search_matches_formula(self, seed):
        """For a random parameter set, the formula and grid search agree.

        This runs 200 times (one per seed) with randomised parameters.
        Doc 09 requires >= 200 random sets before proceeding to Phase 2.
        """
        rng = random.Random(seed)

        # Random but reasonable parameters
        mu_od = rng.uniform(50, 200)
        mu_sp = rng.uniform(30, 150)
        B = rng.uniform(500, 5000)
        t_mig = rng.uniform(0, 100)
        delta = rng.randint(1, 10)
        f = rng.uniform(0.1, 1.0)
        t_up = rng.uniform(0, 20)

        # 1 or 2 buckets
        n_buckets = rng.randint(1, 2)
        buckets = []
        for _ in range(n_buckets):
            sigma = rng.uniform(50, 500)
            rho = rng.uniform(20, mu_od * 0.95)  # keep rho < mu_od for feasibility
            buckets.append((sigma, rho))

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B, t_mig=t_mig,
            delta=delta, buckets=buckets, f=f, t_up=t_up,
        )

        # Random state within reasonable range
        x = rng.uniform(0, B * 0.8)
        b_vals = [rng.uniform(0, sigma) for sigma, _ in buckets]

        # Get formula answer
        result = ctrl.alpha_star(x, b_vals)

        # Get brute-force grid answer
        grid_alpha = self._grid_search_alpha(
            mu_od, mu_sp, B, ctrl.t_eff, ctrl.kappa, buckets, x, b_vals
        )

        # They should agree within grid resolution (~0.001)
        assert result.alpha == pytest.approx(grid_alpha, abs=0.002), (
            f"Seed {seed}: formula={result.alpha:.4f}, grid={grid_alpha:.4f}, "
            f"raw={result.alpha_raw:.4f}"
        )


# ===========================================================================
# Test 3: Feasibility edge cases
# ===========================================================================
# Doc 09 requires:
#   a) A case where rho > mu_od → should be infeasible regardless of buffer.
#   b) A case exactly on the feasibility boundary.

class TestFeasibilityEdgeCases:
    """Test the feasibility detection logic."""

    def test_rho_exceeds_mu_od_is_infeasible(self):
        """When rho > mu_od, even the full On-Demand fleet can't keep up
        with the sustained arrival rate. No allocation of Spot vs. On-Demand
        can fix this — the system is fundamentally under-provisioned.

        The feasibility condition (Doc 08 §5.4):
            H0 >= max(0, rho - mu_od) * t_eff
        With rho > mu_od and a reasonable buffer, this will fail because
        the RHS grows linearly with t_eff.
        """
        # rho=150 > mu_od=100: sustained rate exceeds On-Demand capacity
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 150.0)], f=1.0,
        )
        # H0 = 500 - 200 - max(0, 0-0) = 300
        # Need: 300 >= max(0, 150-100) * 34 = 50 * 34 = 1700  → False
        result = ctrl.alpha_star(x=0.0, b=[0.0])
        assert result.feasible is False, (
            "Should be infeasible when rho > mu_od and buffer is insufficient"
        )
        # Controller should still return alpha=0 (least-bad choice)
        assert result.alpha == 0.0

    def test_rho_exceeds_mu_od_infeasible_even_with_large_buffer(self):
        """Even with a very large buffer, if rho > mu_od and the surplus
        over t_eff ticks exceeds the buffer, it's infeasible.

        This verifies the claim from Doc 08 §5.4 that the issue is
        fundamental under-provisioning, not buffer sizing.
        """
        # rho - mu_od = 50, t_eff = 34. Surplus = 50*34 = 1700.
        # B must be > sigma + 1700 = 200 + 1700 = 1900 to be feasible.
        ctrl_small = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1800.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 150.0)], f=1.0,
        )
        result_small = ctrl_small.alpha_star(x=0.0, b=[0.0])
        assert result_small.feasible is False  # 1800 < 1900, not enough

        ctrl_large = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=2100.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 150.0)], f=1.0,
        )
        result_large = ctrl_large.alpha_star(x=0.0, b=[0.0])
        assert result_large.feasible is True  # 2100 > 1900, sufficient

    def test_exactly_on_feasibility_boundary(self):
        """Test the case where the system is exactly at the boundary.

        H0 = B - sigma - max(0, x-b) = B - sigma  (when x=0, b=0)
        Feasible iff H0 >= max(0, rho - mu_od) * t_eff

        Set B = sigma + max(0, rho - mu_od) * t_eff exactly.
        """
        sigma, rho, mu_od = 200.0, 120.0, 100.0  # rho > mu_od → rho-mu_od = 20
        t_mig, delta = 30.0, 5
        t_eff = t_mig + (delta - 1)  # = 34
        # B exactly at boundary: sigma + (rho - mu_od) * t_eff = 200 + 20*34 = 880
        B_boundary = sigma + (rho - mu_od) * t_eff  # = 880.0

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=75.0, B=B_boundary,
            t_mig=t_mig, delta=delta, buckets=[(sigma, rho)], f=1.0,
        )
        result = ctrl.alpha_star(x=0.0, b=[0.0])
        # At the exact boundary, should be feasible (H0 == RHS)
        assert result.feasible is True, (
            f"Should be feasible at the boundary: H0={B_boundary-sigma}, "
            f"RHS={(rho-mu_od)*t_eff}"
        )

    def test_just_below_feasibility_boundary(self):
        """Just below the boundary — should be infeasible."""
        sigma, rho, mu_od = 200.0, 120.0, 100.0
        t_mig, delta = 30.0, 5
        t_eff = t_mig + (delta - 1)  # = 34
        B_boundary = sigma + (rho - mu_od) * t_eff  # = 880.0

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=75.0, B=B_boundary - 1.0,  # just below
            t_mig=t_mig, delta=delta, buckets=[(sigma, rho)], f=1.0,
        )
        result = ctrl.alpha_star(x=0.0, b=[0.0])
        assert result.feasible is False

    def test_high_backlog_makes_infeasible(self):
        """Feasibility flips from True to False as x(t) climbs toward B_max.

        Doc 08 §5.4: 'a fuller buffer is closer to overflow.' The backlog
        term max(0, x - b) is *added* to the LHS of the infeasibility
        condition (or equivalently, subtracted from headroom).
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        # With low backlog: feasible
        result_low = ctrl.alpha_star(x=100.0, b=[0.0])
        assert result_low.feasible is True

        # With very high backlog: infeasible (H0 approaches 0 or negative)
        # H0 = 1500 - 200 - max(0, 1400 - 0) = -100  → infeasible
        result_high = ctrl.alpha_star(x=1400.0, b=[0.0])
        assert result_high.feasible is False


# ===========================================================================
# Test 4: phi_critical round-trip
# ===========================================================================
# Compute phi_critical, plug it back into alpha_star, confirm alpha_raw ≈ 0.
# This tests that the two methods are algebraically consistent.

class TestPhiCriticalRoundTrip:
    """Verify phi_critical is consistent with alpha_star."""

    def test_single_bucket_round_trip(self):
        """phi_critical should produce a phi that makes alpha_raw ≈ 0.

        Compute phi_crit, then call alpha_star with phi=phi_crit.
        The resulting alpha_raw should be approximately 0.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        x, b = 100.0, [50.0]

        phi_crit = ctrl.phi_critical(x, b)

        # Plug phi_crit back in — alpha_raw should be ≈ 0
        result = ctrl.alpha_star(x, b, phi=phi_crit)
        assert result.alpha_raw == pytest.approx(0.0, abs=1e-4), (
            f"phi_critical round-trip failed: phi_crit={phi_crit:.4f}, "
            f"alpha_raw={result.alpha_raw:.4f} (expected ≈ 0)"
        )

    def test_dual_bucket_round_trip(self):
        """Same round-trip check but with two buckets.

        phi_critical takes the max across buckets. With multiple buckets,
        alpha_star takes the MIN of per-bucket alpha_i values. At phi_crit,
        the loosest bucket's alpha_i = 0 but the tightest may go negative.
        So alpha (clipped) should be 0, and alpha_raw should be <= 0.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=2000.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0), (500.0, 60.0)], f=1.0,
        )
        x, b = 200.0, [30.0, 100.0]

        phi_crit = ctrl.phi_critical(x, b)
        result = ctrl.alpha_star(x, b, phi=phi_crit)
        # With dual buckets, alpha_raw may be slightly negative (not exactly 0)
        # because phi_crit is set by the loosest bucket's threshold.
        # The key invariant: clipped alpha must be 0.
        assert result.alpha == 0.0, (
            f"Dual-bucket phi_critical round-trip failed: "
            f"phi_crit={phi_crit:.4f}, alpha={result.alpha:.4f}"
        )
        assert result.alpha_raw <= 1e-4, (
            f"alpha_raw should be <= 0 at phi_crit, got {result.alpha_raw:.4f}"
        )

    @pytest.mark.parametrize("seed", range(50))
    def test_random_round_trip(self, seed):
        """Randomised phi_critical round-trip across diverse parameters.

        Uses a single bucket so that phi_critical and alpha_star are
        algebraically exact inverses (no min-across-buckets complication).
        Skips cases where kappa <= 0 (Spot surviving capacity exceeds OD)
        since phi_critical is meaningless when alpha is always 1.
        """
        rng = random.Random(seed)
        mu_od = rng.uniform(50, 200)
        mu_sp = rng.uniform(30, 150)
        B = rng.uniform(1000, 5000)
        t_mig = rng.uniform(10, 80)
        delta = rng.randint(1, 10)
        f = rng.uniform(0.3, 1.0)
        sigma = rng.uniform(50, 400)
        rho = rng.uniform(20, mu_od * 0.9)

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B, t_mig=t_mig,
            delta=delta, buckets=[(sigma, rho)], f=f,
        )

        # Skip kappa <= 0 cases — phi has no effect when alpha is always 1
        if ctrl.kappa <= 1e-9:
            pytest.skip("kappa <= 0: phi_critical is undefined")

        x = rng.uniform(0, B * 0.5)
        b_val = rng.uniform(0, sigma)

        phi_crit = ctrl.phi_critical(x, [b_val])
        result = ctrl.alpha_star(x, [b_val], phi=phi_crit)
        assert result.alpha_raw == pytest.approx(0.0, abs=1e-3), (
            f"Seed {seed}: phi_crit={phi_crit:.2f}, alpha_raw={result.alpha_raw:.4f}"
        )


# ===========================================================================
# Test 5: Monotonicity sanity checks (property tests)
# ===========================================================================
# Doc 09 requires these as property tests over random inputs, not single examples:
#   a) Increasing x (more backlog) should never increase alpha*.
#   b) Increasing phi should never increase alpha*.
#   c) Increasing B should never decrease alpha*.

class TestMonotonicity:
    """Verify monotonicity properties that must hold by construction."""

    @pytest.mark.parametrize("seed", range(100))
    def test_increasing_backlog_never_increases_alpha(self, seed):
        """More backlog → less (or equal) Spot. Never more.

        The state-feedback term max(0, x - b) grows with x, reducing
        headroom H, which reduces alpha*. This is the core safety property:
        the controller gets cautious as the buffer fills up.
        """
        rng = random.Random(seed)
        mu_od = rng.uniform(50, 200)
        mu_sp = rng.uniform(30, 150)
        B = rng.uniform(1000, 5000)
        t_mig = rng.uniform(5, 80)
        delta = rng.randint(1, 10)
        f = rng.uniform(0.2, 1.0)
        sigma = rng.uniform(50, 400)
        rho = rng.uniform(20, mu_od * 0.9)

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B, t_mig=t_mig,
            delta=delta, buckets=[(sigma, rho)], f=f,
        )

        b_val = rng.uniform(0, sigma * 0.8)

        # Pick two backlog values where x_low < x_high
        x_low = rng.uniform(0, B * 0.4)
        x_high = rng.uniform(x_low + 1, B * 0.9)

        r_low = ctrl.alpha_star(x_low, [b_val])
        r_high = ctrl.alpha_star(x_high, [b_val])

        assert r_high.alpha <= r_low.alpha + 1e-9, (
            f"Seed {seed}: alpha increased with backlog! "
            f"x={x_low:.1f}→{x_high:.1f}, alpha={r_low.alpha:.4f}→{r_high.alpha:.4f}"
        )

    @pytest.mark.parametrize("seed", range(100))
    def test_increasing_phi_never_increases_alpha(self, seed):
        """More market-stress margin phi → less (or equal) Spot.

        phi is subtracted from headroom H, so larger phi means smaller H,
        which means smaller alpha*. The controller gets more conservative
        when market stress is signalled.
        """
        rng = random.Random(seed)
        mu_od = rng.uniform(50, 200)
        mu_sp = rng.uniform(30, 150)
        B = rng.uniform(1000, 5000)
        t_mig = rng.uniform(5, 80)
        delta = rng.randint(1, 10)
        f = rng.uniform(0.2, 1.0)
        sigma = rng.uniform(50, 400)
        rho = rng.uniform(20, mu_od * 0.9)

        ctrl = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B, t_mig=t_mig,
            delta=delta, buckets=[(sigma, rho)], f=f,
        )

        x = rng.uniform(0, B * 0.5)
        b_val = rng.uniform(0, sigma * 0.8)

        phi_low = rng.uniform(0, 100)
        phi_high = rng.uniform(phi_low + 1, phi_low + 500)

        r_low = ctrl.alpha_star(x, [b_val], phi=phi_low)
        r_high = ctrl.alpha_star(x, [b_val], phi=phi_high)

        assert r_high.alpha <= r_low.alpha + 1e-9, (
            f"Seed {seed}: alpha increased with phi! "
            f"phi={phi_low:.1f}→{phi_high:.1f}, alpha={r_low.alpha:.4f}→{r_high.alpha:.4f}"
        )

    @pytest.mark.parametrize("seed", range(100))
    def test_increasing_B_never_decreases_alpha(self, seed):
        """Larger buffer → more (or equal) Spot.

        A larger buffer gives more headroom to absorb worst-case arrivals
        during an eviction recovery, so the controller can safely allow
        a higher Spot fraction.
        """
        rng = random.Random(seed)
        mu_od = rng.uniform(50, 200)
        mu_sp = rng.uniform(30, 150)
        t_mig = rng.uniform(5, 80)
        delta = rng.randint(1, 10)
        f = rng.uniform(0.2, 1.0)
        sigma = rng.uniform(50, 400)
        rho = rng.uniform(20, mu_od * 0.9)

        B_low = rng.uniform(500, 3000)
        B_high = rng.uniform(B_low + 100, B_low + 3000)

        ctrl_low = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B_low, t_mig=t_mig,
            delta=delta, buckets=[(sigma, rho)], f=f,
        )
        ctrl_high = EnvelopeController(
            mu_od=mu_od, mu_sp=mu_sp, B=B_high, t_mig=t_mig,
            delta=delta, buckets=[(sigma, rho)], f=f,
        )

        x = rng.uniform(0, B_low * 0.4)
        b_val = rng.uniform(0, sigma * 0.8)

        r_low = ctrl_low.alpha_star(x, [b_val])
        r_high = ctrl_high.alpha_star(x, [b_val])

        assert r_high.alpha >= r_low.alpha - 1e-9, (
            f"Seed {seed}: alpha decreased with larger buffer! "
            f"B={B_low:.0f}→{B_high:.0f}, alpha={r_low.alpha:.4f}→{r_high.alpha:.4f}"
        )


# ===========================================================================
# Test 6: Input validation — bad parameters should raise ValueError
# ===========================================================================
# "Validate inputs and raise on bad values — catching a bad parameter here
#  beats debugging it three files later." (Doc 09, Phase 1)

class TestInputValidation:
    """Verify that invalid constructor parameters raise ValueError."""

    # Default valid parameters for easy one-at-a-time override
    VALID = dict(
        mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
        delta=5, buckets=[(200.0, 80.0)], f=1.0,
    )

    def _make(self, **overrides):
        """Create a controller with valid defaults, overriding specific params."""
        params = {**self.VALID, **overrides}
        return EnvelopeController(**params)

    def test_mu_od_zero_raises(self):
        """mu_od=0 means no On-Demand capacity — physically meaningless."""
        with pytest.raises(ValueError, match="mu_od"):
            self._make(mu_od=0.0)

    def test_mu_od_negative_raises(self):
        """Negative capacity is nonsensical."""
        with pytest.raises(ValueError, match="mu_od"):
            self._make(mu_od=-10.0)

    def test_mu_sp_zero_raises(self):
        """mu_sp=0 means Spot instances do nothing — no point controlling."""
        with pytest.raises(ValueError, match="mu_sp"):
            self._make(mu_sp=0.0)

    def test_mu_sp_negative_raises(self):
        """Negative Spot capacity is nonsensical."""
        with pytest.raises(ValueError, match="mu_sp"):
            self._make(mu_sp=-5.0)

    def test_B_zero_raises(self):
        """Zero buffer capacity means everything is dropped — no control needed."""
        with pytest.raises(ValueError, match="B"):
            self._make(B=0.0)

    def test_B_negative_raises(self):
        """Negative buffer is physically impossible."""
        with pytest.raises(ValueError, match="B"):
            self._make(B=-100.0)

    def test_t_mig_negative_raises(self):
        """Negative migration time is meaningless (t_mig=0 is valid: instant)."""
        with pytest.raises(ValueError, match="t_mig"):
            self._make(t_mig=-1.0)

    def test_delta_zero_raises(self):
        """delta=0 means 'recompute every 0 ticks' — undefined."""
        with pytest.raises(ValueError, match="delta"):
            self._make(delta=0)

    def test_delta_negative_raises(self):
        """Negative control period is meaningless."""
        with pytest.raises(ValueError, match="delta"):
            self._make(delta=-3)

    def test_delta_float_raises(self):
        """delta must be an integer (it's a tick count)."""
        with pytest.raises(ValueError, match="delta"):
            self._make(delta=2.5)

    def test_f_zero_raises(self):
        """f=0 means 'no capacity is lost during eviction' — then there's no
        risk to hedge against and kappa would be ill-defined. f must be > 0."""
        with pytest.raises(ValueError, match="f"):
            self._make(f=0.0)

    def test_f_above_one_raises(self):
        """f > 1 means 'more than 100% of Spot is lost' — impossible."""
        with pytest.raises(ValueError, match="f"):
            self._make(f=1.5)

    def test_f_negative_raises(self):
        """Negative loss fraction is nonsensical."""
        with pytest.raises(ValueError, match="f"):
            self._make(f=-0.1)

    def test_empty_buckets_raises(self):
        """Must have at least one arrival curve to reason about safety."""
        with pytest.raises(ValueError, match="bucket"):
            self._make(buckets=[])

    def test_bucket_zero_sigma_raises(self):
        """sigma=0 means 'no burst allowed at all' — too restrictive to be useful."""
        with pytest.raises(ValueError, match="sigma"):
            self._make(buckets=[(0.0, 80.0)])

    def test_bucket_negative_rho_raises(self):
        """Negative sustained rate is meaningless."""
        with pytest.raises(ValueError, match="rho"):
            self._make(buckets=[(200.0, -10.0)])

    def test_t_up_negative_raises(self):
        """Negative actuation lag is meaningless."""
        with pytest.raises(ValueError, match="t_up"):
            self._make(t_up=-1.0)

    def test_t_notice_negative_raises(self):
        """Negative notice time is meaningless."""
        with pytest.raises(ValueError, match="t_notice"):
            self._make(t_notice=-1.0)


# ===========================================================================
# Test 7: Precomputed values (t_eff and kappa)
# ===========================================================================
# Verify the derived constants are computed correctly.

class TestPrecomputedValues:
    """Verify t_eff and kappa are computed per the formulas in Doc 08."""

    def test_t_eff_basic(self):
        """T_eff = max(0, t_mig - t_notice) + (delta - 1) + t_up.

        With t_mig=30, delta=5, t_up=0, t_notice=0:
        T_eff = 30 + 4 + 0 = 34.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.t_eff == pytest.approx(34.0)

    def test_t_eff_with_notice(self):
        """Advance notice reduces the effective exposure window.

        t_notice=10: T_eff = max(0, 30-10) + 4 + 0 = 24.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0, t_notice=10.0,
        )
        assert ctrl.t_eff == pytest.approx(24.0)

    def test_t_eff_notice_exceeds_migration(self):
        """When notice > t_mig, the max(0, ...) clamp kicks in.

        If you get 50 ticks of notice but migration only takes 30,
        the migration-related exposure is 0. Only staleness + actuation remain.
        T_eff = max(0, 30-50) + 4 + 0 = 0 + 4 = 4.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0, t_notice=50.0,
        )
        assert ctrl.t_eff == pytest.approx(4.0)

    def test_t_eff_with_actuation_lag(self):
        """Actuation lag adds to the exposure window.

        t_up=10: T_eff = 30 + 4 + 10 = 44.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0, t_up=10.0,
        )
        assert ctrl.t_eff == pytest.approx(44.0)

    def test_kappa_full_loss(self):
        """kappa = mu_od - (1-f)*mu_sp.

        With f=1 (total loss): kappa = 100 - 0*75 = 100.
        All Spot capacity is lost, so every unit of alpha costs mu_od.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.kappa == pytest.approx(100.0)

    def test_kappa_partial_loss(self):
        """With f=0.5 (half the Spot fleet survives): kappa = 100 - 0.5*75 = 62.5.

        Less capacity is lost per unit alpha, so the controller can be more
        aggressive (higher alpha for the same headroom).
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=0.5,
        )
        assert ctrl.kappa == pytest.approx(62.5)


# ===========================================================================
# Test 8: surviving_capacity
# ===========================================================================
# Verify the capacity-during-migration calculation.

class TestSurvivingCapacity:
    """Verify the surviving capacity formula during eviction."""

    def test_f1_alpha1_zero_capacity(self):
        """With f=1 (total loss) and alpha=1 (all Spot), nothing survives.

        surviving_capacity = (1-1)*100 + 1*(1-1)*75 = 0.
        All instances were Spot and all are gone.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.surviving_capacity(1.0) == pytest.approx(0.0)

    def test_f1_alpha0_full_capacity(self):
        """With alpha=0 (all On-Demand), nothing is lost — full OD capacity.

        surviving_capacity = (1-0)*100 + 0*(1-1)*75 = 100.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.surviving_capacity(0.0) == pytest.approx(100.0)

    def test_partial_f_partial_alpha(self):
        """With f=0.5, alpha=0.6:
        surviving = (1-0.6)*100 + 0.6*(1-0.5)*75 = 40 + 22.5 = 62.5.
        40% OD capacity + 30% of Spot capacity (half of 60% Spot survives).
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=0.5,
        )
        assert ctrl.surviving_capacity(0.6) == pytest.approx(62.5)


# ===========================================================================
# Test 9: min_buffer_for_alpha (design-time sizing)
# ===========================================================================
# Verify the buffer sizing formula from Doc 08 §5.5.

class TestMinBufferForAlpha:
    """Verify design-time buffer sizing formula."""

    def test_alpha_zero_just_sigma(self):
        """With alpha=0 (all On-Demand), surviving capacity = mu_od.

        If rho <= mu_od, no surplus accumulates → B_min = sigma.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        # rho=80 < mu_od=100 → surplus = 0 → B_min = sigma = 200
        assert ctrl.min_buffer_for_alpha(0.0) == pytest.approx(200.0)

    def test_alpha_one_f1(self):
        """With alpha=1, f=1: surviving_capacity = 0.

        B_min = sigma + max(0, rho - 0) * t_eff = 200 + 80*34 = 2920.
        Need a huge buffer because all capacity is lost during migration.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=5000.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        expected = 200.0 + 80.0 * 34.0  # = 2920
        assert ctrl.min_buffer_for_alpha(1.0) == pytest.approx(expected)

    def test_second_bucket_index(self):
        """Can compute sizing for a specific bucket by index."""
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=5000.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0), (500.0, 60.0)], f=1.0,
        )
        # Bucket 1: sigma=500, rho=60
        # surviving_capacity(0.5) = 0.5*100 + 0.5*0*75 = 50
        # surplus = max(0, 60-50) = 10
        # B_min = 500 + 10*34 = 840
        expected = 500.0 + max(0, 60 - ctrl.surviving_capacity(0.5)) * 34.0
        assert ctrl.min_buffer_for_alpha(0.5, bucket_index=1) == pytest.approx(expected)


# ===========================================================================
# Test 10: Dual-bucket behavior
# ===========================================================================
# Verify that with two buckets, the tightest (minimum alpha) governs.

class TestDualBucket:
    """Verify multi-bucket logic: tightest bucket governs alpha."""

    def test_min_alpha_across_buckets(self):
        """With two buckets, alpha_star should return the minimum of the two
        per-bucket alpha values (the most conservative constraint wins).

        This prevents the controller from exceeding what ANY bucket allows.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=2000.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0), (500.0, 60.0)], f=1.0,
        )
        x, b = 100.0, [50.0, 100.0]
        result = ctrl.alpha_star(x, b)

        # Compute per-bucket alphas independently
        t_eff = ctrl.t_eff
        kappa = ctrl.kappa

        H1 = 2000 - 200 - max(0, x - 50.0)
        alpha_1 = (H1 / t_eff + 100 - 80) / kappa

        H2 = 2000 - 500 - max(0, x - 100.0)
        alpha_2 = (H2 / t_eff + 100 - 60) / kappa

        expected = min(alpha_1, alpha_2)
        expected_clipped = max(0, min(1, expected))

        assert result.alpha == pytest.approx(expected_clipped, abs=1e-4), (
            f"Dual-bucket alpha mismatch: got {result.alpha:.4f}, "
            f"expected min({alpha_1:.4f}, {alpha_2:.4f}) = {expected_clipped:.4f}"
        )


# ===========================================================================
# Test 11: Edge cases and special values
# ===========================================================================

class TestEdgeCases:
    """Cover corner cases and boundary conditions."""

    def test_alpha_clips_to_zero_when_negative_raw(self):
        """When the formula produces a negative alpha_raw, it should be
        clipped to 0 (go fully On-Demand). The raw value is preserved
        for diagnostics.
        """
        # Buffer overloaded: x=1200 with B=500.
        # H = 500 - 200 - max(0, 1200-0) = -900
        # alpha_i = (-900/34 + 100 - 80)/100 = (-26.47 + 20)/100 ≈ -0.065
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        result = ctrl.alpha_star(x=1200.0, b=[0.0])
        assert result.alpha == 0.0
        assert result.alpha_raw < 0.0  # raw preserved for debugging

    def test_alpha_clips_to_one_when_above(self):
        """When the formula produces alpha_raw > 1, it clips to 1.

        This can happen with very generous parameters (huge buffer,
        low arrival rate, short migration time).
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=50000.0, t_mig=1.0,
            delta=1, buckets=[(100.0, 10.0)], f=1.0,
        )
        result = ctrl.alpha_star(x=0.0, b=[0.0])
        assert result.alpha == 1.0
        assert result.alpha_raw > 1.0  # raw is above 1 before clipping

    def test_empty_buffer_maximum_alpha(self):
        """An empty buffer with bucket meter at 0 should give maximum alpha
        (most room to absorb an eviction's aftermath).
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        result = ctrl.alpha_star(x=0.0, b=[0.0])
        # Should be feasible and alpha should be > 0
        assert result.feasible is True
        assert result.alpha > 0.0

    def test_f_equals_one_surviving_is_zero(self):
        """Sanity check: f=1 → surviving_capacity with alpha=1 is 0.

        Doc 09 explicitly calls this out: 'double-check your own code
        against the f=1 -> surviving_capacity()==0 behavior before
        trusting it.'
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=5, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.surviving_capacity(1.0) == pytest.approx(0.0)

    def test_delta_one_minimal_staleness(self):
        """delta=1 means alpha is recomputed every tick → staleness term is 0.

        T_eff = t_mig + (1-1) + t_up = t_mig + t_up.
        """
        ctrl = EnvelopeController(
            mu_od=100.0, mu_sp=75.0, B=1500.0, t_mig=30.0,
            delta=1, buckets=[(200.0, 80.0)], f=1.0,
        )
        assert ctrl.t_eff == pytest.approx(30.0)
