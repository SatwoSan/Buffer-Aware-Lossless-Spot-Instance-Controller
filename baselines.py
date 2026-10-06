"""
Baseline policies for the Buffer-Aware Lossless Spot Instance Controller (BALSIC).

Implements the reference policies outlined in the implementation guide:
- All On-Demand (reference zero-risk)
- Static (fixed alpha)
- Reactive (reacts after an eviction)
- Notice-Aware Reactive (reacts on advance warning)
"""

from typing import List, Callable

def policy_all_on_demand(tick: int, x: float, b: List[float], notice_active: bool) -> float:
    """
    The safest, most expensive baseline. Always returns 0.0 (100% On-Demand)[cite: 6].
    
    Parameters
    ----------
    tick : int
        Current simulation tick.
    x : float
        Current buffer backlog.
    b : List[float]
        Current bucket meter states.
    notice_active : bool
        Whether an advance eviction notice is currently active.
        
    Returns
    -------
    float
        Spot fraction (0.0).
    """
    return 0.0


def make_static(alpha: float) -> Callable[[int, float, List[float], bool], float]:
    """
    Returns a policy that always requests a fixed Spot fraction[cite: 6].
    
    Parameters
    ----------
    alpha : float
        The fixed fraction of Spot instances to request (must be in [0, 1]).
        
    Returns
    -------
    Callable
        A stateless policy function.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Static alpha must be in [0, 1], got {alpha}")

    def static_policy(tick: int, x: float, b: List[float], notice_active: bool) -> float:
        return float(alpha)

    return static_policy


class ReactiveBaseline:
    """
    A naive reactive policy. 
    
    Runs a high Spot fraction (`alpha_hi`) normally. When an eviction actually 
    strikes, it drops to 0.0 (all On-Demand) for `cooldown_ticks` to recover, 
    then returns to `alpha_hi`[cite: 6].
    """
    
    def __init__(self, alpha_hi: float, cooldown_ticks: int):
        if not 0.0 <= alpha_hi <= 1.0:
            raise ValueError(f"alpha_hi must be in [0, 1], got {alpha_hi}")
        if cooldown_ticks < 0:
            raise ValueError(f"cooldown_ticks must be >= 0, got {cooldown_ticks}")
            
        self.alpha_hi = float(alpha_hi)
        self.cooldown_ticks = cooldown_ticks
        self.cooldown_remaining = 0

    def on_eviction(self) -> None:
        """Hook called by the simulator EXACTLY when an eviction occurs[cite: 6]."""
        self.cooldown_remaining = self.cooldown_ticks

    def __call__(self, tick: int, x: float, b: List[float], notice_active: bool) -> float:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return 0.0
        return self.alpha_hi


class NoticeAwareBaseline:
    """
    An advanced reactive policy that leverages cloud provider interruption notices.
    
    Like ReactiveBaseline, but it drops to 0.0 as soon as `notice_active` is True, 
    allowing it to preemptively recover before the eviction even lands[cite: 6].
    """
    
    def __init__(self, alpha_hi: float, cooldown_ticks: int):
        if not 0.0 <= alpha_hi <= 1.0:
            raise ValueError(f"alpha_hi must be in [0, 1], got {alpha_hi}")
        if cooldown_ticks < 0:
            raise ValueError(f"cooldown_ticks must be >= 0, got {cooldown_ticks}")
            
        self.alpha_hi = float(alpha_hi)
        self.cooldown_ticks = cooldown_ticks
        self.cooldown_remaining = 0

    def on_eviction(self) -> None:
        """Hook called by the simulator EXACTLY when an eviction occurs."""
        self.cooldown_remaining = self.cooldown_ticks

    def __call__(self, tick: int, x: float, b: List[float], notice_active: bool) -> float:
        # Preemptively start cooldown if we receive advance notice
        if notice_active:
            self.cooldown_remaining = self.cooldown_ticks
            
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return 0.0
        return self.alpha_hi