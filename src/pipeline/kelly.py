# src/pipeline/kelly.py

import numpy as np

def compute_kelly_fraction(returns: np.ndarray) -> float:
    """
    Full-Kelly на основе исторических returns (для backtest-анализа).
    """
    if len(returns) == 0:
        return 0.0
    p = np.mean(returns > 0)
    wins = returns[returns > 0]
    losses = -returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    b = wins.mean() / losses.mean()
    f = (p * (b + 1) - 1) / b
    return float(np.clip(f, 0.0, 1.0))


def kelly_fraction(p: float, b: float, kelly_coeff: float = 1.0) -> float:
    """
    Fractional-Kelly: f* = kelly_coeff * ((p*(b+1) - 1)/b), clipped to [0,1].
    """
    f_star = (p * (b + 1) - 1) / b
    return float(np.clip(f_star * kelly_coeff, 0.0, 1.0))
