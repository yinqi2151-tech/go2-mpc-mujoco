"""Small, dependency-free evaluation statistics."""

from math import sqrt


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval as proportions by default."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("expected 0 <= successes <= trials and trials > 0")
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    half_width = z * sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
