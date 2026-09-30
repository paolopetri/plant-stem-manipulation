"""Reference values for the stem checks: hand-computed sag of the stem model and analysis of a decaying swing.

Used by `scripts/check_stem.py` and `scripts/sweep_stem_solver.py`. No simulator needed.
Tests: `tests/test_stem_reference.py`.
"""

import math
from collections.abc import Sequence


def chain_tip_sag(
    length: float,
    num_segments: int,
    diameter: float,
    density: float,
    bend_modulus: float,
    shear_modulus: float,
    gravity: float = 9.81,
) -> tuple[float, float]:
    """Tip sag of the stem model clamped horizontally, under its own weight, for small deflections.

    The model is a chain of `num_segments` rigid segments of length l = length / num_segments. The first
    segment is clamped. Joint k (k = 1 .. num_segments - 1) sits at distance k * l from the clamp and carries a
    bend spring E I / l and a shear spring G A / l. With m = num_segments - k segments hanging beyond a joint:
    - bending: moment q (m l)^2 / 2, joint angle = moment / (E I / l), tip drop = angle * m l;
    - shear: force q m l, tip drop = force / (G A / l);
    where q = density * A * gravity is the weight per length.

    Args:
        length: Stem length [m].
        num_segments: Number of segments, including the clamped one.
        diameter: Stem diameter [m].
        density: Density [kg/m^3].
        bend_modulus: Bend modulus E [Pa].
        shear_modulus: Shear modulus G [Pa].
        gravity: Gravitational acceleration [m/s^2].

    Returns:
        The bending part and the shear part of the tip sag [m]. Their sum is the expected sag.
    """
    segment_length = length / num_segments
    area = math.pi * diameter**2 / 4
    area_moment = math.pi * diameter**4 / 64
    load = density * area * gravity
    hanging = range(1, num_segments)
    joint_bend_stiffness = bend_modulus * area_moment / segment_length
    bend_sag = load / (2.0 * joint_bend_stiffness) * sum((m * segment_length) ** 3 for m in hanging)
    shear_sag = load * segment_length * sum(hanging) / (shear_modulus * area / segment_length)
    return bend_sag, shear_sag


def beam_first_frequency(length: float, diameter: float, density: float, bend_modulus: float) -> float:
    """First bending frequency [Hz] of a uniform round beam clamped at one end (Euler-Bernoulli, no shear).

    f = (1.8751^2 / (2 pi)) * sqrt(E I / (rho A L^4)). `length` is the free length beyond the clamp [m].
    """
    area = math.pi * diameter**2 / 4
    area_moment = math.pi * diameter**4 / 64
    return 1.8751**2 / (2.0 * math.pi) * math.sqrt(bend_modulus * area_moment / (density * area * length**4))


def analyze_oscillation(signal: Sequence[float], dt: float) -> tuple[float, float, float]:
    """First peak, frequency and damping ratio of a decaying oscillation around zero.

    - Frequency: from the first zero crossings (at most five, i.e. two periods), with the crossing times
      interpolated linearly between samples.
    - Damping ratio: from the logarithmic decrement d = ln(p1 / p2) of the first two positive peaks,
      zeta = d / sqrt(4 pi^2 + d^2).

    Args:
        signal: Samples of the oscillation, e.g. the sideways tip position relative to rest [m].
        dt: Time between samples [s].

    Returns:
        First positive peak (unit of the signal), frequency [Hz], damping ratio [-]. A value that cannot be
        determined (too few peaks or crossings) is nan.
    """
    peaks = [
        signal[i]
        for i in range(1, len(signal) - 1)
        if signal[i] > 0.0 and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]
    ]
    crossings = [
        (i - 1 + signal[i - 1] / (signal[i - 1] - signal[i])) * dt
        for i in range(1, len(signal))
        if signal[i - 1] * signal[i] < 0.0
    ][:5]

    first_peak = peaks[0] if peaks else math.nan
    frequency = math.nan
    if len(crossings) > 1:
        frequency = (len(crossings) - 1) / (2.0 * (crossings[-1] - crossings[0]))
    damping_ratio = math.nan
    if len(peaks) > 1:
        decrement = math.log(peaks[0] / peaks[1])
        damping_ratio = decrement / math.sqrt(4.0 * math.pi**2 + decrement**2)
    return first_peak, frequency, damping_ratio
