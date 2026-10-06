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


def chain_push_deflection(
    force: float,
    length: float,
    num_segments: int,
    diameter: float,
    bend_modulus: float,
    shear_modulus: float,
) -> tuple[float, float]:
    """Sideways deflection of the clamped stem model under a sideways force on its last segment.

    The force acts at the centre of the last segment, and the deflection is that of the same point. Small
    deflections, no gravity. Same chain as in `chain_tip_sag`; the joint with m segments beyond it is
    (m - 1/2) l away from the force point:
    - bending: moment F (m - 1/2) l, joint angle = moment / (E I / l), deflection = angle * (m - 1/2) l;
    - shear: every joint carries the force F, deflection = F / (G A / l) per joint.

    Args:
        force: Sideways force [N].
        length: Stem length [m].
        num_segments: Number of segments, including the clamped one.
        diameter: Stem diameter [m].
        bend_modulus: Bend modulus E [Pa].
        shear_modulus: Shear modulus G [Pa].

    Returns:
        The bending part and the shear part of the deflection [m]. Their sum is the expected deflection.
    """
    segment_length = length / num_segments
    area = math.pi * diameter**2 / 4
    area_moment = math.pi * diameter**4 / 64
    joint_bend_stiffness = bend_modulus * area_moment / segment_length
    levers = [(m - 0.5) * segment_length for m in range(1, num_segments)]
    bend_deflection = force / joint_bend_stiffness * sum(lever**2 for lever in levers)
    shear_deflection = force * len(levers) / (shear_modulus * area / segment_length)
    return bend_deflection, shear_deflection


def chain_point_compliance(
    height: float, length: float, num_segments: int, diameter: float, bend_modulus: float
) -> float:
    """Sideways deflection per force at `height` of the upright clamped stem model, for a force at `height`.

    Same chain as in `chain_tip_sag` (segment 0 clamped, joint j at height j * l with a bend spring E I / l),
    bending only (no shear spring, as in the `chain` model), small deflections, no gravity. A joint below the
    force point carries the moment F (height - j l) and adds its angle times the same lever to the deflection:
    compliance = sum over the joints below `height` of (height - j l)^2 / (E I / l). The upright stem's own
    weight makes it about 4 % softer.

    Args:
        height: Height of the force and of the deflected point above the base [m].
        length: Stem length [m].
        num_segments: Number of segments, including the clamped one.
        diameter: Stem diameter [m].
        bend_modulus: Bend modulus E [Pa].

    Returns:
        Deflection per force [m/N]; the force for a measured deflection is deflection / compliance.
    """
    segment_length = length / num_segments
    joint_bend_stiffness = bend_modulus * math.pi * diameter**4 / 64 / segment_length
    levers = [height - j * segment_length for j in range(1, num_segments) if j * segment_length < height]
    return sum(lever**2 for lever in levers) / joint_bend_stiffness


def chain_weight_strain(
    length: float, num_segments: int, density: float, stretch_modulus: float, gravity: float = 9.81
) -> list[float]:
    """Axial strain at each joint of the upright stem model, clamped at the base, under its own weight.

    Joint k (k = 0 .. num_segments - 2) connects segments k and k + 1 and carries the weight of the
    num_segments - 1 - k segments above it: strain = -(weight above) / (E_stretch A). The area cancels.

    Args:
        length: Stem length [m].
        num_segments: Number of segments, including the clamped one.
        density: Density [kg/m^3].
        stretch_modulus: Stretch modulus actually simulated [Pa].
        gravity: Gravitational acceleration [m/s^2].

    Returns:
        Strain per joint from the base upwards [-], negative = compression.
    """
    segment_length = length / num_segments
    return [
        -density * gravity * segment_length * (num_segments - 1 - joint) / stretch_modulus
        for joint in range(num_segments - 1)
    ]


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
