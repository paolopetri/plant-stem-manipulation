"""Reference values for the stem checks: hand-computed sag of the stem model and analysis of a decaying swing.

Used by `scripts/check_stem.py`, `scripts/sweep_stem_solver.py` and `scripts/check_push_terms.py`. No simulator needed.
Tests: `tests/test_stem_reference.py`.
"""

import math
from collections.abc import Sequence

import numpy as np


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


def slot_approach_distance(radius: float, plate_thickness: float, push_height: float, push_depth: float) -> float:
    """Distance d from the slot base to the stem's centre line with the stem pushed into the fork's slot.

    The slot base sits at the slot bottom on the plate's top face; the plate is `plate_thickness` thick below it.
    The slot bottom seats the upright stem with its centre `radius` from the slot base. Pushed, the stem leans away
    from the fork by theta and rests against the slot bottom at the plate's bottom face, where its centre is
    radius / cos(theta) from the slot bottom, so d = radius + plate_thickness * sin(theta). theta is the slope of
    the clamped stem at a point load at a = push_height - plate_thickness (cantilever, small deflections:
    slope = 1.5 * delta / a), with the deflection delta = push_depth - radius + radius / cos(theta) there, where
    `push_depth` is how far the slot base moves beyond first contact with the upright stem's surface.

    Args:
        radius: Stem radius [m].
        plate_thickness: Plate thickness at the slot bottom, below the slot base [m].
        push_height: Height of the slot base above the stem base [m].
        push_depth: Slot-base travel beyond first contact [m].

    Returns:
        d [m].
    """
    lever = push_height - plate_thickness
    theta = 0.0
    for _ in range(50):  # fixed point of theta = atan(1.5 * delta(theta) / lever); delta depends weakly on theta
        theta = math.atan(1.5 * (push_depth - radius + radius / math.cos(theta)) / lever)
    return radius + plate_thickness * math.sin(theta)


def chain_first_mode(
    length: float,
    num_segments: int,
    diameter: float,
    density: float,
    bend_modulus: float,
    damping_time: float,
    armature: float = 0.0,
    gravity: float = 9.81,
) -> tuple[float, float]:
    """First bending mode of the upright stem model, clamped at the base: frequency and damping ratio.

    Exact linear model of the chain in one bending plane (small angles). Segment 0 is clamped; joint j
    (j = 1 .. num_segments - 1) at height j l has the relative angle theta_j, a bend spring k = E I / l, a damper
    damping_time * k and the joint `armature`. Segment i (mass m = rho A l, a solid cylinder with the moment of
    inertia m (3 r^2 + l^2) / 12 about its centre, as in the `chain` USD) tilts by phi_i = theta_1 + ... + theta_i;
    its centre moves sideways by l (phi_1 + ... + phi_(i-1)) + l phi_i / 2. Gravity lowers the stiffness of the
    upright stem: segment i's weight above its tilted lower neighbours gives the potential
    -m g l / 4 * (2 (phi_1^2 + ... + phi_(i-1)^2) + phi_i^2). Mode 1 of M theta'' + (K_spring - K_gravity) theta = 0
    gives the frequency; its damping ratio is damping_time * (phi^T K_spring phi) / (2 omega phi^T M phi) for the
    mode shape phi of the undamped chain (light damping; only the springs are damped, so with gravity it is above
    damping_time * omega / 2).

    Args:
        length: Stem length [m].
        num_segments: Number of segments, including the clamped one.
        diameter: Stem diameter [m].
        density: Density [kg/m^3].
        bend_modulus: Bend modulus E [Pa].
        damping_time: Damping time tau [s] of the joint dampers (damping = tau * stiffness).
        armature: Extra inertia on every joint coordinate [kg m^2].
        gravity: Gravitational acceleration [m/s^2]; 0 for the stem without its weight.

    Returns:
        Frequency [Hz] and damping ratio [-] of the first bending mode.
    """
    segment_length = length / num_segments
    radius = diameter / 2
    mass = density * math.pi * radius**2 * segment_length
    centre_inertia = mass * (3 * radius**2 + segment_length**2) / 12
    joint_stiffness = bend_modulus * math.pi * diameter**4 / 64 / segment_length
    free = num_segments - 1
    tilt = np.tril(np.ones((free, free)))  # phi = tilt @ theta
    centre = np.tril(np.full((free, free), segment_length), -1) + np.eye(free) * segment_length / 2  # x = centre @ phi
    sideways = centre @ tilt
    mass_matrix = mass * sideways.T @ sideways + centre_inertia * tilt.T @ tilt + armature * np.eye(free)
    spring = joint_stiffness * np.eye(free)
    # phi_k^2 carries the weight of the segments above segment k plus half its own: m g l (free - k + 1/2), k = 1..
    weight = mass * gravity * segment_length * (free - np.arange(free) - 0.5)
    stiffness = spring - tilt.T @ np.diag(weight) @ tilt
    # symmetric generalized eigenproblem K phi = omega^2 M phi via M = C C^T (eigh: ascending eigenvalues)
    cholesky_inv = np.linalg.inv(np.linalg.cholesky(mass_matrix))
    eigenvalues, shapes = np.linalg.eigh(cholesky_inv @ stiffness @ cholesky_inv.T)
    omega = math.sqrt(eigenvalues[0])
    shape = cholesky_inv.T @ shapes[:, 0]
    damping_ratio = damping_time * (shape @ spring @ shape) / (2.0 * omega * (shape @ mass_matrix @ shape))
    return omega / (2.0 * math.pi), float(damping_ratio)


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
