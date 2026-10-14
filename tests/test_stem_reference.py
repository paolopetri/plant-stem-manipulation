"""Unit tests for `stem_manip.utils.stem_reference` (no simulator needed)."""

import math

import pytest

from stem_manip.utils import stem_reference

# the plant's values from assets/stem/stem.yaml (shared by all stem models) and the shear modulus of stem model
# `cable` (assets/stem/cable/cable.yaml); `chain` has no shear spring: see the `inf` tests
STEM = dict(length=0.4, num_segments=20, diameter=0.008, density=1000.0, bend_modulus=5.0e8, shear_modulus=5.0e5)


def test_chain_tip_sag_of_the_placeholder_stem():
    bend_sag, shear_sag = stem_reference.chain_tip_sag(**STEM)
    assert bend_sag == pytest.approx(14.17e-3, abs=0.005e-3)
    assert shear_sag == pytest.approx(1.49e-3, abs=0.005e-3)


def test_chain_tip_sag_approaches_the_beam_formula():
    """Many segments and a stiff shear spring -> cantilever under uniform load, delta = q L^4 / (8 E I)."""
    fine = STEM | dict(num_segments=2000, shear_modulus=1.0e15)
    bend_sag, shear_sag = stem_reference.chain_tip_sag(**fine)
    load = STEM["density"] * math.pi * STEM["diameter"] ** 2 / 4 * 9.81
    area_moment = math.pi * STEM["diameter"] ** 4 / 64
    beam_sag = load * STEM["length"] ** 4 / (8.0 * STEM["bend_modulus"] * area_moment)
    assert bend_sag == pytest.approx(beam_sag, rel=2e-3)
    assert shear_sag < 1e-9


def test_chain_tip_sag_without_a_shear_spring_is_all_bending():
    """Stem model `chain` (rigid in shear): shear_modulus = inf -> no shear sag, bending part unchanged."""
    bend_sag, shear_sag = stem_reference.chain_tip_sag(**STEM | dict(shear_modulus=math.inf))
    assert shear_sag == 0.0
    assert bend_sag == pytest.approx(14.17e-3, abs=0.005e-3)


def test_chain_tip_sag_scales_with_the_moduli():
    bend_sag, shear_sag = stem_reference.chain_tip_sag(**STEM)
    stiffer = STEM | dict(bend_modulus=2 * STEM["bend_modulus"], shear_modulus=4 * STEM["shear_modulus"])
    stiffer_bend_sag, stiffer_shear_sag = stem_reference.chain_tip_sag(**stiffer)
    assert stiffer_bend_sag == pytest.approx(bend_sag / 2)
    assert stiffer_shear_sag == pytest.approx(shear_sag / 4)


PUSH = {key: STEM[key] for key in ("length", "num_segments", "diameter", "bend_modulus", "shear_modulus")}


def test_chain_push_deflection_of_the_placeholder_stem():
    bend_deflection, shear_deflection = stem_reference.chain_push_deflection(force=0.05, **PUSH)
    assert bend_deflection == pytest.approx(9.09e-3, abs=0.005e-3)
    assert shear_deflection == pytest.approx(0.76e-3, abs=0.005e-3)


def test_chain_push_deflection_without_a_shear_spring_is_all_bending():
    bend_deflection, shear_deflection = stem_reference.chain_push_deflection(
        force=0.05, **PUSH | dict(shear_modulus=math.inf)
    )
    assert shear_deflection == 0.0
    assert bend_deflection == pytest.approx(9.09e-3, abs=0.005e-3)


def test_chain_push_deflection_approaches_the_beam_formula():
    """Many segments and a stiff shear spring -> cantilever with a force at its end, delta = F L^3 / (3 E I)."""
    fine = PUSH | dict(num_segments=2000, shear_modulus=1.0e15)
    bend_deflection, shear_deflection = stem_reference.chain_push_deflection(force=0.05, **fine)
    area_moment = math.pi * STEM["diameter"] ** 4 / 64
    beam_deflection = 0.05 * STEM["length"] ** 3 / (3.0 * STEM["bend_modulus"] * area_moment)
    assert bend_deflection == pytest.approx(beam_deflection, rel=2e-3)
    assert shear_deflection < 1e-9


def test_chain_push_deflection_is_proportional_to_the_force():
    single = stem_reference.chain_push_deflection(force=0.05, **PUSH)
    double = stem_reference.chain_push_deflection(force=0.10, **PUSH)
    assert double == pytest.approx((2 * single[0], 2 * single[1]))


GEOMETRY = dict(length=0.4, num_segments=20, diameter=0.008, bend_modulus=5.0e8)


def test_chain_point_compliance_matches_the_push_on_the_last_segment():
    """Force at the centre of the last segment: same as the bending part of `chain_push_deflection`."""
    height = 0.4 - 0.5 * 0.4 / 20
    bend_deflection, _ = stem_reference.chain_push_deflection(force=1.0, shear_modulus=1.0e9, **GEOMETRY)
    assert stem_reference.chain_point_compliance(height, **GEOMETRY) == pytest.approx(bend_deflection)


def test_chain_point_compliance_approaches_the_beam_formula():
    """Many segments -> cantilever with the force at the height h, delta / F = h^3 / (3 E I)."""
    geometry = dict(GEOMETRY, num_segments=4000)
    height = 0.3
    beam = height**3 / (3 * 5.0e8 * math.pi * 0.008**4 / 64)
    assert stem_reference.chain_point_compliance(height, **geometry) == pytest.approx(beam, rel=2e-3)


def test_chain_weight_strain():
    strain = stem_reference.chain_weight_strain(length=0.4, num_segments=20, density=1000.0, stretch_modulus=5.0e6)
    assert len(strain) == 19
    assert strain[0] == pytest.approx(-1000.0 * 9.81 * 0.02 * 19 / 5.0e6)  # base joint carries 19 segments
    assert strain[-1] == pytest.approx(strain[0] / 19)  # top joint carries one segment
    assert all(a < b < 0.0 for a, b in zip(strain, strain[1:], strict=False))  # compression decreases upwards


MODE = {key: STEM[key] for key in ("length", "num_segments", "diameter", "density", "bend_modulus")} | dict(
    damping_time=0.0032
)


@pytest.mark.parametrize(
    "gravity, armature, frequency, digits",
    [(9.81, 0.0, 5.109, 3), (9.81, 1.0e-4, 5.055, 3), (0.0, 0.0, 5.21, 2), (0.0, 1.0e-4, 5.15, 2)],
)
def test_chain_first_mode_of_the_placeholder_stem(gravity, armature, frequency, digits):
    """Mode 1 of the exact linear chain, computed independently on 2026-10-05 (docs/notes/2026-10-05.md, given
    there to `digits` decimals)."""
    mode_frequency, _ = stem_reference.chain_first_mode(**MODE, armature=armature, gravity=gravity)
    assert mode_frequency == pytest.approx(frequency, abs=0.5 * 10**-digits)


def test_chain_first_mode_of_one_free_segment_is_an_inverted_pendulum_on_a_spring():
    """Two segments: one joint, one free segment of mass m and length l rotating about its lower end."""
    length, diameter, density, armature, gravity, damping_time = 0.04, 0.008, 1000.0, 1.0e-6, 9.81, 0.0032
    segment_length, radius = length / 2, diameter / 2
    mass = density * math.pi * radius**2 * segment_length
    inertia = mass * (3 * radius**2 + segment_length**2) / 12 + mass * (segment_length / 2) ** 2 + armature
    spring = 5.0e8 * math.pi * diameter**4 / 64 / segment_length
    omega = math.sqrt((spring - mass * gravity * segment_length / 2) / inertia)
    frequency, damping_ratio = stem_reference.chain_first_mode(
        length, 2, diameter, density, 5.0e8, damping_time, armature=armature, gravity=gravity
    )
    assert frequency == pytest.approx(omega / (2 * math.pi), rel=1e-9)
    assert damping_ratio == pytest.approx(damping_time * spring / (2 * omega * inertia), rel=1e-9)


def test_chain_first_mode_approaches_the_beam_formula():
    """Many segments, no weight -> clamped Euler-Bernoulli beam, f = 1.8751^2 / (2 pi) sqrt(E I / (rho A L^4))."""
    area = math.pi * STEM["diameter"] ** 2 / 4
    area_moment = math.pi * STEM["diameter"] ** 4 / 64
    beam = 1.8751**2 / (2 * math.pi) * math.sqrt(STEM["bend_modulus"] * area_moment / (1000.0 * area * 0.4**4))
    frequency, _ = stem_reference.chain_first_mode(**MODE | dict(num_segments=1000), gravity=0.0)
    assert frequency == pytest.approx(beam, rel=2e-3)


def test_chain_first_mode_damping_without_weight_and_with_weight():
    """Without weight, stiffness-proportional damping gives zeta = tau omega / 2 exactly; the weight lowers omega
    but not the damped spring energy, so zeta rises above tau omega / 2."""
    frequency, damping_ratio = stem_reference.chain_first_mode(**MODE, gravity=0.0)
    assert damping_ratio == pytest.approx(MODE["damping_time"] * math.pi * frequency, rel=1e-9)
    frequency, damping_ratio = stem_reference.chain_first_mode(**MODE, gravity=9.81)
    assert damping_ratio > MODE["damping_time"] * math.pi * frequency


def _damped_swing(frequency: float, damping_ratio: float, amplitude: float, dt: float, duration: float) -> list[float]:
    """Response of a damped oscillator that starts at zero with a positive velocity."""
    natural = 2.0 * math.pi * frequency / math.sqrt(1.0 - damping_ratio**2)  # so that `frequency` is the damped one
    steps = round(duration / dt)
    return [
        amplitude * math.exp(-damping_ratio * natural * i * dt) * math.sin(2.0 * math.pi * frequency * i * dt)
        for i in range(steps)
    ]


@pytest.mark.parametrize("frequency, damping_ratio", [(5.0, 0.05), (5.3, 0.02), (3.1, 0.12)])
def test_analyze_oscillation_recovers_frequency_and_damping(frequency, damping_ratio):
    signal = _damped_swing(frequency, damping_ratio, amplitude=0.035, dt=0.01, duration=4.0)
    first_peak, measured_frequency, measured_damping = stem_reference.analyze_oscillation(signal, dt=0.01)
    assert measured_frequency == pytest.approx(frequency, rel=0.01)
    assert measured_damping == pytest.approx(damping_ratio, abs=0.005)
    assert first_peak == pytest.approx(max(signal))


def test_analyze_oscillation_without_a_swing_returns_nan():
    first_peak, frequency, damping_ratio = stem_reference.analyze_oscillation([0.0] * 50, dt=0.01)
    assert math.isnan(first_peak) and math.isnan(frequency) and math.isnan(damping_ratio)


SLOT_PUSH = dict(radius=0.004, plate_thickness=0.018, push_height=0.30, push_depth=0.04)


def test_slot_approach_distance_of_the_push_check():
    # hand value: theta = atan(1.5 * 40.09 mm / 282 mm) = 0.2101 rad, d = 4 mm + 18 mm * sin(theta)
    assert stem_reference.slot_approach_distance(**SLOT_PUSH) == pytest.approx(7.754e-3, abs=0.005e-3)


@pytest.mark.parametrize("change", [dict(push_depth=0.0), dict(plate_thickness=0.0)])
def test_slot_approach_distance_of_an_upright_stem_or_a_thin_plate_is_the_radius(change):
    # no push: the stem stays upright; no plate: the tool tip is on the touching face
    assert stem_reference.slot_approach_distance(**SLOT_PUSH | change) == pytest.approx(0.004, abs=1e-9)


def test_slot_approach_distance_grows_with_the_push_depth():
    shallow = stem_reference.slot_approach_distance(**SLOT_PUSH | dict(push_depth=0.02))
    assert 0.004 < shallow < stem_reference.slot_approach_distance(**SLOT_PUSH)
