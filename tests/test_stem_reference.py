"""Unit tests for `stem_manip.utils.stem_reference` (no simulator needed)."""

import math

import pytest

from stem_manip.utils import stem_reference

# placeholder stem of stem model `cable` (assets/stem/stem.yaml + assets/stem/cable/cable.yaml)
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
    assert all(a < b < 0.0 for a, b in zip(strain, strain[1:]))  # compression decreases upwards


def test_beam_first_frequency():
    beam = {key: STEM[key] for key in ("length", "diameter", "density", "bend_modulus")}
    frequency = stem_reference.beam_first_frequency(**beam)
    assert frequency == pytest.approx(4.946, abs=0.002)  # hand value for the 0.4 m placeholder stem
    assert stem_reference.beam_first_frequency(**beam | dict(length=0.2)) == pytest.approx(4 * frequency)


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
