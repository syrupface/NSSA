""" 2026-11-03: contact mechanics (contact.py): every model recovers the modulus of synthetic curves. """

import math

import numpy as np
import pytest

import contact

RNG = np.random.default_rng(3)


def _approach(tip, e_star, delta0=50e-9, depth=20e-9, noise=0.0, exact=False, n=400):
    delta = np.linspace(delta0 - 2 * depth, delta0 + depth, n)
    force = e_star * contact.unit_force(delta - delta0, tip, exact)
    return delta, force + noise * force.max() * RNG.standard_normal(n)


@pytest.mark.parametrize("shape", ["sphere", "cone", "pyramid", "punch", "hyperboloid"])
def test_sneddon_recovers_the_modulus(shape):
    tip = contact.Tip(shape, radius=20e-9, half_angle=25.0, poisson=0.5)
    delta, force = _approach(tip, 2e9, exact=True, noise=0.01)
    fit = contact.fit_elastic(delta, force, tip, "sneddon")
    assert fit.reduced_modulus == pytest.approx(2e9, rel=0.03)
    assert fit.modulus == pytest.approx(2e9 * 0.75, rel=0.03)
    assert fit.contact_point == pytest.approx(50e-9, abs=1e-9)
    assert fit.r2 > 0.99


def test_hertz_and_exact_sphere_agree_for_small_indentation():
    tip = contact.Tip("sphere", radius=1e-6)
    h = np.linspace(0, 5e-9, 20)
    hertz, exact = contact.unit_force(h, tip), contact.unit_force(h, tip, exact_sphere=True)
    assert np.allclose(exact[1:], hertz[1:], rtol=0.01)
    deep = contact.unit_force([0.5e-6], tip, exact_sphere=True)[0]    # deep: the sphere is narrower at the rim of
    assert deep < contact.unit_force([0.5e-6], tip)[0]                 # the contact than the paraboloid: less force


def test_hertz_fit_with_noise():
    tip = contact.Tip("sphere", radius=10e-9, poisson=0.3)
    delta, force = _approach(tip, 5e9, noise=0.02)
    fit = contact.fit_elastic(delta, force, tip, "hertz")
    assert fit.modulus == pytest.approx(5e9 * (1 - 0.09), rel=0.05)


def test_hyperboloid_limits():
    tip = contact.Tip("hyperboloid", radius=10e-9, half_angle=20.0)
    small = contact.unit_force([1e-11], tip)[0]
    assert small == pytest.approx(contact.unit_force([1e-11], contact.Tip("sphere", 10e-9))[0], rel=0.02)
    deep = contact.unit_force([2e-6], tip)[0]
    cone = contact.unit_force([2e-6], contact.Tip("cone", half_angle=20.0))[0]
    assert deep == pytest.approx(cone, rel=0.05)


def test_dmt_on_retract():
    tip = contact.Tip("sphere", radius=10e-9, poisson=0.5)
    e_star, f_ad, delta0 = 3e9, 5e-9, 40e-9
    delta = np.linspace(delta0, delta0 + 15e-9, 300)
    force = e_star * contact.unit_force(delta - delta0, tip) - f_ad
    force += 0.02 * f_ad * RNG.standard_normal(delta.size)          # 0.1 nN noise
    fit = contact.fit_elastic(delta, force, tip, "dmt")
    assert fit.reduced_modulus == pytest.approx(e_star, rel=0.05)
    assert fit.adhesion_force == pytest.approx(f_ad, rel=0.05)
    assert fit.work_of_adhesion == pytest.approx(f_ad / (2 * math.pi * 10e-9), rel=0.05)


def test_jkr_on_retract():
    tip = contact.Tip("sphere", radius=50e-9, poisson=0.5)
    e_star, work, delta0 = 1e8, 0.05, 30e-9
    delta = np.linspace(delta0 - 15e-9, delta0 + 30e-9, 500)
    force = contact.jkr_force(delta - delta0, e_star, work, tip.radius)
    pull_off = -force.min()
    assert pull_off == pytest.approx(1.5 * math.pi * work * tip.radius, rel=0.02)    # pull-off of JKR
    fit = contact.fit_jkr(delta, force, tip)
    assert fit.reduced_modulus == pytest.approx(e_star, rel=0.05)
    assert fit.work_of_adhesion == pytest.approx(work, rel=0.05)


@pytest.mark.parametrize("shape", ["sphere", "cone", "pyramid", "punch", "hyperboloid"])
def test_oliver_pharr_elastic_unloading(shape):
    """ A purely elastic unloading of a paraboloid / cone / punch: Oliver-Pharr recovers E* within the accuracy of
        the method (a few % for elastic contacts). """
    tip = contact.Tip(shape, radius=200e-9 if shape != "punch" else 50e-9, half_angle=30.0, poisson=0.3)
    e_star = 1e10
    h = np.linspace(0, 20e-9, 400)
    force = e_star * contact.unit_force(h, tip, exact_sphere=False)
    fit = contact.fit_oliver_pharr(h[::-1], force[::-1], tip, contact_point=0.0)
    assert fit.reduced_modulus == pytest.approx(e_star, rel=0.12)
    assert set(fit.extra) >= {"S", "hc", "A", "hf", "m"}


def test_errors():
    tip = contact.Tip("cone")
    with pytest.raises(contact.FitError):
        contact.fit_elastic([0, 1, 2], [0, 1, 2], tip)
    with pytest.raises(contact.FitError):
        contact.fit_jkr(np.linspace(0, 1e-8, 50), np.linspace(0, 1e-9, 50), tip)
    with pytest.raises(contact.FitError):
        contact.fit_elastic(np.linspace(0, 1e-8, 50), -np.linspace(0, 1e-9, 50), contact.Tip(), "hertz")
