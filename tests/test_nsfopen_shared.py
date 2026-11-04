""" 2026-11-02: the processing modules are NSFopen's (NSFopen >= 2.3.0); the app only adds display helpers. """

import numpy as np

import combine
import fitting
import grains
import main
import measure
import ops
import profile_iso
import psdfit


def test_modules_are_nsfopens():
    from NSFopen.process import fitting as n_fitting, ops as n_ops, psdfit as n_psdfit
    assert ops is n_ops and psdfit is n_psdfit and fitting is n_fitting
    from NSFopen.process import measure as n_measure
    assert measure.Profile is n_measure.Profile and measure.ops is ops


def test_app_helpers_on_nsfopen_results():
    assert hasattr(measure, "line_measurement") and hasattr(grains, "format_value")
    assert hasattr(profile_iso, "table") and hasattr(combine, "default_name")
    assert combine.default_name("Friction force (Forward)", "Friction force (Backward)", "half_diff") == \
        "Friction force ((F - B) / 2)"


def test_registry_and_processing():
    data = np.add.outer(np.linspace(0, 1e-8, 32), np.linspace(0, 2e-8, 32))
    state = ops.State(data, np.zeros(data.shape, bool), 1e-6, 1e-6, "m")
    result = ops.apply_op(state, "level.plane", {})
    assert np.ptp(result.data) < 1e-20


def test_version_check():
    assert main.check_nsfopen() == ""


def test_spectroscopy_and_colormaps_are_nsfopens():
    """ Since NSFopen 2.4 the spectroscopy, contact model, I-V and color map code is NSFopen's own; the app's modules
    of the same names are those modules. """
    import colormaps
    import contact
    import iv
    import spectroscopy
    from NSFopen.process import colormaps as n_colormaps, contact as n_contact, iv as n_iv, spectroscopy as n_spec
    assert contact is n_contact and spectroscopy is n_spec and iv is n_iv
    assert colormaps._shared is n_colormaps
    assert iv.spectroscopy is spectroscopy and spectroscopy.contact is contact
