""" Spectroscopy: the Maugis-Dugdale model and the bottom-effect correction of thin samples in the panel. """

import math
import pathlib

import numpy as np

from test_files_and_gui import app, window  # noqa: F401 (fixtures)


def _open(window):  # noqa: F811
    import NSFopen
    path = pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid"
    window.open_file(path)
    window.set_advanced(True)
    return window._spectroscopy


def test_maugis_dugdale_in_the_panel(window):  # noqa: F811
    import spectroscopy as sp
    page = _open(window)
    page._model.set_value("maugis")
    assert page._shape.keys() == ["sphere"]
    assert not page._thickness_mode.isEnabled()                    # the bottom effect is for Hertz, Sneddon, DMT
    settings = page.force_settings()
    assert settings.model == "maugis"
    curve = sp.ForceAnalysis(page.spec, settings).curve(0)
    assert not curve.error, curve.error
    values = curve.values
    assert values["E"][0] > 0 and values["work of adhesion"][0] > 0 and values["Maugis λ"][0] > 0
    assert values["Tabor μ"][0] == np.float64(1.157 * values["Maugis λ"][0])
    assert "Maugis-Dugdale" in sp.describe_model(sp.model_settings(settings))


def test_thin_sample_in_the_panel(window):  # noqa: F811
    import spectroscopy as sp
    page = _open(window)
    page._model.set_value("hertz")
    page._thickness_mode.set_value("fixed")
    assert page._thickness.isEnabled() and page._bonded.isEnabled() and not page._substrate.isEnabled()
    page._thickness.set_value(30e-9)
    thick = sp.ForceAnalysis(page.spec, sp.ForceSettings(**{**page.force_settings().__dict__,
                                                             "thickness_mode": "off"})).curve(0)
    thin = sp.ForceAnalysis(page.spec, page.force_settings()).curve(0)
    assert thin.values["E"][0] < thick.values["E"][0]          # the substrate made the plain fit read too stiff
    assert thin.values["sample thickness"][0] == 30e-9 and thin.values["bottom factor"][0] > 1
    assert "thin 30 nm, bonded" in sp.describe_model(sp.model_settings(page.force_settings()))
    # from the height: the substrate from the masked curves, every curve its own thickness
    page._thickness_mode.set_value("map")
    assert page._substrate.isEnabled() and page._substrate_button.isEnabled() and not page._thickness.isEnabled()
    assert not page.substrate_from_mask()                       # no mask yet
    mask = np.zeros(page.spec.n_curves, bool)
    mask[:10] = True
    page.maps.mask = mask
    assert page.substrate_from_mask()
    analysis = sp.ForceAnalysis(page.spec, page.force_settings())
    heights = [-analysis.curve(n, fit=False).contact_x for n in range(10)]
    assert page._substrate.value() == np.float64(np.median(heights))
    page._substrate.set_value(page._substrate.value() - 50e-9)
    curve = sp.ForceAnalysis(page.spec, page.force_settings()).curve(5)
    assert curve.values["sample thickness"][0] == np.float64(
        -curve.contact_x - page._substrate.value())
    # a model the correction is not for: an error per curve
    page._model.set_value("jkr")
    assert not page._thickness_mode.isEnabled()
    bad = sp.ForceAnalysis(page.spec, page.force_settings()).curve(0)
    assert "bottom-effect" in bad.error
    page._model.set_value("hertz")
    page._thickness_mode.set_value("off")


def test_old_model_region_file_still_opens(window, tmp_path):  # noqa: F811
    import json
    import spectroscopy as sp
    page = _open(window)
    old = {k: v for k, v in sp.model_settings(sp.ForceSettings()).items()
           if k not in ("thickness_mode", "thickness", "substrate", "bonded")}
    path = tmp_path / "old_fit_regions.json"
    path.write_text(json.dumps({"format": "nanosurfimg-fit-regions", "version": 1, "curves": page.spec.n_curves,
                                "all curves": None, "regions": [{"model": old, "curves": [1, 2, 3]}]}))
    assert page.load_regions(path)
    assert page.regions[0]["model"]["thickness_mode"] == "off"
    assert math.isinf(sp.ForceSettings().tip().thickness)
