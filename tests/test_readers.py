""" 2026-10-31: the built-in .nid reader (two channels with the same name) and Nanosurf's .nhf reader. """

import builtins

import numpy as np
import pytest

import imgdata
from conftest import example

EPROM = example("EPROM_stitching", "Miniscribe_eprom_stich104659.nid")
KPFM = example("Moire_patterns", "Moire_KPFM.nid")
LFM = example("MobileS", "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid")
NHF = example("Studio_PhaseContrast", "measurement_1.1.nhf")


@pytest.mark.skipif(not KPFM.is_file(), reason="example data missing")
def test_two_channels_with_the_same_name_are_both_kept():
    image = imgdata.open_image_file(KPFM)
    assert "Amplitude (Forward)" in image.channels and "Amplitude 2 (Forward)" in image.channels
    first, second = image.channels["Amplitude (Forward)"].data, image.channels["Amplitude 2 (Forward)"].data
    assert not np.allclose(first, second)                            # really two different signals
    assert len(image.channels) == 14 and image.info["File"]["Read with"] == "built-in .nid reader"


@pytest.mark.skipif(not LFM.is_file(), reason="example data missing")
def test_nid_values_match_nsfopen():
    read = pytest.importorskip("NSFopen.read").read
    import contextlib
    import io
    image = imgdata.open_image_file(LFM)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = read(LFM, verbose=False)
    flip = image.info["Scan"]["Scan direction"].lower() == "up"
    for channel in image.channels.values():
        raw = np.asarray(reference.data[("Image", channel.direction, channel.name)], float)
        assert np.allclose(channel.data, raw[::-1] if flip else raw)


def test_broken_nid_files(tmp_path):
    no_marker = tmp_path / "a.nid"
    no_marker.write_bytes(b"[DataSet]\nGroupCount=0\n")
    with pytest.raises(imgdata.ImageFileError, match="data marker"):
        imgdata.open_image_file(no_marker)
    if LFM.is_file():
        cut = tmp_path / "cut.nid"
        cut.write_bytes(LFM.read_bytes()[:-1000])                    # data missing at the end
        image_or_error = None
        try:
            image_or_error = imgdata.open_image_file(cut)
        except imgdata.ImageFileError as e:
            image_or_error = e
        assert image_or_error is not None                            # no crash


@pytest.mark.skipif(not NHF.is_file(), reason="example data missing")
def test_nhf_with_and_without_nanosurf(monkeypatch):
    real_import = builtins.__import__

    def without_nanosurf(name, *args, **kwargs):
        if name.startswith("nanosurf"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_nanosurf)
    built_in = imgdata.open_image_file(NHF)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert built_in.info["File"]["Read with"].startswith("built-in reader")
    try:
        import nanosurf.utils.io.nhf_reader  # noqa: F401
    except Exception:
        pytest.skip("nanosurf >= 2 not installed")
    official = imgdata.open_image_file(NHF)
    assert official.info["File"]["Read with"].startswith("Nanosurf's reader")
    for key, channel in built_in.channels.items():
        if channel.unit == official.channels[key].unit:
            assert np.allclose(channel.data, official.channels[key].data, equal_nan=True)


def test_nanosurf_reader_failure_falls_back(monkeypatch):
    pytest.importorskip("nanosurf.utils.io.nhf_reader")
    from nanosurf.utils.io import nhf_reader
    monkeypatch.setattr(nhf_reader, "NHFFileReader", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    if not NHF.is_file():
        pytest.skip("example data missing")
    image = imgdata.open_image_file(NHF)
    assert "failed" in image.info["File"]["Read with"] and image.channels
