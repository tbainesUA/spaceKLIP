import numpy as np
import pytest
from astropy.table import Table

import spaceKLIP.analysistools as analysistools


class FakeDatabase:
    """Minimal database needed by AnalysisTools.raw_contrast()."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)

        self.red = {
            "concat1": Table(
                {
                    "FITSFILE": ["fake_red.fits"],
                    "MASKFILE": [None],
                    "INSTRUME": ["NIRCAM"],
                    "DETECTOR": ["NRCA5"],
                    "EXP_TYPE": ["NRC_IMAGE"],
                    "SUBARRAY": ["FULL"],
                    "FILTER": ["F444W"],
                    "PIXSCALE": [0.1],
                    "PIXAR_SR": [2.0],
                    "TELESCOP": ["JWST"],
                    "CWAVEL": [4.0],
                    "CORONMSK": ["NONE"],
                    "KLMODES": ["1"],
                    "BUNIT": ["MJy/sr"],
                }
            )
        }

        self.obs = {
            "concat1": Table(
                {
                    "BLURFWHM": [np.nan],
                    "TYPE": ["SCI"],
                    "ROLL_REF": [0.0],
                }
            )
        }


def test_raw_contrast_normalizes_data_before_measuring_contrast(tmp_path, monkeypatch):
    """
    Characterize the numerical normalization performed by raw_contrast().

    The image passed to klip.meas_contrast() should be

        data * pixel_area_sr / stellar_flux_peak

    where

        stellar_flux_peak =
            fzero / 10**(mstar / 2.5) / 1e6 * max(offset_psf).
    """

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    database = FakeDatabase(tmp_path)
    tools = analysistools.AnalysisTools(database)

    # One KL mode and a deliberately small image.
    data = np.full((1, 6, 6), 10.0)

    primary_header = {"CRPIX1": 3.0, "CRPIX2": 4.0, "MODE": "ADI", "ANNULI": 1}

    science_header = {}

    monkeypatch.setattr(
        analysistools.ut,
        "read_red",
        lambda filename: (
            data.copy(),
            primary_header,
            science_header,
            False,
        ),
    )

    # No coronagraph throughput correction for this test.
    monkeypatch.setattr(analysistools.ut, "read_msk", lambda filename: None)

    # This result is currently unused by raw_contrast(), but the function
    # is still called and therefore must be stubbed.
    monkeypatch.setattr(
        analysistools.ut, "get_tp_comsubst", lambda *args, **kwargs: None
    )

    # Choose values that make the expected stellar normalization obvious.
    #
    # mstar = 0 mag
    # fzero = 100 Jy
    monkeypatch.setattr(
        analysistools,
        "get_stellar_magnitudes",
        lambda *args, **kwargs: (
            {"F444W": 0.0},
            {"F444W": 100.0},
            {"F444W": 0.0},
            {"F444W": 0.0},
        ),
    )

    # Integrated PSF is irrelevant here. raw_contrast() only uses max().
    offset_psf = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.5],
        ]
    )

    monkeypatch.setattr(
        analysistools,
        "get_offsetpsf",
        lambda obs: offset_psf,
    )

    # Avoid testing file-copy behavior in this test.
    monkeypatch.setattr(
        analysistools,
        "write_starfile",
        lambda *args, **kwargs: None,
    )

    # Record exactly what raw_contrast() gives to pyKLIP.
    calls = []

    def fake_meas_contrast(
        dat,
        iwa,
        owa,
        resolution,
        center,
        low_pass_filter,
    ):
        calls.append(
            {
                "dat": dat.copy(),
                "iwa": iwa,
                "owa": owa,
                "resolution": resolution,
                "center": center,
                "low_pass_filter": low_pass_filter,
            }
        )

        # Return deterministic fake pyKLIP results.
        sep = np.array([1.0, 2.0])
        contrast = np.array([1e-4, 2e-4])

        return sep, contrast

    monkeypatch.setattr(
        analysistools.klip,
        "meas_contrast",
        fake_meas_contrast,
    )

    # Plotting is not part of this characterization test.
    monkeypatch.setattr(
        analysistools,
        "load_plt_style",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        analysistools.plt,
        "show",
        lambda: None,
    )

    # Avoid touching disk for the .npy outputs while allowing the method
    # to complete normally.
    saved_arrays = {}

    def fake_save(filename, array):
        saved_arrays[str(filename)] = np.asarray(array).copy()

    monkeypatch.setattr(
        analysistools.np,
        "save",
        fake_save,
    )

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    tools.raw_contrast(
        starfile="fake_star.txt",
        output_filetype="npy",
        save_figures=False,
    )

    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------

    assert len(calls) == 1

    call = calls[0]

    # Stellar peak normalization:
    #
    # fstar = 100 Jy / 10**(0 / 2.5) / 1e6 * 0.5
    #       = 5e-5 MJy
    expected_fstar = 100.0 / 1e6 * 0.5

    # raw_contrast passes:
    #
    # data * PIXAR_SR / fstar
    expected_normalized_data = (
        data[0] * database.red["concat1"]["PIXAR_SR"][0] / expected_fstar
    )

    np.testing.assert_allclose(call["dat"], expected_normalized_data)

    # Hard-coded current behavior.
    assert call["iwa"] == 1

    # data.shape[1] // 2 = 6 // 2 = 3
    assert call["owa"] == 3

    # FITS CRPIX coordinates are converted from 1-indexed to 0-indexed.
    assert call["center"] == (2.0, 3.0)

    assert call["low_pass_filter"] is False

    # Current JWST non-NRC_CORON path uses the circumscibed diameter.
    pxscale_arcsec = database.red["concat1"]["PIXSCALE"][0]
    pxscale_rad = pxscale_arcsec / 3600.0 / 180.0 * np.pi

    expected_resolution = (
        1e-6
        * database.red["concat1"]["CWAVEL"][0]
        / analysistools.JWST_CIRCUMSCRIBED_DIAMETER
        / pxscale_rad
    )

    assert call["resolution"] == pytest.approx(expected_resolution)

    # pyKLIP returns separations in pixels. raw_contrast() should save them
    # in arcseconds by multiplying by PIXSCALE.
    expected_separation_arcsec = (
        np.array([1.0, 2.0]) * database.red["concat1"]["PIXSCALE"][0]
    )

    seps_output = next(
        value
        for filename, value in saved_arrays.items()
        if filename.endswith("_seps.npy")
    )

    np.testing.assert_allclose(seps_output[0], expected_separation_arcsec)
