# TEST 1: Given known inputs, did raw_contrast() construct the exact array and parameters that pyKLIP should receive?
# TEST 2: Does spaceKLIP actually construct the throughput-corrected science input correctly and keep its result
#         separate from the uncorrected result?
# TEST 3: Does raw_contrast() mask known companions from the science data before passing it to pyKLIP for contrast
#         measurement?


# New unit test
# lower-level computation
# → did compute_contrast_curves package pyKLIP's outputs correctly
#    into ContrastResult?

import numpy as np
import pytest
from astropy.table import Table

import spaceKLIP.analysis.contrast.plotting as plotting
import spaceKLIP.analysis.contrast.raw as raw
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


def test_raw_contrast_normalizes_data_before_measuring_contrast(
    tmp_path,
    monkeypatch,
):
    """
    Given known inputs, raw_contrast() should construct the expected
    normalized science image and measurement parameters before calling
    pyKLIP.
    """

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    database = FakeDatabase(tmp_path)
    tools = analysistools.AnalysisTools(database)

    data = np.full((1, 6, 6), 10.0)

    primary_header = {"CRPIX1": 3.0, "CRPIX2": 4.0, "MODE": "ADI", "ANNULI": 1}

    science_header = {}

    # Patch these at their NEW lookup locations.
    monkeypatch.setattr(
        raw.ut,
        "read_red",
        lambda filename: (data.copy(), primary_header, science_header, False),
    )

    monkeypatch.setattr(raw.ut, "read_msk", lambda filename: None)

    monkeypatch.setattr(raw.ut, "get_tp_comsubst", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        raw,
        "get_stellar_magnitudes",
        lambda *args, **kwargs: (
            {"F444W": 0.0},
            {"F444W": 100.0},
            {"F444W": 0.0},
            {"F444W": 0.0},
        ),
    )

    offset_psf = np.array([[0.1, 0.2], [0.3, 0.5]])

    monkeypatch.setattr(raw, "get_offsetpsf", lambda obs: offset_psf)

    monkeypatch.setattr(raw, "write_starfile", lambda *args, **kwargs: None)

    # Record the inputs passed to pyKLIP.
    calls = []

    def fake_meas_contrast(dat, iwa, owa, resolution, center, low_pass_filter):
        calls.append(
            {
                "dat": np.asarray(dat).copy(),
                "iwa": iwa,
                "owa": owa,
                "resolution": resolution,
                "center": center,
                "low_pass_filter": low_pass_filter,
            }
        )

        return (
            np.array([1.0, 2.0]),
            np.array([1e-4, 2e-4]),
        )

    monkeypatch.setattr(raw.klip, "meas_contrast", fake_meas_contrast)

    # Patch whatever output writer/orchestrator is called downstream,
    # only so the workflow can finish.
    monkeypatch.setattr(raw, "write_contrast_results", lambda *args, **kwargs: None)

    # PLotting
    # monkeypatch.setattr(plotting, "plot_masked_data", lambda *args, **kwargs: None)

    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    tools.raw_contrast(
        starfile="fake_star.txt", output_filetype="npy", save_figures=False
    )

    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------

    assert len(calls) == 1

    call = calls[0]

    expected_fstar = 100.0 / 1e6 * 0.5

    expected_normalized_data = (
        data[0] * database.red["concat1"]["PIXAR_SR"][0] / expected_fstar
    )

    np.testing.assert_allclose(call["dat"], expected_normalized_data)

    assert call["iwa"] == 1
    assert call["owa"] == 3

    # FITS coordinates -> Python 0-indexed coordinates.
    assert call["center"] == (2.0, 3.0)

    assert call["low_pass_filter"] is False

    pxscale_arcsec = database.red["concat1"]["PIXSCALE"][0]

    pxscale_rad = pxscale_arcsec / 3600.0 / 180.0 * np.pi

    expected_resolution = (
        1e-6
        * database.red["concat1"]["CWAVEL"][0]
        / raw.JWST_CIRCUMSCRIBED_DIAMETER
        / pxscale_rad
    )

    assert call["resolution"] == pytest.approx(expected_resolution)


# def test_compute_contrast_curves_returns_contrast_result():
#     result = compute_contrast_curves(...)

#     assert isinstance(result, ContrastResult)

#     np.testing.assert_allclose(
#         result.separations_pix,
#         expected_separations,
#     )

#     np.testing.assert_allclose(
#         result.raw_contrast,
#         expected_contrast,
#     )

#     assert result.throughput_corrected_contrast is None


def test_compute_contrast_curves_returns_expected_result(monkeypatch):
    """
    The extracted contrast computation should return a ContrastResult
    containing the separations and raw contrast returned by pyKLIP.
    """

    data = np.full((1, 6, 6), 10.0)

    expected_sep = np.array([1.0, 2.0])
    expected_contrast = np.array([1e-4, 2e-4])

    def fake_meas_contrast(
        dat,
        iwa,
        owa,
        resolution,
        center,
        low_pass_filter,
    ):
        return expected_sep, expected_contrast

    monkeypatch.setattr(
        raw.klip,
        "meas_contrast",
        fake_meas_contrast,
    )

    result = raw.compute_contrast_curves(
        data_cube=data,
        pixel_area_sr=2.0,
        stellar_flux_peak=5e-5,
        spatial_resolution_pix=2.0,
        center_pix=(2.0, 3.0),
        inner_working_angle_pix=1,
        outer_working_angle_pix=3,
        coronagraph_transmission_mask=None,
    )

    assert isinstance(result, raw.ContrastResult)

    np.testing.assert_allclose(
        result.separations_pix,
        expected_sep,
    )

    np.testing.assert_allclose(
        result.raw_contrast,
        expected_contrast[np.newaxis, :],
    )

    assert result.throughput_corrected_contrast is None
    assert result.has_throughput_correction is False


def test_raw_contrast_applies_coronagraph_throughput_correction(
    tmp_path,
    monkeypatch,
):
    """
    Characterize the coronagraph throughput-correction behavior.

    When a mask is available, raw_contrast() should measure contrast from both

        data * pixel_area_sr / stellar_flux_peak

    and

        (data / mask) * pixel_area_sr / stellar_flux_peak

    and save the two results separately.
    """

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    database = FakeDatabase(tmp_path)

    # Use a coronagraphic exposure so this test represents the intended
    # science pathway.
    database.red["concat1"]["EXP_TYPE"][0] = "NRC_CORON"
    database.red["concat1"]["CORONMSK"][0] = "MASK335R"

    tools = analysistools.AnalysisTools(database)

    data = np.full((1, 6, 6), 10.0)

    primary_header = {"CRPIX1": 3.0, "CRPIX2": 4.0, "MODE": "ADI", "ANNULI": 1}

    science_header = {}

    monkeypatch.setattr(
        analysistools.ut,
        "read_red",
        lambda filename: (data.copy(), primary_header, science_header, False),
    )

    # A constant throughput of 0.5 makes the expected corrected image simple:
    #
    # data / 0.5 == 2 * data
    mask = np.full((6, 6), 0.5)

    monkeypatch.setattr(analysistools.ut, "read_msk", lambda filename: mask)

    monkeypatch.setattr(
        analysistools.ut, "get_tp_comsubst", lambda *args, **kwargs: None
    )

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

    offset_psf = np.array([[0.1, 0.2], [0.3, 0.5]])

    monkeypatch.setattr(analysistools, "get_offsetpsf", lambda obs: offset_psf)

    monkeypatch.setattr(analysistools, "write_starfile", lambda *args, **kwargs: None)

    monkeypatch.setattr(analysistools, "load_plt_style", lambda *args, **kwargs: None)

    monkeypatch.setattr(analysistools.plt, "show", lambda: None)

    calls = []

    def fake_meas_contrast(dat, iwa, owa, resolution, center, low_pass_filter):
        calls.append(np.asarray(dat).copy())

        sep = np.array([1.0, 2.0])

        # Return a different contrast curve for each call so we can verify
        # that raw and corrected results are persisted separately.
        if len(calls) == 1:
            contrast = np.array([1e-4, 2e-4])
        else:
            contrast = np.array([5e-5, 1e-4])

        return sep, contrast

    monkeypatch.setattr(analysistools.klip, "meas_contrast", fake_meas_contrast)

    saved_arrays = {}

    def fake_save(filename, array):
        saved_arrays[str(filename)] = np.asarray(array).copy()

    monkeypatch.setattr(analysistools.np, "save", fake_save)

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    tools.raw_contrast(
        starfile="fake_star.txt", output_filetype="npy", save_figures=False
    )

    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------

    # One KL mode:
    #
    # call 1 -> raw contrast
    # call 2 -> throughput-corrected contrast
    assert len(calls) == 2

    expected_fstar = 100.0 / 1e6 * 0.5

    expected_raw = data[0] * database.red["concat1"]["PIXAR_SR"][0] / expected_fstar

    expected_corrected = (
        np.true_divide(data[0], mask)
        * database.red["concat1"]["PIXAR_SR"][0]
        / expected_fstar
    )

    np.testing.assert_allclose(calls[0], expected_raw)

    np.testing.assert_allclose(calls[1], expected_corrected)

    # Since the mask is uniformly 0.5, the corrected image should be
    # exactly twice the raw image.
    np.testing.assert_allclose(calls[1], 2.0 * calls[0])

    raw_output = next(
        value
        for filename, value in saved_arrays.items()
        if filename.endswith("_cons.npy")
    )

    corrected_output = next(
        value
        for filename, value in saved_arrays.items()
        if filename.endswith("_cons_mask.npy")
    )

    np.testing.assert_allclose(raw_output[0], np.array([1e-4, 2e-4]))

    np.testing.assert_allclose(corrected_output[0], np.array([5e-5, 1e-4]))


def test_raw_contrast_masks_companion_before_measuring_contrast(
    tmp_path,
    monkeypatch,
):
    """
    Characterize companion masking in raw_contrast().

    A supplied companion should be masked with NaNs before the image is passed
    to klip.meas_contrast(). The companion mask radius is specified in lambda/D
    and converted to pixels using the calculated spatial resolution.
    """

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    database = FakeDatabase(tmp_path)
    tools = analysistools.AnalysisTools(database)

    # Use a larger image so that the companion mask is easy to inspect.
    data = np.ones((1, 11, 11), dtype=float)

    primary_header = {
        "CRPIX1": 6.0,
        "CRPIX2": 6.0,
        "MODE": "ADI",
        "ANNULI": 1,
    }

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

    # No coronagraph transmission correction in this test.
    monkeypatch.setattr(
        analysistools.ut,
        "read_msk",
        lambda filename: None,
    )

    monkeypatch.setattr(
        analysistools.ut,
        "get_tp_comsubst",
        lambda *args, **kwargs: None,
    )

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

    monkeypatch.setattr(
        analysistools,
        "get_offsetpsf",
        lambda obs: np.array(
            [
                [0.1, 0.2],
                [0.3, 0.5],
            ]
        ),
    )

    monkeypatch.setattr(
        analysistools,
        "write_starfile",
        lambda *args, **kwargs: None,
    )

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

    # We only care about the image that reaches pyKLIP.
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
                "dat": np.asarray(dat).copy(),
                "resolution": resolution,
                "center": center,
            }
        )

        return (
            np.array([1.0, 2.0]),
            np.array([1e-4, 2e-4]),
        )

    monkeypatch.setattr(
        analysistools.klip,
        "meas_contrast",
        fake_meas_contrast,
    )

    # Prevent output writing from becoming part of this test.
    monkeypatch.setattr(
        analysistools.np,
        "save",
        lambda *args, **kwargs: None,
    )

    # Companion located directly on the stellar center.
    #
    # [RA offset, Dec offset, mask radius in lambda/D]
    companions = [[0.0, 0.0, 1.0]]

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    tools.raw_contrast(
        starfile="fake_star.txt",
        companions=companions,
        output_filetype="npy",
        save_figures=False,
    )

    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------

    assert len(calls) == 1

    measured_data = calls[0]["dat"]
    resolution = calls[0]["resolution"]
    center = calls[0]["center"]

    assert center == (5.0, 5.0)

    # Reconstruct the expected companion mask using the current baseline
    # convention implemented by raw_contrast().
    yy, xx = np.indices(data.shape[1:])

    ra = 0.0
    dec = 0.0
    radius_lambda_over_d = 1.0

    pixel_scale = database.red["concat1"]["PIXSCALE"][0]

    rr = np.sqrt(
        (xx - center[0] + ra / pixel_scale) ** 2
        + (yy - center[1] - dec / pixel_scale) ** 2
    )

    expected_mask = rr <= radius_lambda_over_d * resolution

    # Every pixel within the expected companion region must already be NaN
    # when pyKLIP receives the image.
    assert np.all(np.isnan(measured_data[expected_mask]))

    # Pixels outside the companion region should remain finite.
    assert np.all(np.isfinite(measured_data[~expected_mask]))
