"""
Characterization tests for ImageTools background-subtraction methods.

These tests verify that background estimators recover and subtract known
background levels from synthetic observations while preserving astrophysical
signal and associated observation products.

subtract_median()
-----------------
The tests answer:

1. Is a background level estimated independently for each integration?
2. Are DO_NOT_USE pixels excluded from background estimation?
3. Does border estimation use the source-free outer image region?
4. Does robust estimation reject bright source contamination?
5. Does sigma-clipped estimation behave consistently with its documented
   parameters?
"""

import numpy as np
import pytest
from astropy.table import Table

from spaceKLIP.imagetools import ImageTools

# ==============================================================================
# MEDIAN BACKGROUND SUBTRACTION TESTS
# ==============================================================================


def test_subtract_median_simple_per_integration(
    image_database, mock_observation, mock_image_io
):
    """Simple median subtraction estimates each integration independently."""

    mock_observation["data"] = np.array(
        [
            [
                [10.0, 10.0, 10.0],
                [10.0, 50.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            [
                [20.0, 20.0, 20.0],
                [20.0, 80.0, 20.0],
                [20.0, 20.0, 20.0],
            ],
        ]
    )

    mock_observation["erro"] = np.ones_like(mock_observation["data"])
    mock_observation["pxdq"] = np.zeros(mock_observation["data"].shape, dtype=np.uint32)

    tools = ImageTools(image_database)

    tools.subtract_median(method="simple")

    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 40.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 60.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )

    np.testing.assert_array_equal(mock_image_io["data"], expected)


def test_subtract_median_excludes_do_not_use_pixels(
    image_database, mock_observation, mock_image_io
):
    """DO_NOT_USE pixels are excluded from simple background estimation."""

    mock_observation["data"] = np.array(
        [
            [
                [10.0, 10.0, 10.0],
                [10.0, 10.0, 999.0],
                [10.0, 20.0, 30.0],
            ]
        ]
    )

    mock_observation["erro"] = np.ones_like(mock_observation["data"])

    mock_observation["pxdq"] = np.zeros(mock_observation["data"].shape, dtype=np.uint32)

    # Mark the extreme-valued pixel as DO_NOT_USE.
    mock_observation["pxdq"][0, 1, 2] = 1

    tools = ImageTools(image_database)

    tools.subtract_median(method="simple")

    # Valid pixels have a median background of 10.
    assert mock_image_io["data"][0, 0, 0] == 0.0

    # The original science pixel is still present in DATA; DQ masking is
    # used only for estimating the background.
    assert mock_image_io["data"][0, 1, 2] == 989.0


def test_subtract_median_border_uses_outer_region(
    image_database, mock_observation, mock_image_io
):
    """Border subtraction estimates background from the outer image region."""

    data = np.full((1, 9, 9), 10.0)

    # Simulate a bright, spatially extended central source.
    data[0, 2:7, 2:7] = 100.0

    mock_observation["data"] = data
    mock_observation["erro"] = np.ones_like(data)
    mock_observation["pxdq"] = np.zeros(data.shape, dtype=np.uint32)

    tools = ImageTools(image_database)

    tools.subtract_median(method="border", borderwidth=1)

    result = mock_image_io["data"]

    # Known background should be removed.
    np.testing.assert_allclose(result[0, 0, 0], 0.0)

    # Source excess should remain.
    np.testing.assert_allclose(result[0, 4, 4], 90.0)


def test_subtract_median_robust_rejects_bright_source(
    image_database, mock_observation, mock_image_io
):
    """Robust subtraction rejects bright source contamination."""

    background = np.array(
        [
            [9.0, 10.0, 11.0, 10.0, 9.0],
            [10.0, 11.0, 9.0, 10.0, 11.0],
            [11.0, 9.0, 10.0, 9.0, 10.0],
            [9.0, 10.0, 11.0, 10.0, 9.0],
            [10.0, 11.0, 9.0, 10.0, 11.0],
        ]
    )

    data = background[None, ...].copy()

    # Inject a bright source.
    data[0, 2, 2] = 1000.0

    mock_observation["data"] = data
    mock_observation["erro"] = np.ones_like(data)
    mock_observation["pxdq"] = np.zeros(data.shape, dtype=np.uint32)

    tools = ImageTools(image_database)

    tools.subtract_median(method="robust")

    result = mock_image_io["data"]

    # The known median background is 10.
    np.testing.assert_allclose(
        np.nanmedian(result[0][background == 10.0]),
        0.0,
    )

    # The bright source should retain its excess above the background.
    np.testing.assert_allclose(result[0, 2, 2], 990.0)


# ==============================================================================
# GODOY BACKGROUND SUBTRACTION TESTS
# ==============================================================================


def test_subtract_background_godoy_uses_science_background(
    godoy_database, godoy_data, make_mock_observation, mock_image_io, monkeypatch
):
    """
    Godoy subtraction uses the corresponding SCI_BG observation
    when processing SCI data.
    """

    science_file = godoy_database.obs["concat1"]["FITSFILE"][0]
    background_file = godoy_database.obs["concat1"]["FITSFILE"][1]

    mock_image_io["observations"][science_file] = make_mock_observation(
        godoy_data["science"]
    )

    mock_image_io["observations"][background_file] = make_mock_observation(
        godoy_data["background"]
    )

    class Result:
        x = np.array([0.0])

    def mock_minimize(*args, **kwargs):
        return Result()

    monkeypatch.setattr("spaceKLIP.imagetools.minimize", mock_minimize)

    tools = ImageTools(godoy_database)

    tools.subtract_background_godoy(types=["SCI"])

    result = mock_image_io["data"]

    # Reproduce only the deterministic transformation surrounding the
    # mocked optimization. With optimizer scale=0, the final product is
    # the initially background-subtracted frame with its median removed.
    initial_subtraction = godoy_data["science"] - godoy_data["background"]

    expected = np.empty_like(initial_subtraction)

    for k in range(initial_subtraction.shape[0]):
        expected[k] = initial_subtraction[k] - np.nanmedian(initial_subtraction[k])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_subtract_background_godoy_median_combines_backgrounds(
    godoy_database,
    godoy_data,
    make_mock_observation,
    mock_image_io,
    monkeypatch,
    tmp_path,
):
    """Multiple SCI_BG observations are median-combined before subtraction."""

    table = godoy_database.obs["concat1"]

    science_file = table["FITSFILE"][0]

    bg_files = [
        str(tmp_path / "background_1.fits"),
        str(tmp_path / "background_2.fits"),
        str(tmp_path / "background_3.fits"),
    ]

    # Replace the original table with one SCI and three SCI_BG products.
    godoy_database.obs["concat1"] = Table(
        {
            "FITSFILE": [
                science_file,
                *bg_files,
            ],
            "TYPE": [
                "SCI",
                "SCI_BG",
                "SCI_BG",
                "SCI_BG",
            ],
            "FILTER": [
                "F1140C",
                "F1140C",
                "F1140C",
                "F1140C",
            ],
        }
    )

    background = godoy_data["background"]

    # Three distinct backgrounds whose pixel-wise median is exactly
    # the original synthetic background.
    background_low = background - 2.0
    background_mid = background
    background_high = background + 4.0

    mock_image_io["observations"][science_file] = make_mock_observation(
        godoy_data["science"]
    )

    for filename, data in zip(
        bg_files,
        [
            background_low,
            background_mid,
            background_high,
        ],
    ):
        mock_image_io["observations"][filename] = make_mock_observation(data)

    class Result:
        x = np.array([0.0])

    def mock_minimize(*args, **kwargs):
        return Result()

    monkeypatch.setattr(
        "spaceKLIP.imagetools.minimize",
        mock_minimize,
    )

    tools = ImageTools(godoy_database)

    tools.subtract_background_godoy(
        types=["SCI"],
    )

    # Median(-2, 0, +4) relative to the base background is 0,
    # so the combined background must equal godoy_data["background"].
    initial_subtraction = godoy_data["science"] - godoy_data["background"]

    expected = np.empty_like(initial_subtraction)

    for k in range(initial_subtraction.shape[0]):
        expected[k] = initial_subtraction[k] - np.nanmedian(initial_subtraction[k])

    np.testing.assert_allclose(mock_image_io["data"], expected)


def test_subtract_background_godoy_rejects_unsupported_filter(
    godoy_database, godoy_data, make_mock_observation, mock_image_io
):
    """Godoy subtraction rejects filters outside the supported MIRI FQPMs."""

    godoy_database.obs["concat1"]["FILTER"][0] = "F2300C"

    science_file = godoy_database.obs["concat1"]["FITSFILE"][0]
    background_file = godoy_database.obs["concat1"]["FITSFILE"][1]

    mock_image_io["observations"][science_file] = make_mock_observation(
        godoy_data["science"]
    )

    mock_image_io["observations"][background_file] = make_mock_observation(
        godoy_data["background"]
    )

    tools = ImageTools(godoy_database)

    with pytest.raises(
        NotImplementedError,
        match="Godoy subtraction is only supported for MIRI FQPMs",
    ):
        tools.subtract_background_godoy(types=["SCI"])


def test_subtract_background_godoy_requires_background(
    godoy_database, godoy_data, make_mock_observation, mock_image_io
):
    """Godoy subtraction fails when no corresponding SCI_BG exists."""

    science_file = godoy_database.obs["concat1"]["FITSFILE"][0]

    godoy_database.obs["concat1"] = Table(
        {
            "FITSFILE": [science_file],
            "TYPE": ["SCI"],
            "FILTER": ["F1140C"],
        }
    )

    mock_image_io["observations"][science_file] = make_mock_observation(
        godoy_data["science"]
    )

    tools = ImageTools(godoy_database)

    with pytest.raises(UserWarning, match="Could not find any background files"):
        tools.subtract_background_godoy(types=["SCI"])


# ==============================================================================
# SUBTRACT BACKGROUND TESTS
# ==============================================================================


def test_subtract_background_uses_median_background(
    background_database,
    make_mock_observation,
    mock_image_io,
):
    """
    Verify that the median background is subtracted from the science data.

    A background observation can contain several integrations with slightly
    different measured background levels. When ``nints_per_med=None``, the
    method should combine all of those integrations into one median background
    image and subtract that image from every science integration.

    This test creates science integrations with a value of 20 and three
    background integrations with values of 8, 10, and 12. Their median is 10,
    so subtracting the background should leave a science value of 10.

    In simple terms:
        We give the method three measurements of the background: 8, 10, and
        12. The middle value is 10. We then check that it subtracts 10 from
        the science picture rather than choosing one of the other background
        measurements.
    """

    science_file = background_database.obs["concat1"]["FITSFILE"][0]
    background_file = background_database.obs["concat1"]["FITSFILE"][1]

    science = np.full((3, 3, 3), 20.0)

    background = np.stack(
        [
            np.full((3, 3), 8.0),
            np.full((3, 3), 10.0),
            np.full((3, 3), 12.0),
        ]
    )

    mock_image_io["observations"][science_file] = make_mock_observation(science)

    mock_image_io["observations"][background_file] = make_mock_observation(background)

    tools = ImageTools(background_database)

    tools.subtract_background(
        nints_per_med=None,
    )

    np.testing.assert_allclose(
        mock_image_io["data"],
        10.0,
    )


def test_subtract_background_groups_integrations(
    background_database,
    make_mock_observation,
    mock_image_io,
):
    """
    Verify that nints_per_med applies different backgrounds to time groups.

    The ``nints_per_med`` option allows the background to change during an
    observation. Instead of making one background from the entire sequence,
    the method should calculate a separate median for each group of
    integrations and subtract it from the matching science group.

    This test uses four background integrations with values 8, 12, 28, and
    32 and sets ``nints_per_med=2``. The first pair has a median of 10 and the
    second pair has a median of 30. Science values of 20, 20, 40, and 40
    should therefore all become 10 after their corresponding backgrounds are
    removed.

    In simple terms:
        Imagine the background gets brighter halfway through an observation.
        Instead of using one average background for everything, we split the
        observation into two pieces. We check that the first two pictures use
        the first background and the last two pictures use the later,
        brighter background.
    """

    science_file = background_database.obs["concat1"]["FITSFILE"][0]
    background_file = background_database.obs["concat1"]["FITSFILE"][1]

    science = np.stack(
        [
            np.full((3, 3), 20.0),
            np.full((3, 3), 20.0),
            np.full((3, 3), 40.0),
            np.full((3, 3), 40.0),
        ]
    )

    background = np.stack(
        [
            np.full((3, 3), 8.0),
            np.full((3, 3), 12.0),
            np.full((3, 3), 28.0),
            np.full((3, 3), 32.0),
        ]
    )

    mock_image_io["observations"][science_file] = make_mock_observation(science)

    mock_image_io["observations"][background_file] = make_mock_observation(background)

    tools = ImageTools(background_database)

    tools.subtract_background(
        nints_per_med=2,
    )

    np.testing.assert_allclose(
        mock_image_io["data"],
        10.0,
    )


def test_subtract_background_propagates_uncertainty(
    background_database,
    make_mock_observation,
    mock_image_io,
):
    """
    Verify that background uncertainty is propagated into science uncertainty.

    Subtracting a measured background adds uncertainty to the final science
    measurement. The method first combines the uncertainties from the
    background integrations in quadrature and then combines that result in
    quadrature with the original science uncertainty.

    This test gives every science pixel an uncertainty of 2 and uses two
    background integrations, each with an uncertainty of 1. According to the
    implemented propagation rule, the combined background uncertainty is
    sqrt(2) / 2. The final expected science uncertainty can therefore be
    calculated exactly.

    In simple terms:
        The background measurement is not perfectly known. When we subtract
        it, that uncertainty does not disappear. We check that the final
        error bar becomes appropriately larger to include uncertainty from
        both the science picture and the background picture.
    """

    science_file = background_database.obs["concat1"]["FITSFILE"][0]
    background_file = background_database.obs["concat1"]["FITSFILE"][1]

    science = np.full((2, 3, 3), 20.0)
    background = np.full((2, 3, 3), 10.0)

    science_error = np.full_like(science, 2.0)

    background_error = np.full_like(background, 1.0)

    mock_image_io["observations"][science_file] = make_mock_observation(
        science,
        erro=science_error,
    )

    mock_image_io["observations"][background_file] = make_mock_observation(
        background,
        erro=background_error,
    )

    tools = ImageTools(background_database)

    tools.subtract_background(
        nints_per_med=None,
    )

    combined_background_error = np.sqrt(2.0) / 2.0

    expected_error = np.sqrt(2.0**2 + combined_background_error**2)

    np.testing.assert_allclose(mock_image_io["erro"], expected_error)


def test_subtract_background_propagates_do_not_use(
    background_database,
    make_mock_observation,
    mock_image_io,
):
    """
    Verify that bad background pixels propagate to the science DQ array.

    A science pixel should not remain marked as good when the background used
    to correct that pixel contains a DO_NOT_USE measurement. The method checks
    all background integrations contributing to each median and records
    whether any of them contain a bad pixel.

    This test marks one pixel as DO_NOT_USE in one background integration
    while leaving the corresponding science pixel unflagged. After background
    subtraction, that science pixel should acquire the DO_NOT_USE flag.

    In simple terms:
        If the background measurement for one pixel is bad, we cannot fully
        trust the corrected science pixel either. We deliberately damage one
        background pixel and check that the warning flag follows it into the
        corrected science data.
    """

    science_file = background_database.obs["concat1"]["FITSFILE"][0]
    background_file = background_database.obs["concat1"]["FITSFILE"][1]

    science = np.full((2, 3, 3), 20.0)
    background = np.full((2, 3, 3), 10.0)

    science_dq = np.zeros(science.shape, dtype=np.uint32)

    background_dq = np.zeros(background.shape, dtype=np.uint32)

    # Mark one background pixel as DO_NOT_USE.
    background_dq[0, 1, 1] = 1

    mock_image_io["observations"][science_file] = make_mock_observation(
        science,
        pxdq=science_dq,
    )

    mock_image_io["observations"][background_file] = make_mock_observation(
        background,
        pxdq=background_dq,
    )

    tools = ImageTools(background_database)

    tools.subtract_background(
        nints_per_med=None,
    )

    assert mock_image_io["pxdq"][0, 1, 1] == 1
    assert mock_image_io["pxdq"][1, 1, 1] == 1

    # An unaffected pixel should remain good.
    assert mock_image_io["pxdq"][0, 0, 0] == 0


def test_subtract_background_science_falls_back_to_reference_background(
    background_database,
    make_mock_observation,
    mock_image_io,
    tmp_path,
):
    """
    Verify that SCI data use REF_BG when no SCI_BG observation is available.

    The implementation contains a fallback for datasets that do not have the
    preferred matching background type. If a science observation has no
    SCI_BG but a REF_BG is available, the method should use that reference
    background rather than failing immediately.

    This test creates a science observation and a REF_BG observation but
    deliberately provides no SCI_BG. The reference background has a known
    value of 10 and the science data have a value of 20, so the corrected
    science data should contain 10.

    In simple terms:
        The science picture normally wants its own background picture. Here we
        deliberately take that background away but provide the reference
        star's background instead. We check that the method uses the backup
        background as it was designed to do.
    """

    science_file = str(tmp_path / "science_calints.fits")
    reference_background_file = str(tmp_path / "reference_background_calints.fits")
    science_mask = str(tmp_path / "science_mask.fits")
    background_mask = str(tmp_path / "reference_background_mask.fits")

    background_database.obs["concat1"] = Table(
        {
            "FITSFILE": [
                science_file,
                reference_background_file,
            ],
            "MASKFILE": [
                science_mask,
                background_mask,
            ],
            "TYPE": [
                "SCI",
                "REF_BG",
            ],
        }
    )

    science = np.full((2, 3, 3), 20.0)

    reference_background = np.full((2, 3, 3), 10.0)

    mock_image_io["observations"][science_file] = make_mock_observation(science)

    mock_image_io["observations"][reference_background_file] = make_mock_observation(
        reference_background
    )

    tools = ImageTools(background_database)

    tools.subtract_background(nints_per_med=None)

    np.testing.assert_allclose(mock_image_io["data"], 10.0)
