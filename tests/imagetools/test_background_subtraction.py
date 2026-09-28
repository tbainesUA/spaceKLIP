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
# MEDIAN BACKGROUND SUBTRACTION TESTS
# ==============================================================================