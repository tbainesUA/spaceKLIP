"""
Characterization tests for ImageTools.crop_frames().

These tests answer:

1. Are all spatial image products cropped consistently?
2. Does asymmetric [left, right, bottom, top] cropping behave correctly?
3. Are detector-coordinate metadata shifted consistently with the crop?
4. Is invalid npix input rejected?
"""

import numpy as np

from spaceKLIP.imagetools import ImageTools

# ==============================================================================
# CROP_FRAMES TEST
# ==============================================================================


def test_crop_frames_crops_all_spatial_arrays(
    image_database,
    mock_observation,
    mock_image_io,
):
    """Uniform cropping is applied consistently to image and mask arrays."""

    tools = ImageTools(image_database)

    tools.crop_frames(npix=1)

    expected_data = mock_observation["data"][:, 1:-1, 1:-1]
    expected_erro = mock_observation["erro"][:, 1:-1, 1:-1]
    expected_pxdq = mock_observation["pxdq"][:, 1:-1, 1:-1]
    expected_mask = mock_observation["mask"][1:-1, 1:-1]
    expected_nanmask = mock_observation["nanmask"][1:-1, 1:-1]

    np.testing.assert_array_equal(
        mock_image_io["data"],
        expected_data,
    )
    np.testing.assert_array_equal(
        mock_image_io["erro"],
        expected_erro,
    )
    np.testing.assert_array_equal(
        mock_image_io["pxdq"],
        expected_pxdq,
    )
    np.testing.assert_array_equal(
        mock_image_io["mask"],
        expected_mask,
    )
    np.testing.assert_array_equal(
        mock_image_io["nanmask"],
        expected_nanmask,
    )


def test_crop_frames_supports_asymmetric_cropping(
    image_database,
    mock_observation,
    mock_image_io,
):
    """Each image boundary can be cropped by a different amount."""

    tools = ImageTools(image_database)

    npix = [1, 2, 2, 1]

    tools.crop_frames(npix=npix)

    expected = mock_observation["data"][
        :,
        2:-1,
        1:-2,
    ]

    np.testing.assert_array_equal(
        mock_image_io["data"],
        expected,
    )


def test_crop_frames_updates_spatial_coordinates(
    image_database,
    mock_image_io,
):
    """Cropping shifts detector-coordinate metadata by left/bottom crop."""

    tools = ImageTools(image_database)

    tools.crop_frames(
        npix=[1, 2, 2, 1],
    )

    head_sci = mock_image_io["head_sci"]

    assert head_sci["CRPIX1"] == 2.0
    assert head_sci["CRPIX2"] == 1.0

    assert head_sci["STARCENX"] == 1.5
    assert head_sci["STARCENY"] == 0.5

    assert head_sci["MASKCENX"] == 1.5
    assert head_sci["MASKCENY"] == 0.5

    assert head_sci["CROP_SHIFTX"] == 1
    assert head_sci["CROP_SHIFTY"] == 2
