"""
Characterization tests for ImageTools.crop_frames().

These tests answer:

1. Are all spatial image products cropped consistently?
2. Does asymmetric [left, right, bottom, top] cropping behave correctly?
3. Are detector-coordinate metadata shifted consistently with the crop?
4. Is invalid npix input rejected?


Known behaviors / follow-up issues
----------------------------------
- crop_frames(npix=0) produces empty spatial dimensions because
  Python interprets -0 as 0 in the slicing stop index.
- Filtering out an observation with `types` may reference
  crop_shiftx/crop_shifty before assignment.
"""

import numpy as np
import pytest

from spaceKLIP.imagetools import ImageTools

# ==============================================================================
# CROP_FRAMES TESTS
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

    np.testing.assert_array_equal(mock_image_io["data"], expected_data)
    np.testing.assert_array_equal(mock_image_io["erro"], expected_erro)
    np.testing.assert_array_equal(mock_image_io["pxdq"], expected_pxdq)
    np.testing.assert_array_equal(mock_image_io["mask"], expected_mask)
    np.testing.assert_array_equal(mock_image_io["nanmask"], expected_nanmask)


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


def test_crop_frames_rejects_invalid_crop_length(
    image_database,
):
    """npix sequences must specify left, right, bottom, and top."""

    tools = ImageTools(image_database)

    with pytest.raises(
        UserWarning,
        match="Parameter npix must either be an int or a list of four int",
    ):
        tools.crop_frames(npix=[1, 2])


# ==============================================================================
# PAD_FRAMES TESTS
# ==============================================================================


def test_pad_frames_pads_all_spatial_arrays(
    image_database, mock_observation, mock_image_io
):
    """Uniform padding is applied consistently to image and mask arrays."""

    tools = ImageTools(image_database)

    tools.pad_frames(npix=1)

    expected_data = np.pad(
        mock_observation["data"],
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan,
    )

    expected_erro = np.pad(
        mock_observation["erro"],
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan,
    )

    expected_pxdq = np.pad(
        mock_observation["pxdq"],
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )

    np.testing.assert_array_equal(mock_image_io["data"], expected_data)
    np.testing.assert_array_equal(mock_image_io["erro"], expected_erro)
    np.testing.assert_array_equal(mock_image_io["pxdq"], expected_pxdq)


def test_pad_frames_supports_asymmetric_padding(
    image_database,
    mock_observation,
    mock_image_io,
):
    """Each image boundary can be padded by a different amount."""

    tools = ImageTools(image_database)

    tools.pad_frames(npix=[1, 2, 3, 4])

    expected = np.pad(
        mock_observation["data"],
        (
            (0, 0),
            (3, 4),
            (1, 2),
        ),
        mode="constant",
        constant_values=np.nan,
    )

    np.testing.assert_array_equal(mock_image_io["data"], expected)


def test_pad_frames_updates_spatial_coordinates(
    image_database,
    mock_image_io,
):
    """Padding shifts detector coordinates by left/bottom padding."""

    tools = ImageTools(image_database)

    tools.pad_frames(npix=[1, 2, 2, 1])

    head_sci = mock_image_io["head_sci"]

    assert head_sci["CRPIX1"] == 4.0
    assert head_sci["CRPIX2"] == 5.0

    assert head_sci["STARCENX"] == 3.5
    assert head_sci["STARCENY"] == 4.5

    assert head_sci["MASKCENX"] == 3.5
    assert head_sci["MASKCENY"] == 4.5


def test_pad_frames_to_target_shape(image_database, mock_image_io):
    """tshape pads frames to the requested spatial dimensions."""

    tools = ImageTools(image_database)

    tools.pad_frames(tshape=[12, 16])

    assert mock_image_io["data"].shape[1:] == (12, 16)
    assert mock_image_io["erro"].shape[1:] == (12, 16)
    assert mock_image_io["pxdq"].shape[1:] == (12, 16)
