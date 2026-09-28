"""
Characterization tests for ImageTools.coadd_frames().

Current implementation note
---------------------------
For nframes < NINTS, the reshape operation groups integrations along
axis 0 after reshape((nframes, ncoadds, ny, nx)). For example, four
frames coadded with nframes=2 combine (0, 2) and (1, 3), rather than
consecutive pairs (0, 1) and (2, 3).

Tests below characterize the existing behavior. The intended scientific
grouping should be reviewed separately before changing the implementation.


verifies the scientific combination rules: science data use the median,
uncertainties are propagated in quadrature, DQ flags are combined bitwise,
and NINTS/EFFINTTM are updated consistently. It also characterizes the existing
frame-grouping behavior.

"""

import numpy as np

from spaceKLIP.imagetools import ImageTools


def test_coadd_frames_combines_science_data(
    image_database, mock_observation, mock_image_io
):
    """Science frames are coadded using the current median grouping."""

    tools = ImageTools(image_database)

    tools.coadd_frames(nframes=2)

    data = mock_observation["data"]

    expected = np.nanmedian(data.reshape(2, 2, data.shape[-2], data.shape[-1]), axis=0)

    np.testing.assert_array_equal(mock_image_io["data"], expected)

    assert mock_image_io["data"].shape[0] == 2


def test_coadd_frames_propagates_uncertainties(image_database, mock_image_io):
    """Coadded uncertainties use quadrature propagation."""

    tools = ImageTools(image_database)

    tools.coadd_frames(nframes=2)

    expected_error = np.sqrt(2) / 2

    np.testing.assert_allclose(mock_image_io["erro"], expected_error)


def test_coadd_frames_combines_dq_with_bitwise_or(
    image_database, mock_observation, mock_image_io
):
    """DQ flags from contributing frames are combined with bitwise OR."""

    mock_observation["pxdq"][0, 0, 0] = 1
    mock_observation["pxdq"][2, 0, 0] = 4

    tools = ImageTools(image_database)

    tools.coadd_frames(nframes=2)

    assert mock_image_io["pxdq"][0, 0, 0] == 5


def test_coadd_frames_updates_integration_metadata(image_database, mock_image_io):
    """Coadding updates integration count and effective integration time."""

    tools = ImageTools(image_database)

    tools.coadd_frames(nframes=2)

    assert mock_image_io["head_pri"]["NINTS"] == 2
    assert mock_image_io["head_pri"]["EFFINTTM"] == 20.0

    assert image_database.obs["concat1"]["NINTS"][0] == 2
    assert image_database.obs["concat1"]["EFFINTTM"][0] == 20.0


def test_coadd_frames_none_coadds_all_integrations(
    image_database, mock_observation, mock_image_io
):
    """nframes=None collapses all integrations into one."""

    tools = ImageTools(image_database)

    tools.coadd_frames(nframes=None)

    expected = np.nanmedian(mock_observation["data"], axis=0, keepdims=True)

    np.testing.assert_array_equal(mock_image_io["data"], expected)

    assert mock_image_io["data"].shape[0] == 1
    assert mock_image_io["head_pri"]["NINTS"] == 1
    assert mock_image_io["head_pri"]["EFFINTTM"] == 40.0
