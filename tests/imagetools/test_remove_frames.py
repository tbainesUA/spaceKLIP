"""
Characterization tests for spaceKLIP.imagetools.ImageTools.

These tests establish the current behavior of the legacy ImageTools class
before refactoring.

remove_frames()
---------------
The tests answer the following questions:

1. Does ImageTools retain the database instance supplied at initialization?
2. Does remove_frames() remove the requested integration from all
   integration-dependent science arrays?
3. Does removing frames correctly update the FITS and database NINTS values?
4. Can multiple integrations be removed in a single call?
5. Are observations whose TYPE is not selected left unchanged?
6. Can different frame indices be specified for individual observations?
"""

import numpy as np

from spaceKLIP.imagetools import ImageTools

# ==============================================================================
# REMOVE_FRAMES TEST
# ==============================================================================


def test_imagetools_stores_database(image_database):
    """ImageTools retains the supplied database instance."""

    tools = ImageTools(image_database)

    assert tools.database is image_database


def test_remove_frames_removes_requested_integration(
    image_database, mock_observation, mock_image_io
):
    """Requested integration is removed from all primary data arrays."""

    tools = ImageTools(image_database)

    tools.remove_frames(index=1)

    expected_data = np.delete(mock_observation["data"], 1, axis=0)
    expected_erro = np.delete(mock_observation["erro"], 1, axis=0)
    expected_pxdq = np.delete(mock_observation["pxdq"], 1, axis=0)

    np.testing.assert_array_equal(mock_image_io["data"], expected_data)
    np.testing.assert_array_equal(mock_image_io["erro"], expected_erro)
    np.testing.assert_array_equal(mock_image_io["pxdq"], expected_pxdq)


def test_remove_frames_updates_nints(image_database, mock_image_io):
    """Frame removal updates both written and database NINTS metadata."""

    tools = ImageTools(image_database)

    tools.remove_frames(index=1)

    assert mock_image_io["head_pri"]["NINTS"] == 3
    assert image_database.obs["concat1"]["NINTS"][0] == 3


def test_remove_frames_removes_multiple_integrations(
    image_database,
    mock_observation,
    mock_image_io,
):
    """Multiple requested integrations are removed in one operation."""

    tools = ImageTools(image_database)

    tools.remove_frames(index=[0, 2])

    expected = np.delete(mock_observation["data"], [0, 2], axis=0)

    np.testing.assert_array_equal(mock_image_io["data"], expected)

    assert mock_image_io["data"].shape[0] == 2
