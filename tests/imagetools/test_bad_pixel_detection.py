import numpy as np
import pytest

from spaceKLIP.imagetools import ImageTools


def test_find_bad_pixels_default_dqarr(
    image_database, make_mock_observation, mock_image_io
):
    """
    Testing the default behavior.

    Characterize the current output of the default dqarr detection method.

    The dqarr method examines the science image for NaN pixels, which
    represent invalid measurements that should be identified for cleaning.

    This test creates an image containing one NaN pixel and an existing
    DO_NOT_USE flag. With set_dq_zero=True, the method starts with an empty
    working DQ array.

    The current implementation identifies NaNs but updates the original
    DQ array rather than the working array used to construct the output.
    As a result, the saved DQ array contains no flagged pixels.

    In simple terms:
        We deliberately place one broken pixel in an image and ask the
        method to find it. The current code looks for the broken pixel,
        but does not transfer that information into the saved DQ array.
        This test records that behavior so we can detect when it changes.
    """

    fitsfile = image_database.obs["concat1"]["FITSFILE"][0]

    data = np.full((2, 5, 5), 10.0)
    data[0, 2, 2] = np.nan

    pxdq = np.zeros(data.shape, dtype=np.uint32)
    pxdq[1, 1, 1] = 1

    mock_image_io["observations"][fitsfile] = make_mock_observation(data, pxdq=pxdq)

    tools = ImageTools(image_database)

    tools.find_bad_pixels(method="dqarr")

    np.testing.assert_array_equal(mock_image_io["pxdq"], np.zeros_like(pxdq))


def test_find_bad_pixels_dispatches_methods_in_order(
    image_database, make_mock_observation, mock_image_io, monkeypatch
):
    """
    When multiple bad-pixel detection methods are requested, does find_bad_pixels()
    execute them in the specified order and pass the same working DQ array between them?

    Verify that multiple detection methods run in the requested order.

    The find_bad_pixels() method allows users to combine detection
    algorithms by separating their names with a '+' sign. Each algorithm
    receives the same working DQ array, allowing later methods to build
    upon pixels identified by earlier methods.

    This test replaces the custom and timeints algorithms with simple
    mock functions. The custom method flags one pixel, and the timeints
    method checks that this flag already exists before adding another.

    We then verify that both methods ran in the correct order and that
    both flagged pixels appear in the saved DQ array.

    In simple terms:
        Imagine two people checking the same picture for broken pixels.
        The first person marks one broken pixel. The second person sees
        that mark and adds another. We check that they work in the right
        order and that neither person's markings are lost.
    """

    fitsfile = image_database.obs["concat1"]["FITSFILE"][0]

    data = np.full((2, 5, 5), 10.0)

    mock_image_io["observations"][fitsfile] = make_mock_observation(data)

    calls = []

    def mock_custom(self, data, erro, pxdq, key, kwargs):
        calls.append("custom")
        pxdq[:, 1, 1] = 1

    def mock_timeints(self, data, erro, pxdq, key, kwargs):
        calls.append("timeints")

        # Verify the first algorithm's flags are available.
        assert np.all(pxdq[:, 1, 1] == 1)

        pxdq[:, 2, 2] = 1

    monkeypatch.setattr(ImageTools, "find_bad_pixels_custom", mock_custom)

    monkeypatch.setattr(ImageTools, "find_bad_pixels_timeints", mock_timeints)

    tools = ImageTools(image_database)

    tools.find_bad_pixels(method="custom+timeints")

    # Verify execution order.
    assert calls == ["custom", "timeints"]

    # Verify both algorithms contributed to the final DQ array.
    output_dq = mock_image_io["pxdq"]

    assert np.all(output_dq[:, 1, 1] == 1)
    assert np.all(output_dq[:, 2, 2] == 1)

    # Verify an unrelated pixel remains unflagged.
    assert np.all(output_dq[:, 0, 0] == 0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "find_bad_pixels converts existing nonzero DQ values to "
        "boolean, unintentionally adding the DO_NOT_USE bit."
    ),
)
def test_find_bad_pixels_preserves_existing_dq(
    image_database, make_mock_observation, mock_image_io, monkeypatch
):
    """
    Verify that existing DQ flags survive additional bad-pixel detection.

    JWST DQ arrays store detector conditions using individual bit flags.
    When set_dq_zero=False, find_bad_pixels() should preserve existing
    flags while allowing detection methods to identify additional bad pixels.

    This test creates an observation with an existing DQ flag of 4 at one
    pixel. A mocked custom detection method then flags a different pixel
    with DO_NOT_USE (value 1).

    The expected output should preserve the original flag (4) and add
    the new flag (1) without modifying unrelated pixels.

    Known issue:
        The current implementation converts the working DQ array into
        a Boolean mask when set_dq_zero=False and the first detection
        method is not 'dqarr':

            pxdq_temp = pxdq_temp > 0

        When the Boolean mask is combined with the original DQ array
        using bitwise OR, existing nonzero flags unintentionally acquire
        the DO_NOT_USE bit.

        For example, an existing DQ value of 4 becomes 5 (4 | 1).

        This test is marked xfail(strict=True) until the DQ handling
        is corrected. The expected output represents the scientifically
        intended behavior, not the current implementation.

    In simple terms:
        Imagine a picture already has a warning sticker describing one
        problem. When we inspect the picture again, the software adds
        a second warning to that same pixel even though no new problem
        was found there.

        We want to preserve the original warning and only add new
        warnings to pixels where additional problems were detected.

    Note: found that when find_bad_pixels() is instructed to preserve
    existing  data-quality flags, it can unintentionally introduce an
    additional DO_NOT_USE flag.

    For example, a pixel originally flagged with DQ value 4 becomes 5, even
    though the detection algorithm never identified that pixel as newly bad.

    The problem comes from converting the existing DQ array into a Boolean mask
    before combining it with the original flags. This loses the distinction between
    different DQ bits.

    Rather than modifying the legacy implementation immediately, I've documented the
    defect and marked the test as an expected failure. The test still checks the
    scientifically correct behavior, so when we fix the implementation, we can
    verify that the issue has been resolved.
    """

    fitsfile = image_database.obs["concat1"]["FITSFILE"][0]

    data = np.full((2, 5, 5), 10.0)

    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Existing DQ flag, distinct from DO_NOT_USE.
    pxdq[:, 0, 0] = 4

    mock_image_io["observations"][fitsfile] = make_mock_observation(data, pxdq=pxdq)

    def mock_custom(self, data, erro, pxdq, key, kwargs):
        # Add a new DO_NOT_USE flag to a different pixel.
        pxdq[:, 2, 2] = 1

    monkeypatch.setattr(ImageTools, "find_bad_pixels_custom", mock_custom)

    tools = ImageTools(image_database)

    tools.find_bad_pixels(method="custom", set_dq_zero=False)

    output_dq = mock_image_io["pxdq"]

    expected_dq = pxdq.copy()
    expected_dq[:, 2, 2] = 1

    np.testing.assert_array_equal(output_dq, expected_dq)


def test_find_bad_pixels_skips_unselected_types(
    image_database, make_mock_observation, mock_image_io, monkeypatch
):
    """
    Verify that detection methods skip unselected observation types.

    The types argument controls which observations are eligible for
    bad-pixel detection. Observations with a TYPE outside this selection
    should not be processed by the detection algorithms.

    This test creates a SCI observation but requests processing only
    for REF observations. A mocked custom detection method records
    whether it was called.

    The method should not be called, and the saved DQ array should
    contain no newly flagged pixels.

    Note:
        The current implementation still writes an output observation
        even when its type is excluded. This test checks that detection
        is skipped, not that the entire observation is skipped.

    In simple terms:
        Imagine we want to inspect only reference images for broken
        pixels. We give the software a science image instead and
        check that it does not perform the inspection or add any
        new warning flags.
    """

    fitsfile = image_database.obs["concat1"]["FITSFILE"][0]

    data = np.full((2, 5, 5), 10.0)

    mock_image_io["observations"][fitsfile] = make_mock_observation(data)

    # Ensure the observation is classified as SCI.
    image_database.obs["concat1"]["TYPE"][0] = "SCI"

    calls = []

    def mock_custom(self, data, erro, pxdq, key, kwargs):
        calls.append("custom")
        pxdq[:, 2, 2] = 1

    monkeypatch.setattr(ImageTools, "find_bad_pixels_custom", mock_custom)

    tools = ImageTools(image_database)

    tools.find_bad_pixels(method="custom", types=["REF"])

    # Detection should not run for a SCI observation.
    assert calls == []

    # No new bad pixels should be flagged.
    np.testing.assert_array_equal(
        mock_image_io["pxdq"], np.zeros_like(data, dtype=np.uint32)
    )

# ===================================
# Individual Pixel detection methods
# ===================================


