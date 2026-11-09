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


def test_find_bad_pixels_sigclip_detects_positive_outlier():
    """
    Verify that local sigma clipping identifies an isolated bright pixel.

    The test creates a smooth synthetic image containing one unusually
    bright pixel. The algorithm should identify that pixel as bad while
    leaving the remaining pixels unflagged.

    It also verifies that the DQ array is updated in place and that
    the original science data is not modified.

    In simple terms:
        Imagine a picture where nearly every pixel has a similar
        brightness except for one that is much brighter than its
        neighbors. We check that the algorithm finds that unusual
        pixel without changing the picture or marking normal pixels
        as bad.

    Note:
        Negative-outlier detection is effectively disabled using
        neg_sigma=100 because this test specifically evaluates positive
        outlier detection.

        The default negative threshold (neg_sigma=1) can flag normal
        low-valued pixels in a smoothly varying background. Testing
        negative-outlier behavior separately may be warranted.
    """

    ny, nx = 101, 101

    y, x = np.indices((ny, nx))

    image = 10.0 + 0.01 * x + 0.02 * y

    data = image[np.newaxis, ...].copy()
    data[0, 50, 50] = 100.0

    original_data = data.copy()

    erro = np.full_like(data, 0.1)
    pxdq = np.zeros(data.shape, dtype=np.uint32)
    non_science = np.zeros(data.shape, dtype=bool)

    ImageTools.find_bad_pixels_sigclip(
        None,
        data,
        erro,
        pxdq,
        non_science,
        sigclip_kwargs={
            "mode": "local",
            "sigma": 5.0,
            "neg_sigma": 100.0,
        },
    )

    # expected_dq = np.zeros_like(pxdq)
    # expected_dq[0, 10, 10] = 1

    expected_dq = np.zeros_like(pxdq)
    expected_dq[0, 50, 50] = 1

    np.testing.assert_array_equal(pxdq, expected_dq)
    np.testing.assert_array_equal(data, original_data)


@pytest.mark.parametrize(
    "custom_input, expected_positions",
    [
        pytest.param(
            "2d_mask",
            [(0, 1, 2), (1, 1, 2)],
            id="2d-mask",
        ),
        pytest.param(
            "3d_mask",
            [(0, 1, 2)],
            id="3d-mask",
        ),
        pytest.param(
            "2d_coordinates",
            [(0, 1, 2), (1, 1, 2)],
            id="2d-coordinates",
        ),
        pytest.param(
            "3d_coordinates",
            [(1, 1, 2)],
            id="3d-coordinates",
        ),
    ],
)
def test_find_bad_pixels_custom_applies_supplied_flags(
    custom_input,
    expected_positions,
):
    """
    Verify that custom bad-pixel inputs flag the requested pixels.

    The custom method accepts either a 2D/3D bad-pixel mask or a list
    of pixel coordinates. A 2D input applies to every integration,
    while a 3D input can select individual integrations.

    This test checks all four supported representations, verifies
    that existing binary DQ flags are retained, and confirms that
    unrelated pixels remain unflagged.

    In simple terms:
        We give the software a list or map showing which pixels are
        broken. We check that it marks exactly those pixels, keeps
        any existing warnings, and doesn't mark other pixels.

    Note:
        The current implementation interprets coordinate inputs as
        [x, y] or [integration, x, y], even though the method
        documentation describes them as [y, x] and [key, y, x].

        This test follows the implemented convention to characterize
        existing behavior. The discrepancy should be clarified before
        changing the public interface.
    """
    data = np.full((2, 5, 5), 10.0)
    erro = np.ones_like(data)

    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Existing bad pixel that should remain flagged.
    pxdq[0, 0, 0] = 1

    if custom_input == "2d_mask":
        custom = np.zeros((5, 5), dtype=np.uint8)
        custom[1, 2] = 1

    elif custom_input == "3d_mask":
        custom = np.zeros(data.shape, dtype=np.uint8)
        custom[0, 1, 2] = 1

    elif custom_input == "2d_coordinates":
        custom = [[2, 1]]  # [x, y]

    elif custom_input == "3d_coordinates":
        custom = [[1, 2, 1]]  # [integration, x, y]

    ImageTools.find_bad_pixels_custom(
        None,
        data,
        erro,
        pxdq,
        key="concat1",
        custom_kwargs={"concat1": custom},
    )

    expected_dq = np.zeros_like(pxdq)
    expected_dq[0, 0, 0] = 1

    for integration, y, x in expected_positions:
        expected_dq[integration, y, x] = 1

    np.testing.assert_array_equal(pxdq, expected_dq)


def test_find_bad_pixels_timeints_detects_temporal_outlier():
    """
    Verify that temporal clipping detects a transient bad pixel.

    This test creates a synthetic image cube containing 21 integrations
    with small temporal variations. One pixel is made unusually bright
    in a single integration.

    The algorithm should identify that measurement as bad without
    flagging the same detector pixel in other integrations or marking
    unrelated pixels.

    The test also verifies that the input science data remains unchanged
    and that the supplied DQ array is updated in place.

    In simple terms:
        Imagine taking 21 pictures of the same scene. One pixel looks
        normal in almost every picture but suddenly becomes much brighter
        in one picture.

        We check that the software identifies that one unusual
        measurement without marking the pixel as broken in every picture.
    """
    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=10.0,
        scale=0.1,
        size=(21, 10, 10),
    )

    # Inject one transient bad pixel.
    data[10, 5, 5] = 100.0

    original_data = data.copy()

    erro = np.full_like(data, 0.1)
    pxdq = np.zeros(data.shape, dtype=np.uint32)
    non_science = np.zeros(data.shape, dtype=bool)

    ImageTools.find_bad_pixels_timeints(
        None,
        data,
        erro,
        pxdq,
        non_science,
        timeints_kwargs={
            "mode": "per_pixel",
            "sigma": 10.0,
        },
    )

    expected_dq = np.zeros_like(pxdq)
    expected_dq[10, 5, 5] = 1

    np.testing.assert_array_equal(pxdq, expected_dq)
    np.testing.assert_array_equal(data, original_data)


def test_find_bad_pixels_timeints_group_pixels_detects_outlier():
    """
    Verify that flux-grouped temporal detection identifies an outlier.

    The test creates two populations of detector pixels with different
    flux levels and small temporal variations. A single anomalous
    measurement is injected into one of the brighter pixels.

    The algorithm should detect the anomalous measurement using
    flux-group statistics while leaving normal measurements unflagged.

    In simple terms:
        Imagine grouping pixels according to their typical brightness.
        Instead of judging each pixel independently, we compare its
        behavior against other pixels with similar brightness.

        We check that an unusually bright measurement is identified
        without marking the other measurements as bad.
    """
    rng = np.random.default_rng(42)

    nint, ny, nx = 21, 10, 10

    # Two distinct flux populations.
    data = np.empty((nint, ny, nx), dtype=float)

    data[:, :, :5] = rng.normal(
        loc=10.0,
        scale=0.1,
        size=(nint, ny, 5),
    )

    data[:, :, 5:] = rng.normal(
        loc=100.0,
        scale=0.5,
        size=(nint, ny, 5),
    )

    # Inject one transient anomaly into the bright population.
    data[10, 5, 7] = 150.0

    original_data = data.copy()

    erro = np.ones_like(data)
    pxdq = np.zeros(data.shape, dtype=np.uint32)
    non_science = np.zeros(data.shape, dtype=bool)

    ImageTools.find_bad_pixels_timeints(
        None,
        data,
        erro,
        pxdq,
        non_science,
        timeints_kwargs={
            "mode": "group_pixels",
            "sigma": 10.0,
            "n_groups": 2,
            "plot_diagnostics": False,
        },
    )

    expected_dq = np.zeros_like(pxdq)
    expected_dq[10, 5, 7] = 1

    np.testing.assert_array_equal(pxdq, expected_dq)
    np.testing.assert_array_equal(data, original_data)


def test_find_bad_pixels_gradient_detects_isolated_outlier():
    """
    Verify that gradient detection identifies an isolated bright pixel.

    The test creates two synthetic images with a uniform background.
    A single bright pixel is injected into the first integration,
    while the second integration remains unchanged.

    The algorithm should flag only the injected pixel in the first
    integration, leaving the other pixels unflagged.

    The test also verifies that the original science data remains
    unchanged and that the DQ array is updated in place.

    In simple terms:
        Imagine two pictures of a smooth, evenly illuminated surface.
        One picture contains a single unusually bright pixel.

        We check that the algorithm identifies that pixel without
        marking the same location in the other picture or changing
        either original image.
    """
    data = np.full((2, 21, 21), 10.0)

    # Inject one isolated bright pixel into the first integration.
    data[0, 10, 10] = 100.0

    original_data = data.copy()

    erro = np.ones_like(data)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    ImageTools.find_bad_pixels_gradient(
        None,
        data,
        erro,
        pxdq,
        key="concat1",
        gradient_kwargs={
            "sigma": 0.5,
            "threshold": 0.05,
            "negative": False,
        },
    )

    expected_dq = np.zeros_like(pxdq)
    expected_dq[0, 10, 10] = 1

    np.testing.assert_array_equal(pxdq, expected_dq)
    np.testing.assert_array_equal(data, original_data)
