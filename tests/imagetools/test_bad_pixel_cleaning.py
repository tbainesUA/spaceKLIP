"""we're testing the behavior of clean_bad_pixels() as an orchestrator,
not the numerical correctness of the individual cleaning algorithms."""

import numpy as np

from spaceKLIP.imagetools import ImageTools


def test_clean_bad_pixels_constructs_cleaning_mask(
    image_database,
    make_mock_observation,
    mock_image_io,
    monkeypatch,
):
    """
    Verify that cleaning selects only eligible bad pixels.

    Pixels containing NaN or flagged DO_NOT_USE should be selected
    for cleaning, except when they are marked NON_SCIENCE or
    protected by the supplied NaN mask.

    The cleaning algorithm is mocked so that this test isolates
    the orchestration logic rather than the correction itself.

    In simple terms:
        Imagine identifying damaged pixels in an image while
        marking certain detector regions as off-limits.

        We check that the software selects the damaged pixels
        without attempting to repair protected regions.
    """
    data = np.full((2, 5, 5), 10.0)
    erro = np.ones_like(data)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Eligible: DO_NOT_USE.
    pxdq[0, 1, 1] = 1

    # Eligible: NaN without a DQ flag.
    data[0, 1, 2] = np.nan

    # Excluded: NON_SCIENCE.
    pxdq[0, 2, 2] = 1 | 512
    data[0, 2, 2] = np.nan

    # Excluded: protected by NaN mask.
    data[0, 3, 3] = np.nan

    nanmask = np.zeros_like(data, dtype=np.uint8)
    nanmask[0, 3, 3] = 1

    captured = {}

    # def mock_timemed(self, data, erro, bad_mask, kwargs):
    #     captured["mask"] = bad_mask.copy()

    def mock_timemed(self, data, erro, bad_mask, kwargs):
        print("Data shape:", data.shape)
        print("Cleaning mask shape:", bad_mask.shape)
        print("Selected pixels:", np.argwhere(bad_mask))

        captured["mask"] = bad_mask.copy()

    monkeypatch.setattr(ImageTools, "fix_bad_pixels_timemed", mock_timemed)

    # Register the synthetic observation using the existing fixtures.
    # observation = make_mock_observation(data=data, erro=erro, pxdq=pxdq)
    key = next(iter(image_database.obs))
    row = image_database.obs[key][0]

    fitsfile = str(row["FITSFILE"])
    nanmaskfile = str(row["NANMASKFILE"])

    observation = make_mock_observation(data=data, erro=erro, pxdq=pxdq)

    mock_image_io["observations"][fitsfile] = observation
    mock_image_io["masks"][nanmaskfile] = nanmask

    assert str(fitsfile) in mock_image_io["observations"]
    # mock_image_io["observations"]["input.fits"] = observation

    # Supply the protected NaN mask through the mocked mask reader.
    # Adapt this assignment to the existing mock_image_io fixture
    # if it stores masks separately.
    mock_image_io["nanmask"] = nanmask

    tool = ImageTools(image_database)

    tool.clean_bad_pixels(method="timemed", plot=False)

    expected_mask = np.zeros_like(data, dtype=bool)
    expected_mask[0, 1, 1] = True
    expected_mask[0, 1, 2] = True

    np.testing.assert_array_equal(captured["mask"], expected_mask)


def test_clean_bad_pixels_dispatches_methods_in_order(
    image_database,
    make_mock_observation,
    mock_image_io,
    monkeypatch,
):
    """
    Verify that cleaning methods execute sequentially and share updates.

    The test replaces the numerical cleaning algorithms with simple
    mocks. The first method corrects a bad pixel and clears its
    cleaning-mask flag. The second method verifies that those changes
    are visible and makes another modification.

    The final output should contain both modifications.

    In simple terms:
        Imagine two workers repairing an image. The first repairs a
        damaged pixel, and the second continues using the updated
        image rather than starting over with the original.

        We check that the workers run in the requested order and
        that their changes are carried through to the saved result.
    """
    data = np.full((2, 5, 5), 10.0)
    erro = np.ones_like(data)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Pixel requiring correction.
    data[0, 2, 2] = np.nan
    pxdq[0, 2, 2] = 1

    # key = next(iter(image_database.obs))
    # fitsfile = str(image_database.obs[key]["FITSFILE"][0])

    # observation = make_mock_observation(
    #     data=data,
    #     erro=erro,
    #     pxdq=pxdq,
    # )

    # mock_image_io["observations"][fitsfile] = observation

    key = next(iter(image_database.obs))
    row = image_database.obs[key][0]

    fitsfile = str(row["FITSFILE"])
    nanmaskfile = str(row["NANMASKFILE"])

    observation = make_mock_observation(
        data=data,
        erro=erro,
        pxdq=pxdq,
    )

    mock_image_io["observations"][fitsfile] = observation

    # Register a compatible NaN mask for this observation.
    mock_image_io["masks"][nanmaskfile] = np.zeros(
        data.shape[-2:],
        dtype=np.uint8,
    )

    calls = []

    def mock_timemed(self, data, erro, bad_mask, kwargs):
        calls.append("timemed")

        assert bad_mask[0, 2, 2]

        data[0, 2, 2] = 10.0
        bad_mask[0, 2, 2] = False

    def mock_localmed(self, data, erro, bad_mask, kwargs):
        calls.append("localmed")

        # The first method's correction must be visible.
        assert data[0, 2, 2] == 10.0
        assert not bad_mask[0, 2, 2]

        # Make an additional change to demonstrate shared data.
        data[0, 1, 1] = 20.0

    monkeypatch.setattr(ImageTools, "fix_bad_pixels_timemed", mock_timemed)
    monkeypatch.setattr(ImageTools, "fix_bad_pixels_localmed", mock_localmed)

    tool = ImageTools(image_database)

    tool.clean_bad_pixels(method="timemed+localmed", plot=False)

    assert calls == ["timemed", "localmed"]

    output_data = mock_image_io["data"]
    output_dq = mock_image_io["pxdq"]

    assert output_data[0, 2, 2] == 10.0
    assert output_data[0, 1, 1] == 20.0

    # The corrected pixel should no longer have DO_NOT_USE set.
    assert output_dq[0, 2, 2] == 0


def test_clean_bad_pixels_preserves_dq_flags(
    image_database,
    make_mock_observation,
    mock_image_io,
    monkeypatch,
):
    """
    Verify that cleaning preserves unrelated DQ flags.

    The test creates pixels with different combinations of JWST
    data-quality flags. A mocked cleaning algorithm repairs one
    DO_NOT_USE pixel while leaving another unresolved.

    The output should clear DO_NOT_USE for the repaired pixel,
    preserve it for the unresolved pixel, and retain all other
    DQ flags.

    In simple terms:
        Imagine a pixel having multiple warning labels. Repairing
        one problem should remove only the corresponding warning,
        not erase the other warnings.

        We check that cleaning updates the appropriate warning
        while preserving the remaining information.
    """
    data = np.full((2, 5, 5), 10.0)
    erro = np.ones_like(data)

    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Successfully cleaned: DO_NOT_USE + another DQ flag.
    pxdq[0, 1, 1] = 5

    # Unresolved: DO_NOT_USE remains set.
    pxdq[0, 1, 2] = 1

    # Unrelated DQ flag: should remain unchanged.
    pxdq[0, 2, 2] = 4

    # NON_SCIENCE: should remain unchanged.
    pxdq[0, 3, 3] = 512

    key = next(iter(image_database.obs))
    row = image_database.obs[key][0]

    fitsfile = str(row["FITSFILE"])
    nanmaskfile = str(row["NANMASKFILE"])

    observation = make_mock_observation(data=data, erro=erro, pxdq=pxdq)

    mock_image_io["observations"][fitsfile] = observation
    mock_image_io["masks"][nanmaskfile] = np.zeros(data.shape[-2:], dtype=np.uint8)

    def mock_timemed(self, data, erro, bad_mask, kwargs):
        # Both DO_NOT_USE pixels should initially be eligible.
        assert bad_mask[0, 1, 1]
        assert bad_mask[0, 1, 2]

        # Successfully repair only the first pixel.
        data[0, 1, 1] = 10.0
        bad_mask[0, 1, 1] = False

        # Leave the second pixel unresolved.
        assert bad_mask[0, 1, 2]

    monkeypatch.setattr(ImageTools, "fix_bad_pixels_timemed", mock_timemed)

    tool = ImageTools(image_database)

    tool.clean_bad_pixels(method="timemed", plot=False)

    expected_dq = pxdq.copy()

    # Clear DO_NOT_USE while preserving the other bit.
    expected_dq[0, 1, 1] = 4

    # The unresolved DO_NOT_USE flag remains set.
    expected_dq[0, 1, 2] = 1

    np.testing.assert_array_equal(mock_image_io["pxdq"], expected_dq)


# ================================
# Individual cleaning method tests
# ================================


def test_fix_bad_pixels_timemed_replaces_temporarily_bad_pixel(image_database):
    """
    Verify temporal-median replacement of a partially flagged pixel.

    A single detector pixel is flagged in one integration. Its science
    value and uncertainty should be replaced by the temporal medians
    of the unflagged integrations, while all other pixels remain
    unchanged.

    In simple terms:
        If a pixel is bad in only one exposure, we should be able
        to estimate its value from the same pixel in the other
        exposures.

        We check that the replacement is correct, its uncertainty
        is updated, and its bad-pixel flag is cleared.
    """
    data = np.full((5, 5, 5), 10.0)
    erro = np.ones_like(data)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Temporal values at one detector pixel.
    data[:, 2, 2] = [8.0, 10.0, 100.0, 12.0, 14.0]
    erro[:, 2, 2] = [1.0, 2.0, 50.0, 3.0, 4.0]

    # Flag the anomalous integration.
    pxdq[2, 2, 2] = 1

    original_data = data.copy()
    original_erro = erro.copy()

    tool = ImageTools(image_database)

    tool.fix_bad_pixels_timemed(data, erro, pxdq)

    # Median of [8, 10, 12, 14] = 11.
    assert data[2, 2, 2] == 11.0

    # Median uncertainty of [1, 2, 3, 4] = 2.5.
    assert erro[2, 2, 2] == 2.5

    # The repaired pixel is no longer flagged.
    assert pxdq[2, 2, 2] == 0

    # All other pixels and integrations remain unchanged.
    unchanged = np.ones(data.shape, dtype=bool)
    unchanged[2, 2, 2] = False

    np.testing.assert_array_equal(data[unchanged], original_data[unchanged])
    np.testing.assert_array_equal(erro[unchanged], original_erro[unchanged])


def test_fix_bad_pixels_localmed_replaces_flagged_pixel(image_database):
    """
    Verify spatial-median replacement using unflagged neighbors.

    A bad pixel is surrounded by pixels with known science values
    and uncertainties. Another flagged neighbor must be excluded
    from the median calculation.

    The algorithm should replace the bad pixel with the median
    of its valid neighbors, update its uncertainty, and clear
    its bad-pixel flag.

    In simple terms:
        If a detector pixel is bad, estimate its value using
        nearby good pixels in the same image.

        We check that the replacement is correct and that
        other flagged pixels do not contaminate the estimate.
    """
    data = np.full((1, 7, 7), 10.0)
    erro = np.full_like(data, 2.0)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Pixel requiring correction.
    data[0, 3, 3] = 100.0
    erro[0, 3, 3] = 50.0
    pxdq[0, 3, 3] = 1

    # Another flagged pixel inside the 5x5 neighborhood.
    data[0, 2, 2] = 1000.0
    erro[0, 2, 2] = 100.0
    pxdq[0, 2, 2] = 1

    # Preserve original arrays for comparison.
    original_data = data.copy()
    original_erro = erro.copy()

    tool = ImageTools(image_database)

    tool.fix_bad_pixels_localmed(data, erro, pxdq)

    # The central pixel is replaced using valid neighbors.
    assert data[0, 3, 3] == 10.0
    assert erro[0, 3, 3] == 2.0
    assert pxdq[0, 3, 3] == 0

    # The neighboring flagged pixel is also recoverable.
    assert data[0, 2, 2] == 10.0
    assert erro[0, 2, 2] == 2.0
    assert pxdq[0, 2, 2] == 0

    # All initially good pixels remain unchanged.
    good_pixels = np.ones(data.shape, dtype=bool)
    good_pixels[0, 3, 3] = False
    good_pixels[0, 2, 2] = False

    np.testing.assert_array_equal(
        data[good_pixels],
        original_data[good_pixels],
    )

    np.testing.assert_array_equal(
        erro[good_pixels],
        original_erro[good_pixels],
    )


def test_fix_bad_pixels_localmed_preserves_unrecoverable_pixels(image_database):
    """
    Verify that pixels without valid neighbors remain flagged.

    An entire image is marked as bad, leaving no valid pixels
    available for spatial-median interpolation.

    The algorithm should preserve the original science values,
    uncertainties, and bad-pixel flags.

    In simple terms:
        If every nearby pixel is bad, there is no reliable
        information available to repair a damaged pixel.

        We check that the algorithm does not invent a replacement
        or incorrectly mark the pixel as repaired.
    """
    data = np.full((1, 5, 5), 10.0)
    erro = np.full_like(data, 2.0)

    # Every pixel is flagged as bad.
    pxdq = np.ones(data.shape, dtype=np.uint32)

    # Give the central pixel distinct values.
    data[0, 2, 2] = 100.0
    erro[0, 2, 2] = 50.0

    original_data = data.copy()
    original_erro = erro.copy()
    original_pxdq = pxdq.copy()

    tool = ImageTools(image_database)

    tool.fix_bad_pixels_localmed(data, erro, pxdq)

    # No valid neighbors exist, so nothing should change.
    np.testing.assert_array_equal(data, original_data)

    np.testing.assert_array_equal(erro, original_erro)

    np.testing.assert_array_equal(pxdq, original_pxdq)


def test_fix_bad_pixels_medfilt_replaces_flagged_pixel(image_database):
    """
    Verify median-filter replacement of flagged pixels.

    A flagged pixel is surrounded by known science values and
    uncertainties. The algorithm should replace it using the
    spatial median filter and clear its bad-pixel flag, without
    modifying unflagged pixels.

    In simple terms:
        Replace a damaged pixel with the median value from its
        local image neighborhood.

        We check that the replacement is correct and that the
        remaining image is preserved.
    """
    data = np.full((1, 5, 5), 10.0)
    erro = np.full_like(data, 2.0)
    pxdq = np.zeros(data.shape, dtype=np.uint32)

    # Flag a central pixel with an anomalous value.
    data[0, 2, 2] = 100.0
    erro[0, 2, 2] = 50.0
    pxdq[0, 2, 2] = 1

    original_data = data.copy()
    original_erro = erro.copy()

    tool = ImageTools(image_database)

    tool.fix_bad_pixels_medfilt(
        data,
        erro,
        pxdq,
        medfilt_kwargs={
            "size": 3,
        },
    )

    # The surrounding pixels establish the expected medians.
    assert data[0, 2, 2] == 10.0
    assert erro[0, 2, 2] == 2.0

    # The bad-pixel flag should be cleared.
    assert pxdq[0, 2, 2] == 0

    # All originally good pixels must remain unchanged.
    good_pixels = np.ones(data.shape, dtype=bool)
    good_pixels[0, 2, 2] = False

    np.testing.assert_array_equal(data[good_pixels], original_data[good_pixels])

    np.testing.assert_array_equal(erro[good_pixels], original_erro[good_pixels])
