import shutil
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.table import Table

from spaceKLIP.database import Database


def test_read_jwst_s012_data_rejects_empty_datapaths(tmp_path):
    """Reject empty input before adding observations to the database."""
    database = Database(output_dir=tmp_path)
    database.verbose = False

    with pytest.raises(UserWarning, match="^Could not find any data paths$"):
        database.read_jwst_s012_data([])

    assert database.obs == {}


def test_read_jwst_s012_data_reads_one_miri_exposure(tmp_path, miri_s012_file):
    """Read one MIRI science exposure into one observation table."""
    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data([miri_s012_file])

    expected_key = "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065"
    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert isinstance(observations, Table)
    assert len(observations) == 1

    row = observations[0]
    assert row["FITSFILE"] == miri_s012_file
    assert row["TYPE"] == "SCI"
    assert row["DATAMODL"] == "STAGE1"
    assert row["FILTER"] == "F1065C"
    assert row["NINTS"] == 2
    assert row["EFFINTTM"] == 10.0


def test_read_jwst_s012_data_reads_one_nircam_exposure(tmp_path, nircam_s012_file):
    """Read one NIRCam science exposure into one observation table."""
    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data([nircam_s012_file])

    expected_key = "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320"
    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert isinstance(observations, Table)
    assert len(observations) == 1

    row = observations[0]
    assert row["FITSFILE"] == nircam_s012_file
    assert row["TYPE"] == "SCI"
    assert row["DATAMODL"] == "STAGE1"
    assert row["FILTER"] == "F335M"
    assert row["NINTS"] == 2
    assert row["EFFINTTM"] == 10.0

    expected_mask = nircam_s012_file.replace(".fits", "_psfmask.fits")
    assert row["MASKFILE"] == expected_mask


@pytest.mark.parametrize(
    "fixture_name, expected_key",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_groups_matching_science_exposures(
    request, tmp_path, fixture_name, expected_key
):
    """Keep two matching science exposures in one concatenation.

    Group two matching science exposures into one concatenation.

    Run for both MIRI and NIRCam using their FITS-file fixtures.
    Copy the first exposure to a second filename so the two inputs
    have identical instrument configuration but distinct paths.
    Copy an existing companion PSF mask to avoid a reference lookup.

    Read both files into a fresh Database and verify that:
    - Exactly one concatenation has the expected configuration key.
    - Both exposures are retained in input order.
    - Both rows are classified as SCI and identified as STAGE1.
    - Each row preserves the integration count from its FITS header.

    This characterizes science-only grouping before refactoring.
    """
    first_path = request.getfixturevalue(fixture_name)
    second_path = str(Path(first_path).with_name("second_science_rateints.fits"))
    shutil.copyfile(first_path, second_path)

    # Preserve the local-mask setup for the second NIRCam exposure.
    first_mask = Path(first_path.replace(".fits", "_psfmask.fits"))
    if first_mask.exists():
        second_mask = second_path.replace(".fits", "_psfmask.fits")
        shutil.copyfile(first_mask, second_mask)

    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data([first_path, second_path])

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 2
    assert list(observations["FITSFILE"]) == [first_path, second_path]
    assert list(observations["TYPE"]) == ["SCI", "SCI"]
    assert list(observations["DATAMODL"]) == ["STAGE1", "STAGE1"]
    assert list(observations["NINTS"]) == [2, 2]


@pytest.mark.parametrize(
    "fixture_name, expected_key",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_classifies_science_and_reference(
    request, tmp_path, fixture_name, expected_key
):
    """Classify matching exposures using their IS_PSF headers.

    Run for MIRI and NIRCam. Copy a science exposure and mark the
    copy as a PSF reference with a different target name. Preserve
    any local PSF mask to avoid a reference-file lookup.

    Supply the reference before the science exposure and verify
    that both share one concatenation, with the science row stored
    first and the reference row second.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = str(Path(science_path).with_name("reference_rateints.fits"))
    shutil.copyfile(science_path, reference_path)

    with fits.open(reference_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"

    science_mask = Path(science_path.replace(".fits", "_psfmask.fits"))
    if science_mask.exists():
        reference_mask = reference_path.replace(".fits", "_psfmask.fits")
        shutil.copyfile(science_mask, reference_mask)

    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data([reference_path, science_path])

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 2
    assert list(observations["FITSFILE"]) == [science_path, reference_path]
    assert list(observations["TYPE"]) == ["SCI", "REF"]
    assert list(observations["TARGPROP"]) == [
        "SCIENCE_TARGET",
        "REFERENCE_TARGET",
    ]


@pytest.mark.parametrize(
    "fixture_name, expected_key",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_psflibpaths_overrides_is_psf(
    request, tmp_path, fixture_name, expected_key
):
    """Explicit PSF-library membership overrides IS_PSF=False.

    Copy a science exposure without changing its header and supply
    the copy only through psflibpaths. Verify that both exposures
    share one concatenation and the copy is classified as REF.

    Preserve any local PSF mask to avoid a reference-file lookup.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = str(
        Path(science_path).with_name("explicit_reference_rateints.fits")
    )
    shutil.copyfile(science_path, reference_path)

    science_mask = Path(science_path.replace(".fits", "_psfmask.fits"))
    if science_mask.exists():
        reference_mask = reference_path.replace(".fits", "_psfmask.fits")
        shutil.copyfile(science_mask, reference_mask)

    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data(
        datapaths=[science_path],
        psflibpaths=[reference_path],
    )

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert list(observations["FITSFILE"]) == [
        science_path,
        reference_path,
    ]
    assert list(observations["TYPE"]) == ["SCI", "REF"]


@pytest.mark.parametrize(
    "fixture_name",
    ["miri_s012_file", "nircam_s012_file"],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_defaults_centers_to_crpix(request, tmp_path, fixture_name):
    """Default missing mask and star centers to FITS CRPIX values.

    Both instrument fixtures provide CRPIX1/2 but omit center
    keywords. Verify that mask and NaN-mask centers inherit CRPIX,
    and star centers inherit the resulting mask centers.
    """
    path = request.getfixturevalue(fixture_name)

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([path])

    assert len(database.obs) == 1
    observations = next(iter(database.obs.values()))
    assert len(observations) == 1
    row = observations[0]

    for column in (
        "CRPIX1",
        "CRPIX2",
        "MASKCENX",
        "MASKCENY",
        "NANMASKCENX",
        "NANMASKCENY",
        "STARCENX",
        "STARCENY",
    ):
        assert row[column] == pytest.approx(2.5), column


@pytest.mark.parametrize(
    "fixture_name",
    ["miri_s012_file", "nircam_s012_file"],
    ids=["miri", "nircam"],
)
@pytest.mark.parametrize(
    "parity, expected_roll",
    [(-1, 35.0), (1, 25.0)],
    ids=["negative-parity", "positive-parity"],
)
def test_read_jwst_s012_data_corrects_roll_angle(
    request, tmp_path, fixture_name, parity, expected_roll
):
    """Apply the aperture-angle and parity correction to ROLL_REF.

    Set nonzero SCI-header angles and exercise both parity signs
    for MIRI and NIRCam. Verify that the stored roll is 35 degrees
    for negative parity and 25 degrees for positive parity.
    """
    path = request.getfixturevalue(fixture_name)

    with fits.open(path, mode="update") as hdul:
        hdul["SCI"].header["ROLL_REF"] = 30.0
        hdul["SCI"].header["V3I_YANG"] = 5.0
        hdul["SCI"].header["VPARITY"] = parity

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([path])

    assert len(database.obs) == 1
    observations = next(iter(database.obs.values()))
    assert len(observations) == 1
    assert observations["ROLL_REF"][0] == pytest.approx(expected_roll)


def test_read_jwst_s012_data_associates_miri_science_background(
    tmp_path, miri_s012_file
):
    """Retain a matching MIRI background as SCI_BG.

    Copy a science exposure and give it a background target name.
    Keep the instrument configuration, integration count, and
    integration time identical to the science exposure.

    Verify that both files share one concatenation and that the
    background is classified as SCI_BG without explicit bgpaths.
    """
    science_path = miri_s012_file
    background_path = str(Path(science_path).with_name("miri_background_rateints.fits"))
    shutil.copyfile(science_path, background_path)

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["TARGPROP"] = "BACKGROUND_TARGET"

    database = Database(output_dir=tmp_path)
    database.verbose = False

    database.read_jwst_s012_data([science_path, background_path])

    expected_key = "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065"
    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 2
    assert list(observations["FITSFILE"]) == [
        science_path,
        background_path,
    ]
    assert list(observations["TYPE"]) == ["SCI", "SCI_BG"]
    assert list(observations["TARGPROP"]) == [
        "SCIENCE_TARGET",
        "BACKGROUND_TARGET",
    ]


@pytest.mark.parametrize(
    "keyword, incompatible_value",
    [
        ("NINTS", 3),
        ("EFFINTTM", 20.0),
    ],
    ids=["integration-count-mismatch", "integration-time-mismatch"],
)
def test_read_jwst_s012_data_rejects_unmatched_miri_background(
    tmp_path, miri_s012_file, keyword, incompatible_value
):
    """Reject a background incompatible with the science exposure.

    Copy a MIRI science exposure and identify it as background.
    Change either NINTS or EFFINTTM while preserving the other.
    With no reference exposure available, verify that association
    raises the current UserWarning exception.
    """
    science_path = miri_s012_file
    background_path = str(
        Path(science_path).with_name("unmatched_background_rateints.fits")
    )
    shutil.copyfile(science_path, background_path)

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["TARGPROP"] = "BACKGROUND_TARGET"
        hdul[0].header[keyword] = incompatible_value

    database = Database(output_dir=tmp_path)
    database.verbose = False

    with pytest.raises(
        UserWarning,
        match="could not be matched with PSF",
    ) as exc_info:
        database.read_jwst_s012_data([science_path, background_path])

    assert background_path in str(exc_info.value)
