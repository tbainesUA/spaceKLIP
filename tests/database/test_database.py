import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from spaceKLIP.database import Database


def _copy_exposure(source_path, filename):
    """Copy an exposure and its companion PSF mask, if present."""
    destination = str(Path(source_path).with_name(filename))
    shutil.copyfile(source_path, destination)

    source_mask = Path(source_path.replace(".fits", "_psfmask.fits"))
    if source_mask.exists():
        destination_mask = destination.replace(".fits", "_psfmask.fits")
        shutil.copyfile(source_mask, destination_mask)

    return destination


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
def test_read_jwst_s012_data_associates_science_background(
    request, tmp_path, fixture_name, expected_key
):
    """Classify a matching background as SCI_BG for either instrument.

    Copy a science exposure and give it a background target name.
    Preserve matching instrument configuration, NINTS, and EFFINTTM.
    Verify automatic classification without explicit bgpaths.
    """
    science_path = request.getfixturevalue(fixture_name)
    background_path = _copy_exposure(science_path, "background_rateints.fits")

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["TARGPROP"] = "BACKGROUND_TARGET"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([science_path, background_path])

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
    "fixture_name",
    ["miri_s012_file", "nircam_s012_file"],
    ids=["miri", "nircam"],
)
@pytest.mark.parametrize(
    "keyword, incompatible_value",
    [
        ("NINTS", 3),
        ("EFFINTTM", 20.0),
    ],
    ids=["integration-count-mismatch", "integration-time-mismatch"],
)
def test_read_jwst_s012_data_rejects_unmatched_background(
    request, tmp_path, fixture_name, keyword, incompatible_value
):
    """Reject incompatible backgrounds for either instrument.

    Change either NINTS or EFFINTTM while preserving the other.
    With no reference exposure available, verify that association
    raises UserWarning and identifies the background file.
    """
    science_path = request.getfixturevalue(fixture_name)
    background_path = _copy_exposure(science_path, "unmatched_background_rateints.fits")

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
def test_read_jwst_s012_data_reassigns_background_to_reference(
    request, tmp_path, fixture_name, expected_key
):
    """Reassign SCI_BG to REF_BG when exposure settings match REF.

    Give the reference and background EFFINTTM=20 seconds while
    the science retains 10 seconds. Keep background IS_PSF=False.
    Use distinct target names so exposure matching determines
    reassignment rather than target-name reassociation.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = _copy_exposure(science_path, "reference_rateints.fits")
    background_path = _copy_exposure(science_path, "background_rateints.fits")

    with fits.open(reference_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"
        hdul[0].header["EFFINTTM"] = 20.0

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["TARGPROP"] = "BACKGROUND_TARGET"
        hdul[0].header["EFFINTTM"] = 20.0

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([science_path, reference_path, background_path])

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 3

    rows_by_path = {row["FITSFILE"]: row for row in observations}
    assert set(rows_by_path) == {
        science_path,
        reference_path,
        background_path,
    }
    assert rows_by_path[science_path]["TYPE"] == "SCI"
    assert rows_by_path[reference_path]["TYPE"] == "REF"
    assert rows_by_path[background_path]["TYPE"] == "REF_BG"


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
def test_read_jwst_s012_data_reassigns_background_to_science(
    request, tmp_path, fixture_name, expected_key
):
    """Reassign REF_BG to SCI_BG when exposure settings match SCI.

    Mark the background IS_PSF=True but retain the science
    integration time of 10 seconds. Give the reference an
    integration time of 20 seconds.

    Use distinct target names so exposure matching determines
    reassignment rather than target-name reassociation.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = _copy_exposure(science_path, "reference_rateints.fits")
    background_path = _copy_exposure(science_path, "background_rateints.fits")

    with fits.open(reference_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"
        hdul[0].header["EFFINTTM"] = 20.0

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "BACKGROUND_TARGET"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([science_path, reference_path, background_path])

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 3

    rows_by_path = {row["FITSFILE"]: row for row in observations}
    assert set(rows_by_path) == {
        science_path,
        reference_path,
        background_path,
    }
    assert rows_by_path[science_path]["TYPE"] == "SCI"
    assert rows_by_path[reference_path]["TYPE"] == "REF"
    assert rows_by_path[background_path]["TYPE"] == "SCI_BG"


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
@pytest.mark.parametrize(
    "assoc_using_targname, expected_type",
    [
        (True, "REF_BG"),
        (False, "SCI_BG"),
    ],
    ids=["target-association-enabled", "target-association-disabled"],
)
def test_read_jwst_s012_data_associates_background_by_target_name(
    request,
    tmp_path,
    fixture_name,
    expected_key,
    assoc_using_targname,
    expected_type,
):
    """Control background reassociation using the target-name switch.

    Give science, reference, and background identical exposure
    settings so exposure matching retains the initial SCI_BG role.
    Identify the background explicitly through bgpaths and give
    it the reference target name.

    Verify reassignment to REF_BG when target-name association is
    enabled, and retention of SCI_BG when it is disabled.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = _copy_exposure(science_path, "reference_rateints.fits")
    background_path = _copy_exposure(science_path, "background_rateints.fits")

    with fits.open(reference_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"

    with fits.open(background_path, mode="update") as hdul:
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data(
        datapaths=[science_path, reference_path, background_path],
        bgpaths=[background_path],
        assoc_using_targname=assoc_using_targname,
    )

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 3

    rows_by_path = {row["FITSFILE"]: row for row in observations}
    assert set(rows_by_path) == {
        science_path,
        reference_path,
        background_path,
    }
    assert rows_by_path[science_path]["TYPE"] == "SCI"
    assert rows_by_path[reference_path]["TYPE"] == "REF"
    assert rows_by_path[background_path]["TYPE"] == expected_type


@pytest.mark.parametrize(
    "fixture_name, original_key, second_key",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_FULL",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_FULL",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_separates_different_subarrays(
    request, tmp_path, fixture_name, original_key, second_key
):
    """Group different SUBARRAY values into separate concatenations.

    Copy one science exposure and change only its SUBARRAY header.
    Verify that each configuration receives its own table and
    contains only its corresponding exposure.

    This isolates metadata grouping; the copied file is not a
    physically representative FULL-frame observation.
    """
    first_path = request.getfixturevalue(fixture_name)
    second_path = _copy_exposure(first_path, "second_configuration_rateints.fits")

    with fits.open(second_path, mode="update") as hdul:
        hdul[0].header["SUBARRAY"] = "FULL"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([first_path, second_path])

    assert set(database.obs) == {original_key, second_key}

    for key, expected_path in (
        (original_key, first_path),
        (second_key, second_path),
    ):
        observations = database.obs[key]
        assert len(observations) == 1
        assert observations["FITSFILE"][0] == expected_path
        assert observations["TYPE"][0] == "SCI"

    assert database.obs[second_key]["SUBARRAY"][0] == "FULL"


@pytest.mark.parametrize(
    "fixture_name, expected_key, ta_exp_type",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
            "MIR_TACQ",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
            "NRC_TACQ",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_associates_target_acquisitions(
    request, tmp_path, fixture_name, expected_key, ta_exp_type
):
    """Merge TA exposures and classify them by their target names.

    Create science and reference exposures with distinct targets,
    plus one TA exposure for each target. Give TA files the mask
    metadata that triggers the reader's TA grouping branch.

    Verify all four files share the coronagraphic concatenation
    and receive SCI, REF, SCI_TA, and REF_TA roles.

    These small synthetic files isolate metadata association;
    they do not reproduce physical TA image geometry.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = _copy_exposure(science_path, "reference_rateints.fits")
    science_ta_path = _copy_exposure(science_path, "science_ta_rateints.fits")
    reference_ta_path = _copy_exposure(science_path, "reference_ta_rateints.fits")

    with fits.open(reference_path, mode="update") as hdul:
        hdul[0].header["IS_PSF"] = "True"
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"

    for path, target in (
        (science_ta_path, "SCIENCE_TARGET"),
        (reference_ta_path, "REFERENCE_TARGET"),
    ):
        with fits.open(path, mode="update") as hdul:
            header = hdul[0].header
            header["EXP_TYPE"] = ta_exp_type
            header["TARGPROP"] = target

            if header["INSTRUME"] == "MIRI":
                header["CORONMSK"] = "MASK1065"
            else:
                # The reader derives NIRCam CORONMSK from PPS_APER.
                header["PPS_APER"] = "NRCA5_FULL"
                header["CORONMSK"] = "NONE"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data(
        [
            science_path,
            reference_path,
            science_ta_path,
            reference_ta_path,
        ]
    )

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 4

    actual_types = {row["FITSFILE"]: row["TYPE"] for row in observations}
    assert actual_types == {
        science_path: "SCI",
        reference_path: "REF",
        science_ta_path: "SCI_TA",
        reference_ta_path: "REF_TA",
    }


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
def test_read_jwst_s012_data_classifies_by_dither_without_is_psf(
    request, tmp_path, fixture_name, expected_key
):
    """Use NUMDTHPT to classify exposures when IS_PSF is absent.

    Remove IS_PSF from both files. Give science one dither position
    and reference nine positions, triggering the reader's fallback.
    Supply reference first and verify science/reference roles and
    science-before-reference ordering.
    """
    science_path = request.getfixturevalue(fixture_name)
    reference_path = _copy_exposure(science_path, "reference_rateints.fits")

    with fits.open(science_path, mode="update") as hdul:
        del hdul[0].header["IS_PSF"]
        hdul[0].header["NUMDTHPT"] = 1

    with fits.open(reference_path, mode="update") as hdul:
        del hdul[0].header["IS_PSF"]
        hdul[0].header["NUMDTHPT"] = 9
        hdul[0].header["TARGPROP"] = "REFERENCE_TARGET"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([reference_path, science_path])

    assert list(database.obs) == [expected_key]

    observations = database.obs[expected_key]
    assert len(observations) == 2
    assert list(observations["FITSFILE"]) == [science_path, reference_path]
    assert list(observations["TYPE"]) == ["SCI", "REF"]
    assert list(observations["NUMDTHPT"]) == [1, 9]


@pytest.mark.parametrize(
    "suffix, expected_stage",
    [
        ("uncal", "STAGE0"),
        ("rate", "STAGE1"),
        ("rateints", "STAGE1"),
        ("cal", "STAGE2"),
        ("calints", "STAGE2"),
    ],
    ids=["uncal", "rate", "rateints", "cal", "calints"],
)
def test_read_jwst_s012_data_identifies_stage_from_filename(
    tmp_path, miri_s012_file, suffix, expected_stage
):
    """Identify the pipeline stage from each supported filename token.

    Copy the same MIRI exposure under each supported filename.
    Verify that the reader records the expected DATAMODL value
    and retains the supplied file path.

    This tests filename classification, not the image dimensions
    or data-model structure of actual products from each stage.
    """
    path = _copy_exposure(miri_s012_file, f"science_{suffix}.fits")

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([path])

    assert len(database.obs) == 1
    observations = next(iter(database.obs.values()))
    assert len(observations) == 1
    assert observations["FITSFILE"][0] == path
    assert observations["DATAMODL"][0] == expected_stage


@pytest.mark.parametrize(
    "image_shape, expected_nints",
    [
        ((3, 4, 4), 3),
        ((4, 4), 1),
    ],
    ids=["integration-cube", "single-image"],
)
def test_read_jwst_s012_data_infers_missing_nints(
    tmp_path, miri_s012_file, image_shape, expected_nints
):
    """Infer NINTS from SCI dimensions when the keyword is absent.

    A 3-D array uses its first dimension as the integration count.
    A 2-D image defaults to one integration.
    """
    with fits.open(miri_s012_file, mode="update") as hdul:
        del hdul[0].header["NINTS"]
        hdul["SCI"].data = np.zeros(image_shape, dtype=np.float32)

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([miri_s012_file])

    assert len(database.obs) == 1
    observations = next(iter(database.obs.values()))
    assert len(observations) == 1
    assert observations["NINTS"][0] == expected_nints


@pytest.mark.parametrize(
    "fixture_name, original_key, unrelated_key",
    [
        (
            "miri_s012_file",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_MASK1065",
            "JWST_MIRI_MIRIMAGE_F1065C_NONE_4QPM_FULL",
        ),
        (
            "nircam_s012_file",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_SUB320",
            "JWST_NIRCAM_NRCA5_F335M_MASKRND_MASK335R_FULL",
        ),
    ],
    ids=["miri", "nircam"],
)
def test_read_jwst_s012_data_replaces_only_matching_configuration(
    request, tmp_path, fixture_name, original_key, unrelated_key
):
    """Replace an existing configuration without removing other tables.

    Populate the database with two configurations. Read a new file
    matching the first configuration and verify that it replaces
    that table rather than appending a row. Verify the unrelated
    configuration retains its table and contents.
    """
    original_path = request.getfixturevalue(fixture_name)
    unrelated_path = _copy_exposure(original_path, "unrelated_rateints.fits")
    replacement_path = _copy_exposure(original_path, "replacement_rateints.fits")

    with fits.open(unrelated_path, mode="update") as hdul:
        hdul[0].header["SUBARRAY"] = "FULL"

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([original_path, unrelated_path])

    assert set(database.obs) == {original_key, unrelated_key}
    assert list(database.obs[original_key]["FITSFILE"]) == [
        original_path,
    ]

    unrelated_table = database.obs[unrelated_key]
    unrelated_before = unrelated_table.copy(copy_data=True)

    database.read_jwst_s012_data([replacement_path])

    assert set(database.obs) == {original_key, unrelated_key}
    assert list(database.obs[original_key]["FITSFILE"]) == [
        replacement_path,
    ]
    assert database.obs[original_key]["TYPE"][0] == "SCI"

    assert database.obs[unrelated_key] is unrelated_table
    assert unrelated_table.colnames == unrelated_before.colnames

    for column in unrelated_before.colnames:
        np.testing.assert_equal(
            np.asarray(unrelated_table[column]),
            np.asarray(unrelated_before[column]),
            err_msg=f"Unrelated column changed: {column}",
        )
