import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import spaceKLIP.database as database_module
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
        for row_index in range(len(unrelated_before)):
            np.testing.assert_equal(
                unrelated_table[column][row_index],
                unrelated_before[column][row_index],
                err_msg=(f"Unrelated column changed: {column}, row: {row_index}"),
            )


def test_read_jwst_s3_data_reads_coron3_product(tmp_path, coron3_file):
    """Read CORON3 metadata into red and load observation context.

    Verify fixed CORON3 reduction settings, explicit KL-mode
    serialization, SCI-header image metadata, and companion-table
    loading for both instruments.
    """
    path, key, source_path = coron3_file

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s3_data([path])

    assert list(database.red) == [key]
    reductions = database.red[key]
    assert isinstance(reductions, Table)
    assert len(reductions) == 1

    row = reductions[0]
    assert row["FITSFILE"] == path
    assert row["TYPE"] == "CORON3"
    assert row["DATAMODL"] == "STAGE3"
    assert row["MODE"] == "RDI"
    assert row["ANNULI"] == 1
    assert row["SUBSECTS"] == 1
    assert row["KLMODES"] == "7"
    assert row["BUNIT"] == "MJy/sr"
    assert row["STARCENX"] == pytest.approx(1.25)
    assert row["STARCENY"] == pytest.approx(2.75)

    assert list(database.obs) == [key]
    assert list(database.obs[key]["FITSFILE"]) == [source_path]
    assert list(database.obs[key]["MASKFILE"]) == ["NONE"]
    assert database.obs[key]["FITSFILE"].dtype == np.dtype("object")
    assert database.obs[key]["MASKFILE"].dtype == np.dtype("object")


def test_read_jwst_s3_data_reads_pyklip_product(tmp_path, pyklip_file):
    """Read PYKLIP settings and image metadata from the primary HDU.

    Verify supplied reduction settings and serialization of three
    consecutive KL-mode keywords into a comma-separated string.
    The input has no SCI extension.
    """
    path, key, source_path = pyklip_file

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s3_data([path])

    assert list(database.red) == [key]
    reductions = database.red[key]
    assert len(reductions) == 1

    row = reductions[0]
    assert row["FITSFILE"] == path
    assert row["TYPE"] == "PYKLIP"
    assert row["DATAMODL"] == "STAGE3"
    assert row["MODE"] == "ADI+RDI"
    assert row["ANNULI"] == 3
    assert row["SUBSECTS"] == 4
    assert row["KLMODES"] == "1,5,10"
    assert row["BUNIT"] == "MJy/sr"
    assert row["STARCENX"] == pytest.approx(1.5)
    assert row["STARCENY"] == pytest.approx(2.5)

    assert list(database.obs) == [key]
    assert list(database.obs[key]["FITSFILE"]) == [source_path]


def test_read_jwst_s3_data_defaults_missing_coron3_klmode(tmp_path, coron3_file):
    """Default CORON3 KLMODES to '50' when KLMODE0 is absent."""
    path, key, _ = coron3_file

    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["KLMODE0"]

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s3_data([path])

    assert list(database.red) == [key]
    reductions = database.red[key]
    assert len(reductions) == 1
    assert reductions["TYPE"][0] == "CORON3"
    assert reductions["KLMODES"][0] == "50"


def test_read_jwst_s3_data_appends_and_preserves_observations(tmp_path, coron3_file):
    """Append a matching reduction and preserve existing obs context.

    Read one CORON3 product, then a second product with the same
    configuration. Remove the companion observation file before
    the second read to verify that existing obs context is reused.
    """
    first_path, key, _ = coron3_file
    second_path = _copy_exposure(first_path, "second_i2d.fits")

    with fits.open(second_path, mode="update") as hdul:
        hdul[0].header["KLMODE0"] = 12

    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s3_data([first_path])

    assert list(database.red[key]["FITSFILE"]) == [first_path]
    original_observations = database.obs[key]

    companion_path = Path(first_path).parent / f"{key}.dat"
    companion_path.unlink()

    database.read_jwst_s3_data([second_path])

    assert list(database.red) == [key]
    reductions = database.red[key]
    assert len(reductions) == 2
    assert list(reductions["FITSFILE"]) == [
        first_path,
        second_path,
    ]
    assert list(reductions["KLMODES"]) == ["7", "12"]
    assert list(reductions["TYPE"]) == ["CORON3", "CORON3"]

    assert list(database.obs) == [key]
    assert database.obs[key] is original_observations


def test_read_jwst_s3_data_rejects_missing_observation_context(tmp_path, coron3_file):
    """Raise when neither existing obs context nor its companion exists."""
    path, key, _ = coron3_file
    companion_path = Path(path).parent / f"{key}.dat"
    companion_path.unlink()

    database = Database(output_dir=tmp_path)
    database.verbose = False

    with pytest.raises(
        UserWarning,
        match="Observations database for concatenation",
    ) as exc_info:
        database.read_jwst_s3_data([path])

    assert key in str(exc_info.value)
    assert key not in database.obs


# read_jwst_s4_data was determined low priority so coverage is deferred


def test_update_obs_updates_required_fields_only(tmp_path, miri_s012_file):
    """Update the selected row's path, stage, and brightness unit.

    Supply no optional updates. Verify all other metadata and
    the neighboring row remain unchanged, even when the new FITS
    header contains different NINTS and EFFINTTM values.


    This checks the full preservation contract without separate assertions for every
    optional column. The cell-by-cell comparison also handles the array-valued columns
    that caused the earlier comparison failure.

    The deliberately different FITS-header NINTS and EFFINTTM values verify that these
    fields are not automatically refreshed when their optional arguments are omitted.

    """
    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([miri_s012_file])

    key = next(iter(database.obs))
    observations = database.obs[key]

    # Add a second row so we can verify index selection.
    observations.add_row(observations[0])
    before = observations.copy(copy_data=True)

    updated_path = _copy_exposure(miri_s012_file, "updated_calints.fits")
    with fits.open(updated_path, mode="update") as hdul:
        hdul["SCI"].header["BUNIT"] = "MJy/sr"
        hdul[0].header["NINTS"] = 9
        hdul[0].header["EFFINTTM"] = 99.0

    database.update_obs(
        key=key,
        index=1,
        fitsfile=updated_path,
    )

    assert list(database.obs) == [key]
    actual = database.obs[key]
    assert len(actual) == 2

    expected = before.copy(copy_data=True)
    expected["FITSFILE"][1] = updated_path
    expected["DATAMODL"][1] = "STAGE2"
    expected["BUNIT"][1] = "MJy/sr"

    assert actual.colnames == expected.colnames

    for column in expected.colnames:
        for row_index in range(len(expected)):
            np.testing.assert_equal(
                actual[column][row_index],
                expected[column][row_index],
                err_msg=(f"Unexpected update: {column}, row {row_index}"),
            )


def test_update_obs_applies_optional_metadata(tmp_path, miri_s012_file):
    """Store supplied scalar, path, and array metadata in the target row.

    Use distinct values to detect swapped column assignments.
    Read from an unchanged FITS file so optional arguments, rather
    than FITS-header values, determine the updated metadata.
    """
    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([miri_s012_file])
    key = next(iter(database.obs))

    # Start with a different value so the update is observable.
    # See fixture.
    database.obs[key]["NINTS"][0] = 1

    updates = {
        "maskfile": str(tmp_path / "new_psfmask.fits"),
        "nanmaskfile": str(tmp_path / "new_nanmask.fits"),
        "nints": 2,
        "effinttm": 25.0,
        "xoffset": 0.1,
        "yoffset": -0.2,
        "crpix1": 3.1,
        "crpix2": 3.2,
        "maskcenx": 4.1,
        "maskceny": 4.2,
        "nanmaskcenx": 5.1,
        "nanmaskceny": 5.2,
        "starcenx": 6.1,
        "starceny": 6.2,
        "crop_shiftx": -1.0,
        "crop_shifty": 2.0,
        "align_shift": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "center_shift": np.array([[1.1, 1.2], [1.3, 1.4]]),
        "align_mask": np.array([2.1, 2.2]),
        "center_mask": np.array([3.1, 3.2]),
        "blurfwhm": 1.5,
    }

    database.update_obs(
        key=key,
        index=0,
        fitsfile=miri_s012_file,
        **updates,
    )

    row = database.obs[key][0]
    for argument, expected_value in updates.items():
        column = argument.upper()
        np.testing.assert_equal(
            row[column],
            expected_value,
            err_msg=f"Optional update failed: {argument}",
        )


def test_update_obs_applies_zero_and_preserves_none(tmp_path, miri_s012_file):
    """Apply explicit zeros while preserving fields passed as None.

    Start with nonzero values so ignored zero updates are detected.
    Also verify that updating CRPIX does not automatically change
    a mask center whose argument is None.
    """
    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([miri_s012_file])
    key = next(iter(database.obs))
    observations = database.obs[key]

    initial_values = {
        "XOFFSET": 1.5,
        "YOFFSET": -2.5,
        "CRPIX1": 3.5,
        "CRPIX2": 4.5,
        "MASKCENX": 5.5,
        "BLURFWHM": 1.5,
    }
    for column, value in initial_values.items():
        observations[column][0] = value

    database.update_obs(
        key=key,
        index=0,
        fitsfile=miri_s012_file,
        xoffset=0.0,
        yoffset=None,
        crpix1=0.0,
        crpix2=None,
        maskcenx=None,
        blurfwhm=0.0,
    )

    row = database.obs[key][0]
    expected_values = {
        "XOFFSET": 0.0,
        "YOFFSET": -2.5,
        "CRPIX1": 0.0,
        "CRPIX2": 4.5,
        "MASKCENX": 5.5,
        "BLURFWHM": 0.0,
    }
    for column, expected_value in expected_values.items():
        assert row[column] == pytest.approx(expected_value), column


@pytest.mark.parametrize(
    "header_pixel_area, expected_pixel_area",
    [
        (2.5e-13, 2.5e-13),
        (None, 1.0e-13),
    ],
    ids=["keyword-present", "keyword-missing"],
)
def test_update_obs_updates_pixel_area_when_available(
    tmp_path,
    miri_s012_file,
    header_pixel_area,
    expected_pixel_area,
):
    """Refresh PIXAR_SR when available; preserve it when missing.

    Start with a known database value. Request a pixel-area refresh
    from the new FITS file and verify either replacement or silent
    preservation when the SCI header lacks PIXAR_SR.
    """
    database = Database(output_dir=tmp_path)
    database.verbose = False
    database.read_jwst_s012_data([miri_s012_file])
    key = next(iter(database.obs))
    database.obs[key]["PIXAR_SR"][0] = 1.0e-13

    updated_path = _copy_exposure(miri_s012_file, "updated_calints.fits")

    with fits.open(updated_path, mode="update") as hdul:
        header = hdul["SCI"].header
        if header_pixel_area is None:
            header.remove("PIXAR_SR", ignore_missing=True)
        else:
            header["PIXAR_SR"] = header_pixel_area

    database.update_obs(
        key=key,
        index=0,
        fitsfile=updated_path,
        update_pxar=True,
    )

    row = database.obs[key][0]
    assert row["FITSFILE"] == updated_path
    assert row["PIXAR_SR"] == pytest.approx(
        expected_pixel_area,
        rel=1e-12,
        abs=0.0,
    )


# Create database user api function


def test_create_database_discovers_files_and_calls_s012_reader(tmp_path, monkeypatch):
    """Discover each observation and forward options to the default reader.

    Keep the real Database constructor but mock file discovery and
    ingestion. Verify flattened, joined paths, forwarded discovery
    filters and reader options, and the returned database settings.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"

    discover = Mock(
        side_effect=[
            ["first_rateints.fits"],
            ["second_rateints.fits"],
        ]
    )
    read_s012 = Mock()

    monkeypatch.setattr(database_module, "get_files", discover)
    monkeypatch.setattr(
        database_module.Database,
        "read_jwst_s012_data",
        read_s012,
    )

    psflibpaths = [str(input_dir / "reference_rateints.fits")]
    bgpaths = [str(input_dir / "second_rateints.fits")]

    database = database_module.create_database(
        output_dir=str(output_dir),
        input_dir=str(input_dir),
        pid=1234,
        obsids=[1, 2],
        psflibpaths=psflibpaths,
        bgpaths=bgpaths,
        assoc_using_targname=False,
        cr_from_siaf=True,
        verbose=False,
        filt="F1065C",
        file_type="rateints.fits",
    )

    assert discover.call_args_list == [
        (
            (str(input_dir),),
            {
                "pid": 1234,
                "obsid": 1,
                "filt": "F1065C",
                "file_type": "rateints.fits",
            },
        ),
        (
            (str(input_dir),),
            {
                "pid": 1234,
                "obsid": 2,
                "filt": "F1065C",
                "file_type": "rateints.fits",
            },
        ),
    ]

    read_s012.assert_called_once_with(
        datapaths=[
            str(input_dir / "first_rateints.fits"),
            str(input_dir / "second_rateints.fits"),
        ],
        psflibpaths=psflibpaths,
        bgpaths=bgpaths,
        cr_from_siaf=True,
        assoc_using_targname=False,
    )

    assert isinstance(database, database_module.Database)
    assert database.output_dir == str(output_dir)
    assert database.verbose is False
    assert output_dir.is_dir()


@pytest.mark.parametrize(
    "readlevel, expected_reader",
    [
        # IDs identify the input's type and value in test output.
        pytest.param("012", "read_jwst_s012_data", id="str-012"),
        pytest.param(0, "read_jwst_s012_data", id="int-0"),
        pytest.param(1, "read_jwst_s012_data", id="int-1"),
        pytest.param(2, "read_jwst_s012_data", id="int-2"),
        pytest.param("0", "read_jwst_s012_data", id="str-0"),
        pytest.param("1", "read_jwst_s012_data", id="str-1"),
        pytest.param("2", "read_jwst_s012_data", id="str-2"),
        pytest.param(3, "read_jwst_s3_data", id="int-3"),
        pytest.param("3", "read_jwst_s3_data", id="str-3"),
        pytest.param(4, "read_jwst_s4_data", id="int-4"),
        pytest.param("4", "read_jwst_s4_data", id="str-4"),
    ],
)
def test_create_database_selects_reader(
    tmp_path, monkeypatch, readlevel, expected_reader
):
    """Dispatch each supported readlevel to exactly one reader.

    Exercise integer and string forms separately because the public
    API accepts both. Case IDs such as 'int-0' and 'str-0' identify
    the input type and value in PyTest output; they do not affect
    test execution.

    Mock discovery and ingestion to isolate reader selection,
    including routing to the deferred stage-4 reader.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    monkeypatch.setattr(database_module, "get_files", Mock(return_value=[]))

    readers = {}
    for name in (
        "read_jwst_s012_data",
        "read_jwst_s3_data",
        "read_jwst_s4_data",
    ):
        readers[name] = Mock()
        monkeypatch.setattr(database_module.Database, name, readers[name])

    database = database_module.create_database(
        output_dir=str(tmp_path / "outputs"),
        input_dir=str(input_dir),
        readlevel=readlevel,
        cr_from_siaf=True,
        verbose=False,
    )

    expected_kwargs = {"datapaths": []}
    if expected_reader == "read_jwst_s012_data":
        expected_kwargs.update(
            {
                "psflibpaths": None,
                "bgpaths": None,
                "cr_from_siaf": True,
                "assoc_using_targname": True,
            }
        )
    elif expected_reader == "read_jwst_s3_data":
        expected_kwargs["cr_from_siaf"] = True

    for name, reader in readers.items():
        if name == expected_reader:
            reader.assert_called_once_with(**expected_kwargs)
        else:
            reader.assert_not_called()

    assert isinstance(database, database_module.Database)


def test_create_database_includes_sorted_klip_products(tmp_path, monkeypatch):
    """Combine discovered files with sorted stage-3 KLIP products.

    Preserve get_files ordering, then append matching KLIP paths
    in alphabetical order. Exclude files with other endings.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    # Create in reverse order so the expected result requires sorting.
    for filename in (
        "z_KLmodes-all.fits",
        "a_KLmodes-all.fits",
        "ignored_KLmodes.fits",
    ):
        (input_dir / filename).touch()

    discover = Mock(return_value=["science_i2d.fits"])
    read_s3 = Mock()

    monkeypatch.setattr(database_module, "get_files", discover)
    monkeypatch.setattr(
        database_module.Database,
        "read_jwst_s3_data",
        read_s3,
    )

    database = database_module.create_database(
        output_dir=str(tmp_path / "outputs"),
        input_dir=str(input_dir),
        readlevel=3,
        cr_from_siaf=True,
        verbose=False,
    )

    discover.assert_called_once_with(
        str(input_dir),
        pid=None,
        obsid=None,
    )
    read_s3.assert_called_once_with(
        datapaths=[
            str(input_dir / "science_i2d.fits"),
            str(input_dir / "a_KLmodes-all.fits"),
            str(input_dir / "z_KLmodes-all.fits"),
        ],
        cr_from_siaf=True,
    )
    assert isinstance(database, database_module.Database)


def test_create_database_resolves_input_directory_from_pid(tmp_path, monkeypatch):
    """Resolve a five-digit program directory from JWSTDOWNLOAD_OUTDIR.

    Omit input_dir and supply a scalar observation ID. Verify
    discovery receives the resolved directory and observation ID,
    and the reader receives the fully joined file path.
    """
    download_dir = tmp_path / "downloads"
    program_dir = download_dir / "01234"
    program_dir.mkdir(parents=True)

    monkeypatch.setenv("JWSTDOWNLOAD_OUTDIR", str(download_dir))

    discover = Mock(return_value=["science_rateints.fits"])
    read_s012 = Mock()

    monkeypatch.setattr(database_module, "get_files", discover)
    monkeypatch.setattr(
        database_module.Database,
        "read_jwst_s012_data",
        read_s012,
    )

    database = database_module.create_database(
        output_dir=str(tmp_path / "outputs"),
        pid=1234,
        obsids=7,
        verbose=False,
    )

    discover.assert_called_once_with(
        str(program_dir),
        pid=1234,
        obsid=7,
    )
    read_s012.assert_called_once_with(
        datapaths=[str(program_dir / "science_rateints.fits")],
        psflibpaths=None,
        bgpaths=None,
        cr_from_siaf=False,
        assoc_using_targname=True,
    )
    assert isinstance(database, database_module.Database)


@pytest.mark.parametrize(
    "scenario, expected_message",
    [
        pytest.param(
            "missing-input",
            "Must provide either a pid or an input_dir",
            id="missing-input-location",
        ),
        pytest.param(
            "invalid-readlevel",
            "Invalid/unknown value for readlevel parameter",
            id="unsupported-readlevel",
        ),
    ],
)
def test_create_database_rejects_invalid_inputs(
    tmp_path, monkeypatch, scenario, expected_message
):
    """Reject missing input location and an unsupported read level.

    Mock discovery and readers to isolate factory validation.
    Verify that neither invalid request invokes an ingestion reader.
    """
    discover = Mock(return_value=[])
    monkeypatch.setattr(database_module, "get_files", discover)

    readers = []
    for name in (
        "read_jwst_s012_data",
        "read_jwst_s3_data",
        "read_jwst_s4_data",
    ):
        reader = Mock()
        monkeypatch.setattr(database_module.Database, name, reader)
        readers.append(reader)

    kwargs = {
        "output_dir": str(tmp_path / "outputs"),
        "verbose": False,
    }

    if scenario == "invalid-readlevel":
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        kwargs.update(
            {
                "input_dir": str(input_dir),
                "readlevel": 5,
            }
        )

    with pytest.raises(ValueError) as exc_info:
        database_module.create_database(**kwargs)

    assert str(exc_info.value) == expected_message

    for reader in readers:
        reader.assert_not_called()

    if scenario == "missing-input":
        discover.assert_not_called()
