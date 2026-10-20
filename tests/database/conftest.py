import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import spaceKLIP.database as database_module


@pytest.fixture
def miri_s012_file(tmp_path):
    """Create one small MIRI stage-1 science exposure."""
    path = tmp_path / "miri_science_rateints.fits"

    primary = fits.PrimaryHDU()
    primary.header.update(
        {
            "TELESCOP": "JWST",
            "INSTRUME": "MIRI",
            "DETECTOR": "MIRIMAGE",
            "FILTER": "F1065C",
            "PUPIL": "NONE",
            "CORONMSK": "4QPM",
            "SUBARRAY": "MASK1065",
            "APERNAME": "MIRIM_MASK1065",
            "PPS_APER": "MIRIM_MASK1065",
            "EXP_TYPE": "MIR_4QPM",
            "TARGPROP": "SCIENCE_TARGET",
            "IS_PSF": "False",
            "NINTS": 2,
            "EFFINTTM": 10.0,
            "NUMDTHPT": 1,
        }
    )

    science = fits.ImageHDU(
        data=np.zeros((2, 4, 4), dtype=np.float32),
        name="SCI",
    )
    science.header.update(
        {
            "BUNIT": "DN/s",
            "CRPIX1": 2.5,
            "CRPIX2": 2.5,
        }
    )

    fits.HDUList([primary, science]).writeto(path)

    return str(path)


@pytest.fixture
def nircam_s012_file(tmp_path, monkeypatch):
    """Create one NIRCam science exposure with a local PSF mask."""
    path = tmp_path / "nircam_science_rateints.fits"

    primary = fits.PrimaryHDU()
    primary.header.update(
        {
            "TELESCOP": "JWST",
            "INSTRUME": "NIRCAM",
            "DETECTOR": "NRCA5",
            "FILTER": "F335M",
            "PUPIL": "MASKRND",
            "CORONMSK": "MASK335R",
            "SUBARRAY": "SUB320",
            "APERNAME": "NRCA5_MASK335R",
            "PPS_APER": "NRCA5_MASK335R",
            "EXP_TYPE": "NRC_CORON",
            "TARGPROP": "SCIENCE_TARGET",
            "IS_PSF": "False",
            "NINTS": 2,
            "EFFINTTM": 10.0,
            "NUMDTHPT": 1,
        }
    )

    science = fits.ImageHDU(
        data=np.zeros((2, 4, 4), dtype=np.float32),
        name="SCI",
    )
    science.header.update(
        {
            "BUNIT": "DN/s",
            "CRPIX1": 2.5,
            "CRPIX2": 2.5,
        }
    )

    fits.HDUList([primary, science]).writeto(path)

    mask_path = tmp_path / "nircam_science_rateints_psfmask.fits"
    fits.PrimaryHDU(
        data=np.ones((4, 4), dtype=np.float32),
    ).writeto(mask_path)

    monkeypatch.setattr(
        database_module,
        "nircam_apname",
        lambda header: header["APERNAME"],
    )

    return str(path)


@pytest.fixture(
    params=[
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
def coron3_file(request, tmp_path):
    """Create a CORON3 product and its observation-table companion."""
    fixture_name, key = request.param
    source_path = request.getfixturevalue(fixture_name)
    path = tmp_path / "science_i2d.fits"

    with fits.open(source_path) as hdul:
        primary = hdul[0].copy()

    primary.header["KLMODE0"] = 7

    # Conflicting values verify that CORON3 reads image metadata
    # from SCI rather than from the primary header.
    primary.header["STARCENX"] = -100.0
    primary.header["STARCENY"] = -200.0

    science = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="SCI")
    science.header.update(
        {
            "BUNIT": "MJy/sr",
            "STARCENX": 1.25,
            "STARCENY": 2.75,
        }
    )

    fits.HDUList([primary, science]).writeto(path)

    # Minimal companion schema required by this reader.
    observations = Table(
        {
            "TYPE": ["SCI"],
            "FITSFILE": [source_path],
            "MASKFILE": ["NONE"],
        }
    )
    observations.write(tmp_path / f"{key}.dat", format="ascii")

    return str(path), key, source_path


@pytest.fixture
def pyklip_file(coron3_file, tmp_path):
    """Create a primary-HDU PYKLIP product for either instrument."""
    coron3_path, key, source_path = coron3_file
    path = tmp_path / "science_KLmodes-all.fits"

    with fits.open(coron3_path) as hdul:
        primary = hdul[0].copy()

    primary.data = np.zeros((3, 4, 4), dtype=np.float32)
    primary.header.update(
        {
            "MODE": "ADI+RDI",
            "ANNULI": 3,
            "SUBSECTS": 4,
            "KLMODE0": 1,
            "KLMODE1": 5,
            "KLMODE2": 10,
            "BUNIT": "MJy/sr",
            "STARCENX": 1.5,
            "STARCENY": 2.5,
        }
    )

    fits.HDUList([primary]).writeto(path)

    return str(path), key, source_path
