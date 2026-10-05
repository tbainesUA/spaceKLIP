import numpy as np
import pytest
from astropy.io import fits

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
