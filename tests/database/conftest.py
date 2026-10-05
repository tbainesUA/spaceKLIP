import numpy as np
import pytest
from astropy.io import fits


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
