import numpy as np
import pytest
from astropy.table import Table


@pytest.fixture
def mock_observation():
    """Minimal in-memory observation product for ImageTools tests."""

    nints = 4
    ny, nx = 5, 5

    data = np.stack([np.full((ny, nx), i, dtype=float) for i in range(nints)])
    erro = np.ones((nints, ny, nx), dtype=float)
    pxdq = np.zeros((nints, ny, nx), dtype=np.uint32)

    return {
        "data": data,
        "erro": erro,
        "pxdq": pxdq,
        "head_pri": {"NINTS": nints},
        "head_sci": {},
        "is2d": False,
        "align_shift": None,
        "center_shift": None,
        "align_mask": None,
        "center_mask": None,
        "maskoffs": None,
        "mask": np.ones((ny, nx)),
        "nanmask": np.ones((ny, nx)),
    }


@pytest.fixture
def image_database(tmp_path):
    """Minimal database state for ImageTools tests."""

    fitsfile = tmp_path / "example_calints.fits"
    maskfile = tmp_path / "example_mask.fits"
    nanmaskfile = tmp_path / "example_nanmask.fits"
    # obs = Table(
    #     {
    #         "FITSFILE": [str(fitsfile)],
    #         "MASKFILE": [str(maskfile)],
    #         "NINTS": [4],
    #         "TYPE": ["SCI"],
    #     }
    # )

    obs = Table(
        {
            "FITSFILE": [str(fitsfile)],
            "MASKFILE": [str(maskfile)],
            "NANMASKFILE": [str(nanmaskfile)],
            "NINTS": [4],
            "TYPE": ["SCI"],
            "CRPIX1": [3.0],
            "CRPIX2": [3.0],
            "STARCENX": [2.5],
            "STARCENY": [2.5],
            "MASKCENX": [2.5],
            "MASKCENY": [2.5],
        }
    )

    class MockDatabase:
        def __init__(self):
            self.output_dir = str(tmp_path)
            self.obs = {"concat1": obs}

        def update_obs(
            self,
            key,
            index,
            fitsfile,
            maskfile,
            **kwargs,
        ):
            self.obs[key]["FITSFILE"][index] = fitsfile
            self.obs[key]["MASKFILE"][index] = maskfile

            for name, value in kwargs.items():
                column = name.upper()

                if column in self.obs[key].colnames:
                    self.obs[key][column][index] = value

    return MockDatabase()


@pytest.fixture
def mock_image_io(monkeypatch, mock_observation, tmp_path):
    """Mock ImageTools observation I/O and capture written products."""

    written = {}

    def mock_read_obs(fitsfile):
        obs = mock_observation

        return (
            obs["data"].copy(),
            obs["erro"].copy(),
            obs["pxdq"].copy(),
            obs["head_pri"].copy(),
            obs["head_sci"].copy(),
            obs["is2d"],
            obs["align_shift"],
            obs["center_shift"],
            obs["align_mask"],
            obs["center_mask"],
            obs["maskoffs"],
        )

    def mock_read_msk(maskfile):
        if "nanmask" in str(maskfile):
            return mock_observation["nanmask"].copy()

        return mock_observation["mask"].copy()

    def mock_write_obs(
        fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, **kwargs
    ):
        written["data"] = data.copy()
        written["erro"] = erro.copy()
        written["pxdq"] = pxdq.copy()
        written["head_pri"] = head_pri.copy()
        written["head_sci"] = head_sci.copy()
        written["is2d"] = is2d
        written.update(kwargs)

        return str(tmp_path / "output_calints.fits")

    def mock_write_msk(maskfile, mask, fitsfile):
        written["mask"] = mask.copy()
        return str(tmp_path / "output_mask.fits")

    monkeypatch.setattr("spaceKLIP.imagetools.ut.read_obs", mock_read_obs)
    monkeypatch.setattr("spaceKLIP.imagetools.ut.read_msk", mock_read_msk)
    monkeypatch.setattr("spaceKLIP.imagetools.ut.write_obs", mock_write_obs)
    monkeypatch.setattr("spaceKLIP.imagetools.ut.write_msk", mock_write_msk)

    return written
