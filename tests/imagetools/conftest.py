import numpy as np
import pytest
from astropy.table import Table

# @pytest.fixture
# def mock_observation():
#     """Minimal in-memory observation product for ImageTools tests."""

#     nints = 4
#     ny, nx = 5, 5

#     data = np.stack([np.full((ny, nx), i, dtype=float) for i in range(nints)])
#     erro = np.ones((nints, ny, nx), dtype=float)
#     pxdq = np.zeros((nints, ny, nx), dtype=np.uint32)

#     return {
#         "data": data,
#         "erro": erro,
#         "pxdq": pxdq,
#         "head_pri": {"NINTS": nints},
#         "head_sci": {},
#         "is2d": False,
#         "align_shift": None,
#         "center_shift": None,
#         "align_mask": None,
#         "center_mask": None,
#         "maskoffs": None,
#         "mask": np.ones((ny, nx)),
#         "nanmask": np.ones((ny, nx)),
#     }


# @pytest.fixture
# def image_database(tmp_path):
#     """Minimal database state for ImageTools tests."""

#     fitsfile = tmp_path / "example_calints.fits"
#     maskfile = tmp_path / "example_mask.fits"
#     nanmaskfile = tmp_path / "example_nanmask.fits"
#     # obs = Table(
#     #     {
#     #         "FITSFILE": [str(fitsfile)],
#     #         "MASKFILE": [str(maskfile)],
#     #         "NINTS": [4],
#     #         "TYPE": ["SCI"],
#     #     }
#     # )

#     obs = Table(
#         {
#             "FITSFILE": [str(fitsfile)],
#             "MASKFILE": [str(maskfile)],
#             "NANMASKFILE": [str(nanmaskfile)],
#             "NINTS": [4],
#             "EFFINTTM": [10.0],
#             "TYPE": ["SCI"],
#             "CRPIX1": [3.0],
#             "CRPIX2": [3.0],
#             "STARCENX": [2.5],
#             "STARCENY": [2.5],
#             "MASKCENX": [2.5],
#             "MASKCENY": [2.5],
#         }
#     )

#     class MockDatabase:
#         def __init__(self):
#             self.output_dir = str(tmp_path)
#             self.obs = {"concat1": obs}

#         def update_obs(
#             self,
#             key,
#             index,
#             fitsfile,
#             maskfile,
#             **kwargs,
#         ):
#             self.obs[key]["FITSFILE"][index] = fitsfile
#             self.obs[key]["MASKFILE"][index] = maskfile

#             for name, value in kwargs.items():
#                 column = name.upper()

#                 if column in self.obs[key].colnames:
#                     self.obs[key][column][index] = value

#     return MockDatabase()


# @pytest.fixture
# def mock_image_io(monkeypatch, mock_observation, tmp_path):
#     """Mock ImageTools observation I/O and capture written products."""

#     written = {}

#     def mock_read_obs(fitsfile):
#         obs = mock_observation

#         return (
#             obs["data"].copy(),
#             obs["erro"].copy(),
#             obs["pxdq"].copy(),
#             obs["head_pri"].copy(),
#             obs["head_sci"].copy(),
#             obs["is2d"],
#             obs["align_shift"],
#             obs["center_shift"],
#             obs["align_mask"],
#             obs["center_mask"],
#             obs["maskoffs"],
#         )

#     def mock_read_msk(maskfile):
#         if "nanmask" in str(maskfile):
#             return mock_observation["nanmask"].copy()

#         return mock_observation["mask"].copy()

#     def mock_write_obs(
#         fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, **kwargs
#     ):
#         written["data"] = data.copy()
#         written["erro"] = erro.copy()
#         written["pxdq"] = pxdq.copy()
#         written["head_pri"] = head_pri.copy()
#         written["head_sci"] = head_sci.copy()
#         written["is2d"] = is2d
#         written.update(kwargs)

#         return str(tmp_path / "output_calints.fits")

#     def mock_write_msk(maskfile, mask, fitsfile, suffix=None):
#         if suffix == "_nanmask.fits":
#             written["nanmask"] = None if mask is None else mask.copy()
#             return str(tmp_path / "output_nanmask.fits")

#         written["mask"] = None if mask is None else mask.copy()
#         return str(tmp_path / "output_mask.fits")

#     def mock_get_dqmask(pxdq, flag, return_bool=False):
#         assert flag == "DO_NOT_USE"

#         mask = (pxdq & 1) != 0

#         if return_bool:
#             return mask

#         return mask.astype(pxdq.dtype)

#     monkeypatch.setattr("spaceKLIP.imagetools.ut.read_obs", mock_read_obs)
#     monkeypatch.setattr("spaceKLIP.imagetools.ut.read_msk", mock_read_msk)
#     monkeypatch.setattr("spaceKLIP.imagetools.ut.write_obs", mock_write_obs)
#     monkeypatch.setattr("spaceKLIP.imagetools.ut.write_msk", mock_write_msk)
#     monkeypatch.setattr("spaceKLIP.imagetools.ut.get_dqmask", mock_get_dqmask)
#     return written


# Updated for godoy background subtraction


@pytest.fixture
def mock_observation():
    """Minimal in-memory observation product for ImageTools tests."""

    nints = 4
    ny, nx = 8, 10

    data = np.stack([np.full((ny, nx), i, dtype=float) for i in range(nints)])

    erro = np.ones_like(data)

    pxdq = np.zeros(
        data.shape,
        dtype=np.uint32,
    )

    return {
        "data": data,
        "erro": erro,
        "pxdq": pxdq,
        "head_pri": {
            "NINTS": nints,
        },
        "head_sci": {
            "CRPIX1": 3.0,
            "CRPIX2": 3.0,
            "STARCENX": 2.5,
            "STARCENY": 2.5,
            "MASKCENX": 2.5,
            "MASKCENY": 2.5,
        },
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
def make_mock_observation():
    """
    Factory for creating custom in-memory observation products.

    Useful when a test requires multiple FITS files containing different
    science data, such as paired science/background observations.
    """

    def _make(
        data,
        erro=None,
        pxdq=None,
        head_pri=None,
        head_sci=None,
        is2d=False,
        align_shift=None,
        center_shift=None,
        align_mask=None,
        center_mask=None,
        maskoffs=None,
    ):
        data = np.asarray(data, dtype=float)

        if erro is None:
            erro = np.ones_like(data)

        if pxdq is None:
            pxdq = np.zeros(data.shape, dtype=np.uint32)

        return {
            "data": data,
            "erro": np.asarray(erro),
            "pxdq": np.asarray(pxdq),
            "head_pri": ({} if head_pri is None else head_pri),
            "head_sci": ({} if head_sci is None else head_sci),
            "is2d": is2d,
            "align_shift": align_shift,
            "center_shift": center_shift,
            "align_mask": align_mask,
            "center_mask": center_mask,
            "maskoffs": maskoffs,
        }

    return _make


@pytest.fixture
def image_database(tmp_path):
    """Minimal ImageTools database used by general characterization tests."""

    fitsfile = tmp_path / "example_calints.fits"
    maskfile = tmp_path / "example_mask.fits"
    nanmaskfile = tmp_path / "example_nanmask.fits"

    obs = Table(
        {
            "FITSFILE": [str(fitsfile)],
            "MASKFILE": [str(maskfile)],
            "NANMASKFILE": [str(nanmaskfile)],
            "NINTS": [4],
            "EFFINTTM": [10.0],
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
            maskfile=None,
            **kwargs,
        ):
            self.obs[key]["FITSFILE"][index] = fitsfile

            if maskfile is not None and "MASKFILE" in self.obs[key].colnames:
                self.obs[key]["MASKFILE"][index] = maskfile

            for name, value in kwargs.items():
                column = name.upper()

                if column in self.obs[key].colnames:
                    self.obs[key][column][index] = value

    return MockDatabase()


@pytest.fixture
def background_database(tmp_path):
    """
    Minimal SCI + SCI_BG database for standard background-subtraction tests.
    """

    science_file = tmp_path / "science_calints.fits"
    background_file = tmp_path / "science_background_calints.fits"
    science_mask = tmp_path / "science_mask.fits"
    background_mask = tmp_path / "background_mask.fits"

    obs = Table(
        {
            "FITSFILE": [
                str(science_file),
                str(background_file),
            ],
            "MASKFILE": [
                str(science_mask),
                str(background_mask),
            ],
            "TYPE": [
                "SCI",
                "SCI_BG",
            ],
        }
    )

    class MockDatabase:
        def __init__(self):
            self.output_dir = str(tmp_path)
            self.obs = {
                "concat1": obs,
            }

        def update_obs(
            self,
            key,
            index,
            fitsfile,
            maskfile=None,
            **kwargs,
        ):
            self.obs[key]["FITSFILE"][index] = fitsfile

            if maskfile is not None:
                self.obs[key]["MASKFILE"][index] = maskfile

    return MockDatabase()


@pytest.fixture
def mock_image_io(
    monkeypatch,
    mock_observation,
    tmp_path,
):
    """
    Mock ImageTools observation I/O.

    Existing tests use the default ``mock_observation``. Tests requiring
    multiple distinct FITS products may register observations by filename
    using::

        mock_image_io["observations"][fitsfile] = observation

    ``mock_image_io["data"]`` and related entries always describe the most
    recently written observation. ``mock_image_io["writes"]`` contains the
    complete write history.
    """

    written = {}
    observations = {}
    writes = []

    def get_observation(fitsfile):
        return observations.get(
            str(fitsfile),
            mock_observation,
        )

    def copy_optional(value):
        if value is None:
            return None

        return value.copy()

    def mock_read_obs(fitsfile):
        obs = get_observation(fitsfile)

        return (
            obs["data"].copy(),
            obs["erro"].copy(),
            obs["pxdq"].copy(),
            obs["head_pri"].copy(),
            obs["head_sci"].copy(),
            obs["is2d"],
            copy_optional(obs["align_shift"]),
            copy_optional(obs["center_shift"]),
            copy_optional(obs["align_mask"]),
            copy_optional(obs["center_mask"]),
            copy_optional(obs["maskoffs"]),
        )

    def mock_read_msk(maskfile):
        if "nanmask" in str(maskfile):
            return mock_observation["nanmask"].copy()

        return mock_observation["mask"].copy()

    def mock_write_obs(
        fitsfile,
        output_dir,
        data,
        erro,
        pxdq,
        head_pri,
        head_sci,
        is2d,
        **kwargs,
    ):
        record = {
            "input_fitsfile": str(fitsfile),
            "data": data.copy(),
            "erro": erro.copy(),
            "pxdq": pxdq.copy(),
            "head_pri": head_pri.copy(),
            "head_sci": head_sci.copy(),
            "is2d": is2d,
            **kwargs,
        }

        writes.append(record)

        # Preserve the original interface used by existing tests:
        # mock_image_io["data"], ["erro"], ["pxdq"], etc.
        written.update(record)

        return str(tmp_path / "output_calints.fits")

    def mock_write_msk(
        maskfile,
        mask,
        fitsfile,
        suffix=None,
    ):
        if suffix == "_nanmask.fits":
            written["nanmask"] = None if mask is None else mask.copy()

            return str(tmp_path / "output_nanmask.fits")

        written["mask"] = None if mask is None else mask.copy()

        return str(tmp_path / "output_mask.fits")

    def mock_get_dqmask(
        pxdq,
        flag,
        return_bool=False,
    ):
        assert flag == "DO_NOT_USE"

        result = (pxdq & 1) != 0

        if return_bool:
            return result

        return result.astype(pxdq.dtype)

    monkeypatch.setattr("spaceKLIP.imagetools.ut.read_obs", mock_read_obs)

    monkeypatch.setattr("spaceKLIP.imagetools.ut.read_msk", mock_read_msk)

    monkeypatch.setattr("spaceKLIP.imagetools.ut.write_obs", mock_write_obs)

    monkeypatch.setattr("spaceKLIP.imagetools.ut.write_msk", mock_write_msk)

    monkeypatch.setattr("spaceKLIP.imagetools.ut.get_dqmask", mock_get_dqmask)

    # New capabilities while retaining the old fixture interface.
    written["observations"] = observations
    written["writes"] = writes

    return written


@pytest.fixture
def godoy_database(tmp_path):
    """Minimal SCI + SCI_BG database for Godoy background subtraction."""

    science_file = tmp_path / "science_calints.fits"

    background_file = tmp_path / "science_background_calints.fits"

    obs = Table(
        {
            "FITSFILE": [str(science_file), str(background_file)],
            "TYPE": ["SCI", "SCI_BG"],
            "FILTER": ["F1140C", "F1140C"],
        }
    )

    class MockDatabase:
        def __init__(self):
            self.output_dir = str(tmp_path)

            self.obs = {"concat1": obs}

            self.updated = []

        def update_obs(
            self,
            key,
            index,
            fitsfile,
            maskfile=None,
            **kwargs,
        ):
            self.updated.append(
                {
                    "key": key,
                    "index": index,
                    "fitsfile": fitsfile,
                    "maskfile": maskfile,
                    **kwargs,
                }
            )

    return MockDatabase()


@pytest.fixture
def godoy_data():
    """Structured synthetic data for Godoy background-subtraction tests."""

    nints = 2
    ny = 128
    nx = 256

    yy, xx = np.indices((ny, nx))

    background_2d = 10.0 + 0.01 * xx + 0.02 * yy

    background = np.stack(
        [
            background_2d,
            background_2d,
        ]
    )

    science = background.copy()

    # Add a localized source-like signal.
    science[:, 50:70, 100:120] += 25.0

    return {
        "science": science,
        "background": background,
    }
