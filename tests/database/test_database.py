import pytest
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
