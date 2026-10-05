import pytest

from spaceKLIP.database import Database


def test_read_jwst_s012_data_rejects_empty_datapaths(tmp_path):
    """Reject empty input before adding observations to the database."""
    database = Database(output_dir=tmp_path)
    database.verbose = False

    with pytest.raises(UserWarning, match="^Could not find any data paths$"):
        database.read_jwst_s012_data([])

    assert database.obs == {}
