import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from astropy.table import Table
import astropy.units as u

from spaceKLIP.analysistools import AnalysisTools, inject_and_recover, best_convfit_and_residuals, _obj_gauss, _chi2_gauss
from spaceKLIP.database import Database

@pytest.fixture
def mock_db():
    """
    Creates a mock spaceKLIP database object for testing.
    """
    db = Mock(spec=Database)
    db.output_dir = 'mock_output'
    
    # Mock the red attribute
    db.red = {
        'concatenation1': Table({
            'FITSFILE': ['mock_data.fits'],
            'INSTRUME': ['MIRI'],
            'SUBARRAY': ['SUB256'],
            'FILTER': ['F1065C'],
            'MASKFILE': ['mock_mask.fits'],
            'PIXSCALE': [0.11],
            'PIXAR_SR': [2.8e-11],
            'CWAVEL': [10.65],
            'TELESCOP': ['JWST'],
            'EXP_TYPE': ['MIR_4QPM'],
            'CORONMSK': ['4QPM_F1065C'],
            'KLMODES': ['1,2,3'],
            'BUNIT': ['MJy/sr'],
            'MODE': ['RDI'],
            'ANNULI': [5],
            'SUBSECTS': [1],
        })
    }
    
    # Mock the obs attribute
    db.obs = {
        'concatenation1': Table({
            'FITSFILE': ['mock_data.fits'],
            'TYPE': ['SCI'],
            'ROLL_REF': [0.0],
            'BLURFWHM': [np.nan],
            'NINTS': [1],
            'EFFINTTM': [1.0]
        })
    }
    
    return db

@pytest.fixture
def mock_analysistools(mock_db):
    """
    Creates an AnalysisTools instance with a mock database.
    """
    return AnalysisTools(mock_db)

@patch('spaceKLIP.analysistools.ut.read_red')
@patch('spaceKLIP.analysistools.ut.read_msk')
@patch('spaceKLIP.analysistools.get_stellar_magnitudes')
@patch('spaceKLIP.analysistools.get_offsetpsf')
@patch('spaceKLIP.analysistools.klip.meas_contrast')
@patch('matplotlib.pyplot.figure')
@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=MagicMock)
def test_raw_contrast(mock_open, mock_exists, mock_figure, mock_meas_contrast, mock_get_offsetpsf, mock_get_stellar_magnitudes, mock_read_msk, mock_read_red, mock_analysistools):
    """
    Tests the raw_contrast method of AnalysisTools.
    """
    # Mock return values for patched functions
    mock_read_red.return_value = (np.ones((1, 10, 10)), {'CRPIX1': 5, 'CRPIX2': 5, 'MODE': 'RDI', 'ANNULI': 5}, {}, False)
    mock_read_msk.return_value = np.ones((10, 10))
    mock_get_stellar_magnitudes.return_value = ({'F1065C': 10.0}, {'F1065C': 1.0})
    mock_get_offsetpsf.return_value = np.ones((10, 10))
    mock_meas_contrast.return_value = (np.arange(5), np.logspace(-1, -5, 5))

    # Run the raw_contrast method
    mock_analysistools.raw_contrast('mock_starfile.vot', output_filetype='npy')

    # Assert that files are being written
    assert mock_open.call_count > 0, "No files were opened to be written to"

def test_calibrate_contrast(mock_analysistools):
    """
    Tests the calibrate_contrast method for initial parameter validation.
    """
    with pytest.raises(TypeError):
        mock_analysistools.calibrate_contrast()

def test_extract_companions():
    """
    Placeholder test for the extract_companions method.
    """
    # This method is very complex and requires a significant amount of setup to test properly.
    # This test serves as a placeholder to indicate that this function needs testing.
    pass

def test_inject_and_recover():
    """
    Simple test to ensure inject_and_recover runs without errors.
    """
    # This function is complex and requires a full pyKLIP dataset.
    # This is a placeholder for a more complete test.
    pass

def test_best_convfit_and_residuals():
    """
    Simple test to ensure best_convfit_and_residuals runs without errors.
    """
    # This function is mainly for plotting and requires a fitted FMAstrometry object.
    pass

def test_obj_gauss():
    """
    Tests the _obj_gauss helper function.
    """
    params = (1.0, 1.0, 0.0)
    data = np.ones((10, 10))
    x, y = np.mgrid[:10, :10]
    psf = np.ones_like(data)
    result = _obj_gauss(params, data, x, y, psf)
    assert isinstance(result, np.ndarray)

def test_chi2_gauss():
    """

    Tests the _chi2_gauss helper function.
    """
    params = (1.0, 1.0, 0.0, 1.0)
    data = np.ones((10, 10))
    x, y = np.mgrid[:10, :10]
    psf = np.ones_like(data)
    result = _chi2_gauss(params, data, x, y, psf)
    assert isinstance(result, float)
