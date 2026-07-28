from __future__ import division

import copy
import logging

# =============================================================================
# IMPORTS
# =============================================================================
import sys
from functools import partial
from io import StringIO

import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyklip.fakes as fakes
import scipy.ndimage.interpolation as sinterp
from pyklip import parallelized
from scipy.ndimage import convolve
from scipy.optimize import minimize
from tqdm.auto import trange

from spaceKLIP.imagetools import gaussian_kernel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


#
# Helpers
#

from contrast.calibrate import calibrate_contrast
from contrast.extract_companions import extract_companions
from contrast.raw import raw_contrast

# # =============================================================================
# # DOMAIN CONSTANTS
# # =============================================================================

# MIRI_4QPM_ROTATION_OFFSET_DEG: Final[float] = 4.83544897

# MIRI_4QPM_WIDE_BAR_WIDTH_PIX: Final[int] = 10
# MIRI_4QPM_THIN_BAR_WIDTH_PIX: Final[int] = 2
# MIRI_4QPM_INNER_CLEAR_RADIUS_PIX: Final[int] = 15

# MIRI_ROTATION_REFERENCE_DEG: Final[float] = 90.0

# MASK_OCCUPANCY_THRESHOLD: Final[float] = 0.5


# # =============================================================================
# # LOW-LEVEL GEOMETRIC KERNELS
# # =============================================================================


# def rotate_coordinates(
#     x_coords: np.ndarray,
#     y_coords: np.ndarray,
#     rotation_angles_deg: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Rotate detector coordinates using broadcasted rotation matrices.

#     Returns
#     -------
#     x_rotated, y_rotated
#         Arrays with shape (n_rolls, ny, nx)
#     """
#     theta_rad = np.deg2rad(rotation_angles_deg)

#     cos_theta = np.cos(theta_rad)[:, None, None]
#     sin_theta = np.sin(theta_rad)[:, None, None]

#     x_rotated = x_coords[None, :, :] * cos_theta - y_coords[None, :, :] * sin_theta

#     y_rotated = x_coords[None, :, :] * sin_theta + y_coords[None, :, :] * cos_theta

#     return x_rotated, y_rotated


# def evaluate_cross_occultation(
#     x_coords_pix: np.ndarray,
#     y_coords_pix: np.ndarray,
#     wide_bar_width_pix: float,
#     thin_bar_width_pix: float,
#     inner_clear_radius_pix: float,
# ) -> np.ndarray:
#     """
#     Evaluate analytic 4QPM occultation geometry.
#     """
#     half_wide = wide_bar_width_pix / 2.0
#     half_thin = thin_bar_width_pix / 2.0

#     radius_squared = x_coords_pix**2 + y_coords_pix**2

#     central_clear_region = radius_squared < inner_clear_radius_pix**2

#     vertical_wide = np.abs(x_coords_pix) <= half_wide
#     horizontal_wide = np.abs(y_coords_pix) <= half_wide

#     vertical_thin = np.abs(x_coords_pix) <= half_thin
#     horizontal_thin = np.abs(y_coords_pix) <= half_thin

#     occulted_region = (vertical_wide | horizontal_wide) & ~central_clear_region

#     structural_cross = vertical_thin | horizontal_thin

#     return occulted_region | structural_cross


# # =============================================================================
# # HIGH-LEVEL DOMAIN API
# # =============================================================================


# def generate_miri_4qpm_mask(
#     detector_shape: Tuple[int, int],
#     coronagraph_center_pix: Tuple[float, float],
#     roll_reference_angles_deg: np.ndarray,
#     upsample_factor: int = 1,
#     padding_pix: int = 0,
# ) -> np.ndarray:
#     """
#     Generate the MIRI 4QPM occultation mask analytically.

#     Parameters
#     ----------
#     detector_shape
#         Detector shape as (ny, nx).

#     coronagraph_center_pix
#         Coronagraph center coordinates as (x_center, y_center).

#     roll_reference_angles_deg
#         Telescope roll angles in degrees.

#     upsample_factor
#         Integer detector upsampling factor.

#     Returns
#     -------
#     np.ndarray
#         Float mask where occulted pixels are NaN.
#     """
#     if upsample_factor < 1:
#         raise ValueError("upsample_factor must be >= 1")

#     if np.any(~np.isfinite(roll_reference_angles_deg)):
#         raise ValueError("roll_reference_angles_deg contains NaN or Inf values.")

#     ny, nx = detector_shape

#     # Here we're preserving the padding and upsampling
#     padded_ny = ny + 2 * padding_pix
#     padded_nx = nx + 2 * padding_pix

#     x_center_pix, y_center_pix = coronagraph_center_pix
#     x_center_pix += padding_pix
#     y_center_pix += padding_pix

#     sampled_ny = padded_ny * upsample_factor
#     sampled_nx = padded_nx * upsample_factor

#     y_coords_pix = (
#         np.arange(sampled_ny, dtype=np.float32) / upsample_factor - y_center_pix
#     )

#     x_coords_pix = (
#         np.arange(sampled_nx, dtype=np.float32) / upsample_factor - x_center_pix
#     )

#     x_grid_pix, y_grid_pix = np.meshgrid(
#         x_coords_pix,
#         y_coords_pix,
#         indexing="xy",
#         sparse=False,
#     )

#     rotation_angles_deg = (
#         MIRI_ROTATION_REFERENCE_DEG
#         - roll_reference_angles_deg
#         + MIRI_4QPM_ROTATION_OFFSET_DEG
#     )

#     x_rotated_pix, y_rotated_pix = rotate_coordinates(
#         x_grid_pix,
#         y_grid_pix,
#         rotation_angles_deg,
#     )

#     occultation_stack = evaluate_cross_occultation(
#         x_coords_pix=x_rotated_pix,
#         y_coords_pix=y_rotated_pix,
#         wide_bar_width_pix=MIRI_4QPM_WIDE_BAR_WIDTH_PIX,
#         thin_bar_width_pix=MIRI_4QPM_THIN_BAR_WIDTH_PIX,
#         inner_clear_radius_pix=MIRI_4QPM_INNER_CLEAR_RADIUS_PIX,
#     )

#     combined_occultation = np.any(occultation_stack, axis=0)

#     if upsample_factor > 1:
#         combined_occultation = combined_occultation[
#             ::upsample_factor,
#             ::upsample_factor,
#         ]

#     combined_occultation = combined_occultation[
#         padding_pix : padding_pix + ny,
#         padding_pix : padding_pix + nx,
#     ]

#     final_mask = np.ones(
#         detector_shape,
#         dtype=np.float32,
#     )

#     final_mask[combined_occultation] = np.nan

#     return set_surrounded_pixels(final_mask)


# # =============================================================================
# # HIGH-LEVEL Contrast calculation API
# # =============================================================================
# from typing import Optional


# def calc_single_contrast_curve(
#     normalized_frame: np.ndarray,
#     spatial_resolution_pix: float,
#     center_pix: tuple[float, float],
#     inner_working_angle_pix: int | float,
#     outer_working_angle_pix: int | float,
#     coronagraph_transmission_mask: Optional[np.ndarray] = None,
#     low_pass_filter: bool = False,
# ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#     """Compute raw and throughput-corrected contrast curve for single detector frame"""

#     separations_pix, raw_contrast = klip.meas_contrast(
#         dat=normalized_frame,
#         center=center_pix,
#         iwa=inner_working_angle_pix,
#         owa=outer_working_angle_pix,
#         resolution=spatial_resolution_pix,
#         low_pass_filter=low_pass_filter,
#     )

#     corrected_contrast = None
#     if coronagraph_transmission_mask is not None:
#         corrected_frame = normalized_frame / coronagraph_transmission_mask

#         _, corrected_contrast = klip.meas_contrast(
#             dat=corrected_frame,
#             center=center_pix,
#             iwa=inner_working_angle_pix,
#             owa=outer_working_angle_pix,
#             resolution=spatial_resolution_pix,
#             low_pass_filter=low_pass_filter,
#         )

#     return (separations_pix, raw_contrast, corrected_contrast)


# MIN_CORONAGRAPH_TRANSMISSION = 1e-12


# def normalize_contrast_frame(
#     frame_data: np.ndarray,
#     normalization_factor: float,
# ) -> np.ndarray:
#     """
#     Normalize detector frame into contrast units.
#     """
#     return frame_data * normalization_factor


# def apply_throughput_correction(
#     normalized_frame: np.ndarray,
#     transmission_mask: np.ndarray,
#     minimum_transmission: float = MIN_CORONAGRAPH_TRANSMISSION,
# ) -> np.ndarray:
#     """
#     Apply bounded coronagraph throughput correction.

#     Pixels below minimum transmission are masked to NaN.
#     """
#     corrected_frame = np.full_like(
#         normalized_frame,
#         np.nan,
#         dtype=np.float32,
#     )

#     valid_transmission = transmission_mask >= minimum_transmission

#     np.divide(
#         normalized_frame,
#         transmission_mask,
#         out=corrected_frame,
#         where=valid_transmission,
#     )

#     return corrected_frame


# def compute_contrast_curves(
#     data_cube: np.ndarray,
#     pixel_area_sr: float,
#     stellar_flux_peak: float,
#     spatial_resolution_pix: float,
#     center_pix: Tuple[float, float],
#     inner_working_angle_pix: int | float = 1,
#     outer_working_angle_pix: Optional[int | float] = None,
#     coronagraph_transmission_mask: Optional[np.ndarray] = None,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     Optional[np.ndarray],
# ]:
#     """
#     Compute raw and throughput-corrected contrast curves.

#     Parameters
#     ----------
#     data_cube
#         Input detector cube with shape (n_frames, ny, nx).

#     pixel_area_sr
#         Pixel solid angle in steradians.

#     stellar_flux_peak
#         Stellar peak normalization flux.

#     spatial_resolution_pix
#         Resolution element diameter in pixels.

#     center_pix
#         PSF center coordinates as (x, y).

#     inner_working_angle_pix
#         Inner working angle in pixels.

#     outer_working_angle_pix
#         Outer working angle in pixels.

#     coronagraph_transmission_mask
#         Optional coronagraph throughput transmission map.

#     Returns
#     -------
#     separations_pix
#         Radial separations in pixels.

#     raw_contrast_curves
#         Raw 5-sigma contrast curves.

#     throughput_corrected_contrast_curves
#         Throughput-corrected contrast curves.
#     """
#     if data_cube.ndim != 3:
#         raise ValueError("data_cube must have shape (n_frames, ny, nx)")

#     if stellar_flux_peak <= 0:
#         raise ValueError("stellar_flux_peak must be positive.")

#     if coronagraph_transmission_mask is not None:
#         if coronagraph_transmission_mask.shape != data_cube.shape[1:]:
#             raise ValueError("coronagraph_transmission_mask shape mismatch.")

#     n_frames = data_cube.shape[0]

#     if outer_working_angle_pix is None:
#         outer_working_angle_pix = min(data_cube.shape[1:]) // 2

#     normalization_factor = pixel_area_sr / stellar_flux_peak

#     # Precompute normalized cube
#     normalized_cube = (data_cube * normalization_factor).astype(np.float32, copy=False)

#     raw_contrast_curves = []
#     throughput_corrected_curves = []

#     radial_separations_pix = None

#     contrast_kwargs = dict(
#         inner_working_angle_pix=inner_working_angle_pix,
#         outer_working_angle_pix=outer_working_angle_pix,
#         spatial_resolution_pix=spatial_resolution_pix,
#         center_pix=center_pix,
#         low_pass_filter=False,
#         coronagraph_transmission_mask=coronagraph_transmission_mask,
#     )

#     for normalized_frame in normalized_cube:
#         (
#             separations_pix,
#             raw_contrast,
#             corrected_contrast,
#         ) = calc_single_contrast_curve(
#             normalized_frame=normalized_frame, **contrast_kwargs
#         )
#         # shared radial separations
#         if radial_separations_pix is None:
#             radial_separations_pix = separations_pix

#         raw_contrast_curves.append(raw_contrast)
#         throughput_corrected_curves.append(corrected_contrast)

#     # convert to arrays
#     raw_contrast_curves = np.asarray(
#         raw_contrast_curves,
#         dtype=np.float32,
#     )

#     throughput_corrected_array = None

#     if coronagraph_transmission_mask is not None:
#         throughput_corrected_array = np.asarray(
#             throughput_corrected_curves,
#             dtype=np.float32,
#         )

#     return (
#         radial_separations_pix,
#         raw_contrast_curves,
#         throughput_corrected_array,
#     )


# # =============================================================================
# # Adding in the companion masks now
# # =============================================================================
# from typing import List, Tuple

# import numpy as np
# from numpy.typing import NDArray

# # =============================================================================
# # ATOMIC MATHEMATICAL KERNEL (Pure Point Logic)
# # =============================================================================


# def compute_single_companion_mask(
#     x_grid: NDArray[np.float64],
#     y_grid: NDArray[np.float64],
#     center_pix: Tuple[float, float],
#     pixscale_arcsec: float,
#     spatial_resolution_pix: float,
#     ra_offset_arcsec: float,
#     dec_offset_arcsec: float,
#     mask_radius_ld: float,
# ) -> NDArray[np.bool_]:
#     """
#     Compute a 2D boolean exclusion mask for a single celestial companion.

#     This function acts as a pure stateless coordinate-space kernel. It does
#     not know about data frames, arrays of companions, or file paths.

#     Parameters
#     ----------
#     x_grid : NDArray[np.float64]
#         2D matrix containing the horizontal (X) pixel coordinates.
#     y_grid : NDArray[np.float64]
#         2D matrix containing the vertical (Y) pixel coordinates.
#     center_pix : Tuple[float, float]
#         0-indexed host star coordinate array (X_center, Y_center).
#     pixscale_arcsec : float
#         Pixel scale of the instrument detector in arcseconds/pixel.
#     spatial_resolution_pix : float
#         Calculated resolution element size (lambda/D) expressed in pixels.
#     ra_offset_arcsec : float
#         Right Ascension angular offset relative to host star in arcseconds.
#     dec_offset_arcsec : float
#         Declination angular offset relative to host star in arcseconds.
#     mask_radius_ld : float
#         Exclusion zone cutoff radius in units of lambda/D.

#     Returns
#     -------
#     NDArray[np.bool_]
#         A 2D boolean mask of shape matching the input grid, where True
#         denotes coordinates inside the companion exclusion boundary.
#     """
#     # Transform physical angular coordinates into raw pixel displacements
#     delta_x_pix = ra_offset_arcsec / pixscale_arcsec
#     delta_y_pix = dec_offset_arcsec / pixscale_arcsec
#     cutoff_radius_pix = mask_radius_ld * spatial_resolution_pix

#     # Calculate radial Euclidean distance from the companion's shifted center
#     radial_distance_map = np.sqrt(
#         (x_grid - center_pix[0] + delta_x_pix) ** 2
#         + (y_grid - center_pix[1] - delta_y_pix) ** 2
#     )

#     return radial_distance_map <= cutoff_radius_pix


# # =============================================================================
# # GEOMETRIC EXCLUSION FACTORY (State Aggregator Layer)
# # =============================================================================


# def generate_companion_spatial_mask(
#     spatial_shape: Tuple[int, int],
#     center_pix: Tuple[float, float],
#     pixscale_arcsec: float,
#     spatial_resolution_pix: float,
#     companions: List[List[float]],
# ) -> NDArray[np.bool_]:
#     """
#     Generate a unified 2D binary mask indicating multi-companion exclusion zones.

#     Parameters
#     ----------
#     spatial_shape : Tuple[int, int]
#         The (Y, X) pixel dimensions of the target frame layout.
#     center_pix : Tuple[float, float]
#         0-indexed target coordinate array (X_center, Y_center).
#     pixscale_arcsec : float
#         Pixel scale of the instrument detector in arcseconds/pixel.
#     spatial_resolution_pix : float
#         Calculated resolution element size (lambda/D) expressed in pixels.
#     companions : List[List[float]]
#         Matrix of target companions where each row represents
#         [RA_offset, Dec_offset, radius_ld].

#     Returns
#     -------
#     NDArray[np.bool_]
#         2D unified spatial boolean mask layout.
#     """
#     ny, nx = spatial_shape
#     y_indices, x_indices = np.indices((ny, nx), dtype=np.float64)

#     # Initialize a master empty mask
#     combined_mask = np.zeros((ny, nx), dtype=bool)

#     # Map the companion list across our mathematical kernel
#     for companion in companions:
#         ra_off, dec_off, radius_ld = companion

#         single_mask = compute_single_companion_mask(
#             x_grid=x_indices,
#             y_grid=y_indices,
#             center_pix=center_pix,
#             pixscale_arcsec=pixscale_arcsec,
#             spatial_resolution_pix=spatial_resolution_pix,
#             ra_offset_arcsec=ra_off,
#             dec_offset_arcsec=dec_off,
#             mask_radius_ld=radius_ld,
#         )

#         # Accumulate the mask footprint using bitwise OR
#         combined_mask |= single_mask

#     return combined_mask


# # =============================================================================
# # math
# # =============================================================================
# ARCSEC_PER_RADIAN = 180.0 * 3600.0 / np.pi


# def calculate_pixel_area_sr(pixel_scale_arcsec: float) -> float:
#     """
#     Calculate the solid angle subtended by a single detector pixel in steradians.

#     Parameters
#     ----------
#     pixel_scale_arcsec : float
#         The physical size of a single detector pixel in arcseconds.

#     Returns
#     -------
#     float
#         The solid angle area equivalent expressed in units of steradians (sr).
#     """
#     if pixel_scale_arcsec <= 0:
#         raise ValueError("pixel_scale_arcsec must be a positive non-zero value.")

#     # Convert linear pixel edge dimension from arcseconds to radians
#     pixel_scale_rad = pixel_scale_arcsec / ARCSEC_PER_RADIAN

#     # Compute 2D area component (square radians is structurally equivalent to steradians)
#     pixel_area_sr = pixel_scale_rad**2

#     return float(pixel_area_sr)


# # Official JWST Entrance Pupil/Primary Mirror Diameter Constants (Meters)
# # JWST_CIRCUMSCRIBED_DIAMETER = 6.6  # Outer edge boundary
# MICRON_TO_METERS = 1e-6
# JWST_CORONAGRAPH_DIAMETER = (
#     5.2  # Effective clearance diameter for specialized coronagraph masking
# )


# def calculate_spatial_resolution_pix(
#     wavelength_um: float,
#     pixel_scale_rad: float,
#     telescope_name: str,
#     exposure_type: str,
#     blur_fwhm_pix: float = 0.0,
#     factor: float = 1.0,
# ) -> float:
#     """
#     Calculate the effective structural resolution element in pixels, accounting
#     for optical diffraction limitations and instrument-level blurring.

#     Parameters
#     ----------
#     wavelength_um : float
#         The observation center wavelength in micrometers (CWAVEL).
#     pixel_scale_rad : float
#         The pixel scale size expressed in radians per pixel (pxsc_rad).
#     telescope_name : str
#         Name identifier of the origin observatory (e.g., 'JWST').
#     exposure_type : str
#         Telemetry exposure type block code (e.g., 'NRC_CORON').
#     blur_fwhm_pix : float, optional
#         Additional atmospheric/instrument blur FWHM in pixel units (BLURFWHM). Defaults to 0.0.
#     factor : float, optional
#         Diffraction element scaling modifier. Defaults to 1.0 (lambda/D). Change to 1.22
#         if strictly computing a Rayleigh criteria resolution bound.

#     Returns
#     -------
#     float
#         The effective physical resolution element scaled into detector pixels.
#     """
#     # 1. Enforce strict telescope type gating
#     if str(telescope_name).upper() != "JWST":
#         raise ValueError(
#             f"Unsupported or unknown telescope platform: '{telescope_name}'"
#         )

#     if wavelength_um <= 0 or pixel_scale_rad <= 0:
#         raise ValueError(
#             "Wavelength and pixel scale inputs must evaluate to positive non-zero floats."
#         )

#     # 2. Assign effective optical diameter based on specialized pupil masking profiles
#     if str(exposure_type).upper() in ["NRC_CORON", "NRC_TACONFIRM", "NRC_TACQ"]:
#         # NIRCam coronagraph masks constrain the open entrance pupil to an effective 5.2m
#         aperture_diameter_m = JWST_CORONAGRAPH_DIAMETER
#     else:
#         aperture_diameter_m = JWST_CIRCUMSCRIBED_DIAMETER

#     # 3. Compute structural diffraction footprint (meters to micrometers cancel via 1e-6)
#     wavelength_m = wavelength_um * 1e-6  # MICRON_TO_METERS
#     diffraction_limit_rad = (factor * wavelength_m) / aperture_diameter_m

#     # Map the spatial angular bounds directly into detector pixel grids
#     spatial_resolution_pix = diffraction_limit_rad / pixel_scale_rad

#     # 4. Apply additional instrumental/smearing blurring adjustments via quadrature addition
#     if not np.isnan(blur_fwhm_pix) and blur_fwhm_pix > 0:
#         spatial_resolution_pix = np.hypot(spatial_resolution_pix, blur_fwhm_pix)

#     return float(spatial_resolution_pix)


# # =============================================================================
# # MAIN
# # =============================================================================


# def validate_companions(companions):
#     """Validation check to ensure that a list of companions (objects) have
#     3 elements (ra, dec, size lambda/D units) want a list of lists"""
#     if companions is None:
#         return None

#     if not companions:
#         return []

#     if not isinstance(companions[0], (list, tuple)):
#         companions = [companions]

#     if any(len(c) != 3 for c in companions):
#         raise ValueError("Each companion must contain exactly 3 elements")

#     return companions


class AnalysisTools:
    """
    The spaceKLIP astrophysical analysis tools class.

    """

    def __init__(self, database):
        """
        Initialize the spaceKLIP astrophysical analysis tools class.

        Parameters
        ----------
        database : spaceKLIP.Database
            SpaceKLIP database on which the astrophysical analysis steps shall
            be run.

        Returns
        -------
        None.

        """

        # Make an internal alias of the spaceKLIP database class.
        print("Hello")
        self.database = database

        pass

    def raw_contrast(
        self,
        starfile,
        spectral_type="G2V",
        companions=None,
        overwrite_crpix=None,
        subdir="rawcon",
        output_filetype="npy",
        plot_xlim=(0, 10),
        save_figures=True,
        plot_style=None,
        **kwargs,
    ):
        return raw_contrast(
            self.database,
            starfile,
            spectral_type="G2V",
            companions=None,
            overwrite_crpix=None,
            subdir="rawcon",
            output_filetype="npy",
            plot_xlim=(0, 10),
            save_figures=True,
            plot_style=None,
            **kwargs,
        )

    def calibrate_contrast(
        self,
        subdir="calcon",
        rawcon_subdir="rawcon",
        rawcon_filetype="npy",
        companions=None,
        injection_seps="default",
        injection_pas="default",
        injection_flux_sigma=20,
        multi_injection_spacing=None,
        use_saved=False,
        thrput_fit_method="median",
        plot_xlim=(0, 10),
        plot_style=None,
        **kwargs,
    ):
        """
        Compute a calibrated contrast curve relative to the host star flux.

        Parameters
        ----------
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'calcon'.
        rawcon_subdir : str, optional
            Name of the directory where the raw contrast data products have been
            saved. The default is 'rawcon'.
        rawcon_filetype : str
            Save filetype of the raw contrast files.
        companions : list of list of three float, optional
            List of companions to be masked before computing the raw contrast.
            For each companion, there should be a three element list containing
            [RA offset (arcsec), Dec offset (arcsec), mask radius (lambda/D)].
            The default is None.
        injection_seps : 1D-array, optional
            List of separations to inject companions at (arcsec).
        injection_pas : 1D-array, optional
            List of position angles to inject companions at (degrees).
        injection_flux_sigma : float, optional
            The peak flux of all injected companions in units of sigma, relative
            to the 1sigma contrast at the injected separation.
        multi_injection_spacing : int, None, optional
            Spacing between companions injected in a single image. If companions
            are too close then it can pollute the recovered flux. Set to 'None'
            to inject only one companion at a time (lambda/D).
        use_saved : bool, optional
            Toggle to use existing saved injected and recovered fluxes instead of
            repeating the process.
        thrput_fit_method : str, optional
            Method to use when fitting/interpolating the measure KLIP throughputs
            across all of the injection positions. 'median' for a median of PAs at
            with the same separation. 'log_grow' for a logistic growth function.

        Returns
        -------
        None.
        """

        return calibrate_contrast(
            self.database,
            subdir="calcon",
            rawcon_subdir="rawcon",
            rawcon_filetype="npy",
            companions=None,
            injection_seps="default",
            injection_pas="default",
            injection_flux_sigma=20,
            multi_injection_spacing=None,
            use_saved=False,
            thrput_fit_method="median",
            plot_xlim=(0, 10),
            plot_style=None,
            **kwargs,
        )

    def extract_companions(
        self,
        companions,
        starfile,
        mstar_err,
        spectral_type="G2V",
        planetfile=None,
        klmode="max",
        date="auto",
        use_fm_psf=True,
        flip_fmpsf_xy=None,
        highpass=False,
        fitmethod="mcmc",
        minmethod=None,
        fitkernel="diag",
        subtract=True,
        inject=False,
        remove_background=False,
        save_preklip=False,
        overwrite=True,
        subdir="companions",
        save_figures=True,
        **kwargs,
    ):
        """
        Extract the best fit parameters of a number of companions from each
        reduction in the spaceKLIP reductions database.

        Parameters
        ----------
        companions : list of list of three float, optional
            List of companions to be extracted. For each companion, there
            should be a three element list containing guesses for [RA offset
            (arcsec), Dec offset (arcsec), contrast].
        starfile : path
            Path of VizieR VOTable containing host star photometry or two
            column TXT file with wavelength (micron) and flux (Jy).
        mstar_err : float or dict of float
            Error on the host star magnitude. If float, will use the same value
            for each filter. If dict of float, the dictionary keys must be the
            JWST filters in use and a different value can be used for each
            filter.
        spectral_type : str, optional
            Host star spectral type for the stellar model SED. The default is
            'G2V'.
        planetfile : path
            Path of VizieR VOTable containing companion photometry or two
            column TXT file with wavelength (micron) and flux (Jy).
        klmode : int or 'max', optional
            KL mode for which the companions shall be extracted. If 'max', then
            the maximum possible KL mode will be used. The default is 'max'.
        date : str, optional
            Observation date in the format 'YYYY-MM-DDTHH:MM:SS.MMM'. Will
            query for the wavefront measurement closest in time to the given
            date. If 'auto', will grab date from the FITS file header. If None,
            then the default WebbPSF OPD is used (RevAA). The default is
            'auto'.
        use_fm_psf : bool, optional
            If True, use a FM PSF generated with pyKLIP, otherwise use a more
            simple integration time-averaged model offset PSF which does not
            incorporate any KLIP throughput losses. The default is True.
        flip_fmpsf_xy : str, optional
            If 'x', flip the x-axis of the FM PSF. If 'y', flip the y-axis of the FM PSF. 'xy' or 'yx' for both.
        highpass : bool or float, optional
            If float, will apply a high-pass filter to the FM PSF and KLIP
            dataset. The default is False.
        fitmethod : 'mcmc' or 'nested', optional
            Sampling algorithm which shall be used. If None and minmethod not None, it will mock the MCMC fit results
            using the initial guesses and perform only the Gaussian convolution fit to estimate extension.
            The default is 'mcmc'.
        minmethod: str, optional
            scipy.optimize.minimize minimization method which shall be used to fit for the extension of the source.
            The default is None.
        fitkernel : str, optional
            Pyklip.fitpsf.FitPSF covariance kernel which shall be used for the
            Gaussian process regression. The default is 'diag'.
        subtract : bool, optional
            If True, subtract each extracted companion from the pyKLIP dataset
            before fitting the next one in the list. The default is True.
        inject : bool, optional
            Instead of fitting for a companion at the guessed location and
            contrast, inject one into the data.
        remove_background : bool, optional
            Remove a constant background level from the KLIP-subtracted data
            before fitting the FM PSF. The default is False.
        save_preklip : bool, optional
            Save the stage 2 files when injecting/killing a companion? The
            default is False.
        overwrite : bool, optional
            If True, compute a new FM PSF and overwrite any existing one,
            otherwise try to load an existing one and only compute a new one if
            none exists yet. The default is True.
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'companions'.
        save_figures : bool, optional
            Save the plots in a PDF?

        Returns
        -------
        None.

        """

        return extract_companions(
            self.database,
            companions,
            starfile,
            mstar_err,
            spectral_type="G2V",
            planetfile=None,
            klmode="max",
            date="auto",
            use_fm_psf=True,
            flip_fmpsf_xy=None,
            highpass=False,
            fitmethod="mcmc",
            minmethod=None,
            fitkernel="diag",
            subtract=True,
            inject=False,
            remove_background=False,
            save_preklip=False,
            overwrite=True,
            subdir="companions",
            save_figures=True,
            **kwargs,
        )


def loss_function(params, offset_psf, target_array):
    """
    Loss function for the minimization process in fit_for_extended_sources.
    """
    sigma_x, sigma_y, theta_degrees, scale = params
    kernel = gaussian_kernel(
        sigma_x=sigma_x, sigma_y=sigma_y, theta_degrees=theta_degrees, n=6
    )
    convolved_image = convolve(offset_psf * 10**scale, kernel)

    # mse = np.nanmean((target_array - convolved_image) ** 2)
    mse = np.nanmean((target_array - convolved_image) ** 2 * (target_array))
    return mse


def best_convfit_and_residuals(
    fma, minmethod="Powell", bounds=None, initial_params=None, fig=None
):
    """
    Fit the companion with a 2D Gaussian using a minimization algorithm and evaluate the sigma_x, sigma_y and theta.
    Then, generate a plot of the best fit FM compared with the data_stamp and also the residuals

    Parameters
    ----------
    fma: fitpsf.FMAstrometry
        FMAstronomy object with the desired properties
    method: str
        Minimization method. Default is 'Powell'.
    bounds: list of 2-D arrays.
        list of 3 elements, containing the min, max values for the sigma_x, sigma_y and theta_degrees parameters
        for the 2-D gaussian kernel. Default is None.
    initial_params: list of floats
        list of initial guesses for the sigma_x, sigma_y and theta_degrees parameters
        for the 2-D gaussian kernel. Default is None.
    fig: matplotlib.Figure
        if not None, a matplotlib Figure object, function will make a new one. Default is None.


    Returns
    -------
    fig: matplotlib.Figure
        the Figure object. If input fig is None, function will make a new one
    result: array
        Array containing the fitted  parameters from the minimization process
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 4))

    # create best fit FM
    dx = fma.fit_x.bestfit - fma.data_stamp_x_center
    dy = fma.fit_y.bestfit - fma.data_stamp_y_center

    fm_bestfit = fma.fit_flux.bestfit * sinterp.shift(fma.fm_stamp, [dy, dx])

    if fma.padding > 0:
        fm_bestfit = fm_bestfit[fma.padding : -fma.padding, fma.padding : -fma.padding]

    if minmethod is not None:
        result = estimate_extended(
            fma.data_stamp,
            fm_bestfit,
            bounds=bounds,
            initial_params=initial_params,
            method=minmethod,
        )
        # Convolve the PSF by a 2D gaussian
        kernel = gaussian_kernel(
            sigma_x=result.x[0], sigma_y=result.x[1], theta_degrees=result.x[2], n=6
        )
        fm_bestfit_convolved = convolve(fm_bestfit * 10 ** result.x[3], kernel)
    else:
        result = None
        # Convolve the PSF by a 2D gaussian
        kernel = gaussian_kernel(
            sigma_x=fma.fit_sigma_x.bestfit,
            sigma_y=fma.fit_sigma_y.bestfit,
            theta_degrees=fma.fit_theta.bestfit,
            n=6,
        )
        fm_bestfit_convolved = convolve(fm_bestfit * fma.fit_flux.bestfit, kernel)

    # make residual map
    residual_map = fma.data_stamp - fm_bestfit_convolved

    # normalize all images to same scale
    colornorm = matplotlib.colors.Normalize(
        vmin=np.nanpercentile(fma.data_stamp, 0.03),
        vmax=np.nanpercentile(fma.data_stamp, 99.7),
    )

    # plot the data_stamp
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(
        fma.data_stamp, interpolation="nearest", cmap="cubehelix", norm=colornorm
    )
    ax1.invert_yaxis()
    ax1.set_title("Data")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(
        fm_bestfit_convolved, interpolation="nearest", cmap="cubehelix", norm=colornorm
    )
    ax2.invert_yaxis()
    ax2.set_title("Best-fit Model convolved\nby a 2D Gaussian")
    ax2.set_xlabel("X (pixels)")

    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(
        residual_map, interpolation="nearest", cmap="cubehelix", norm=colornorm
    )
    ax3.invert_yaxis()
    ax3.set_title("Residuals")
    ax3.set_xlabel("X (pixels)")

    fig.subplots_adjust(right=0.82)
    fig.subplots_adjust(hspace=0.4)
    ax_pos = ax3.get_position()

    cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, ax_pos.height])
    cb = fig.colorbar(im1, cax=cbar_ax)
    cb.set_label("Counts (DN)")

    return fig, result


def estimate_extended(
    target, offset_psf, bounds=None, initial_params=None, method="Powell"
):
    """
    Fit for extended sources with a 2D gaussian kernel.

    Parameters
    ----------
    target: 2-D array.
        The target tile containing the companion to be fitted.
    offset_psf: 2-D array.
        The PSF of the companion to be used for the fit.
    bounds: list of 2-D arrays.
        list of 3 elements, containing the min, max values for the sigma_x, sigma_y and theta_degrees parameters
        for the 2-D gaussian kernel. If None, use default bounds = [(0.01, 20),(0.01, 20),(-180, 180)]
    initial_params: list of floats
        list of initial guesses for the sigma_x, sigma_y and theta_degrees parameters
        for the 2-D gaussian kernel. If None, use default initial_params = [0.1, 0.1, 0]
    method: str
        Minimization method. Default is 'Powell'.

    Returns
    -------
    result: array
        Array containing the fitted  parameters from the minimization process

    """
    if initial_params is None:
        initial_params = [0.1, 0.1, 0, 0]

    if bounds is None:
        # Bounds for parameters (sigma_x, sigma_y, theta, intensity)
        bounds = [
            (0.01, 5),  # sigma_x should be positive and within a reasonable range
            (0.01, 5),  # sigma_y should be positive and within a reasonable range
            (-180, 180),  # theta should be between -180 and 180 degrees
            (-1, 1),
        ]  # log flux range should be positive and within a reasonable range

    # Use partial to pass target_array as a fixed argument to the loss function
    loss_with_target = partial(
        loss_function, offset_psf=offset_psf, target_array=target
    )
    result = minimize(loss_with_target, initial_params, method=method, bounds=bounds)
    return result


def inject_and_recover(
    raw_dataset,
    injection_psf,
    injection_seps,
    injection_pas,
    injection_spacing,
    injection_fluxes,
    klip_args,
    retrieve_fwhm,
    true_companions=None,
):
    """
    Function to inject synthetic PSFs into a pyKLIP dataset, then perform
    KLIP subtraction, then calculate the flux losses from the KLIP process.

    Parameters
    ----------
    raw_dataset : pyKLIP dataset
        A pyKLIP dataset which companions will be injected into and KLIP
        will be performed on.
    injection_psf : 2D-array
        The PSF of the companion to be injected.
    injection_seps : 1D-array
        List of separations to inject companions at (pixels).
    injection_pas : 1D-array
        List of position angles to inject companions at (degrees).
    injection_spacing : int, None
        Spacing between companions injected in a single image. If companions
        are too close then it can pollute the recovered flux. Set to 'None'
        to inject only one companion at a time (pixels).
    injection_fluxes : 1D-array
        Same size as injection_seps, units should correspond to the image
        units. This is the *peak* flux of the injection.
    klip_args : dict
        Arguments to be passed into the KLIP subtraction process
    retrieve_fwhm : float
        Full-Width Half-Maximum value to estimate the 2D gaussian fit when
        retrieving the companion fluxes.
    true_companions : list of list of three float, optional
        List of real companions to be masked before computing the raw contrast.
        For each companion, there should be a three element list containing
        [RA offset (pixels), Dec offset (pixels), mask radius (pixels)].
        The default is None.

    Returns
    -------
    all_seps : np.array
        Array containing the separations of all injected
        companions across all images.
    all_pas : np.array
        Array containing the position angles of all injected
        companions across all images.
    all_inj_fluxes : np.array
        Array containing the injected peak fluxes of all injected
        companions across all images.
    all_retr_fluxes : np.array
        Array containing the retrieved peak fluxes of all injected
        companions across all images.
    """

    # Initialise some arrays and quantities
    Nsep = len(injection_seps)
    Npa = len(injection_pas)
    list_of_injected = []
    all_injected = False
    all_seps = []
    all_pas = []
    all_inj_fluxes = []
    all_retr_fluxes = []

    # Ensure provided PSF is normalised to a peak intensity of 1
    injection_psf_norm = injection_psf / np.max(injection_psf)

    # Don't want to inject near any known companions, eliminate any
    # of these positions straight away.
    if true_companions is not None:
        for tcomp in true_companions:
            tcomp_ra, tcomp_de, tcomp_rad = tcomp
            for i in range(Nsep):
                for j in range(Npa):
                    pos_id = i * Npa + j
                    # Convert position to x-y (RA-DEC) offset in pixels
                    inj_ra = injection_seps[i] * np.sin(
                        np.deg2rad(injection_pas[j])
                    )  # pixels
                    inj_de = injection_seps[i] * np.cos(
                        np.deg2rad(injection_pas[j])
                    )  # pixels
                    # Calculate distance to companion
                    dist = np.sqrt((tcomp_ra - inj_ra) ** 2 + (tcomp_de - inj_de) ** 2)
                    # Check if too close, if so, lie to the code and say its already injected
                    if dist < tcomp_rad:
                        list_of_injected += [pos_id]
    if len(list_of_injected) != 0:
        log.info(
            "--> {}/{} source positions not suitable for injection.".format(
                len(list_of_injected), Nsep * Npa
            )
        )
    else:
        log.info(
            "--> All {} source positions suitable for injection.".format(Nsep * Npa)
        )

    # Want to keep going until a companion has been injected and recovered
    # at each given separation and position angle.
    counter = 1
    remaining_to_inject = (Nsep * Npa) - len(list_of_injected)
    with trange(remaining_to_inject, position=0, leave=True) as t:
        while all_injected == False:
            # Make a copy of the dataset
            dataset = copy.deepcopy(raw_dataset)
            # Define array to keep track of currently injected positions
            current_injected = []
            # Loop over separations
            for i in range(Nsep):
                new_sep = injection_seps[i]
                new_flux = injection_fluxes[i]
                # Loop over position angles
                for j in range(Npa):
                    new_pa = injection_pas[j]

                    # Get specific id for this position
                    pos_id = i * Npa + j
                    if pos_id in list_of_injected:
                        # Already injected at this position, skip
                        continue

                    # Need to check if this position is too close to already
                    # injected positions. By default, assume we want to inject.
                    inject_flag = True
                    for inj_id in current_injected:
                        # If we don't want to inject more than one companion
                        # per image, then flag to not inject.
                        if injection_spacing == None:
                            inject_flag = False
                            break

                        # Get separation and PA for injected position
                        inj_j = inj_id % Npa
                        inj_i = (inj_id - inj_j) // Npa
                        inj_sep = injection_seps[inj_i]
                        inj_pa = injection_pas[inj_j]
                        inj_flux = injection_fluxes[inj_i]

                        # If something was injected close to the coronagraph
                        # don't inject anything else in this image.
                        if inj_sep < 5:
                            inject_flag = False
                            break

                        # Calculate distance between this injected position
                        # and the new position we'd also like to inject at.
                        # If object is too close to something that's already
                        # injected, we don't want to inject.
                        dist = np.sqrt(
                            new_sep**2
                            + inj_sep**2
                            - 2
                            * new_sep
                            * inj_sep
                            * np.cos(np.deg2rad(inj_pa - new_pa))
                        )
                        if dist < injection_spacing:
                            inject_flag = False
                            break

                        # If the difference in fluxes is too large, don't inject
                        # as this can really affect things.
                        flux_factor = max(inj_flux, new_flux) / min(inj_flux, new_flux)
                        if flux_factor > 10:
                            inject_flag = False
                            break

                    # If this position survived the filtering, inject into images
                    if inject_flag == True:
                        # Mark as injected in this dataset and overall.
                        current_injected += [pos_id]
                        list_of_injected += [pos_id]

                        # Injected PSF needs to be a 3D array that matches dataset
                        inj_psf_3d = np.array(
                            [
                                injection_psf_norm * new_flux
                                for k in range(dataset.input.shape[0])
                            ]
                        )

                        # Inject the PSF
                        fakes.inject_planet(
                            frames=dataset.input,
                            centers=dataset.centers,
                            inputflux=inj_psf_3d,
                            astr_hdrs=dataset.wcs,
                            radius=new_sep,
                            pa=new_pa,
                            stampsize=65,
                        )

            # Figure out how many sources were injected
            Ninjected = len(current_injected)
            t.update(Ninjected)

            # Reroute KLIP printing for our own progress bar
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            # Still in the while loop, need to run KLIP on the dataset we
            # have injected companions into.
            fileprefix = "INJ_ITER{}_{}COMP".format(counter, Ninjected)
            parallelized.klip_dataset(
                dataset=dataset,
                psf_library=dataset.psflib,
                fileprefix=fileprefix,
                **klip_args,
            )

            # Restore printing
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Now need to recover the flux by fitting a 2D Gaussian, mainly interested in the peak
            # flux so this is an okay approximation. Could improve in the future.
            klipped_file = klip_args["outputdir"] + fileprefix + "-KLmodes-all.fits"
            with fits.open(klipped_file) as hdul:
                klipped_data = hdul[0].data
                frame_ids = range(klipped_data.shape[0])
                centers = [
                    [hdul[0].header["PSFCENTX"], hdul[0].header["PSFCENTY"]]
                    for c in frame_ids
                ]
                # Get fluxes for all companions that were injected, for all KL modes used.
                for inj_id in current_injected:
                    inj_j = inj_id % Npa
                    inj_i = (inj_id - inj_j) // Npa
                    inj_sep = injection_seps[inj_i]
                    inj_pa = injection_pas[inj_j]
                    inj_flux = injection_fluxes[inj_i]

                    # Need to loop over each KL mode individually due to pyKLIP subtleties,
                    # basically the same as what pyKLIP would be doing anyway.
                    retrieved_fluxes = []
                    for img_i in range(klipped_data.shape[0]):
                        retrieved_flux = fakes.retrieve_planet_flux(
                            frames=klipped_data[img_i],
                            centers=centers[img_i],
                            astr_hdrs=dataset.output_wcs[0],
                            sep=inj_sep,
                            pa=inj_pa,
                            searchrad=5,
                            guessfwhm=retrieve_fwhm,
                            guesspeak=inj_flux,
                            refinefit=True,
                        )
                        retrieved_fluxes.append(retrieved_flux)
                    retrieved_fluxes = np.array(
                        retrieved_fluxes
                    )  # Convert to numpy array

                    # Flux should never be negative, if it is, assume ~=zero flux retrieved
                    neg_mask = np.where(retrieved_fluxes < 0)
                    retrieved_fluxes[neg_mask] = 1e-10

                    # Need to save things to some arrays
                    all_seps += [inj_sep]
                    all_pas += [inj_pa]
                    all_inj_fluxes += [inj_flux]
                    all_retr_fluxes += [retrieved_fluxes]

            # If a companion has been injected and retrieved at every input position then
            # flag to exit the loop. If not increment the counter and continue.
            if len(list_of_injected) == Nsep * Npa:
                all_injected = True
            else:
                counter += 1

    # Return as numpy arrays
    all_seps = np.array(all_seps)
    all_pas = np.array(all_pas)
    all_inj_fluxes = np.array(all_inj_fluxes)
    all_retr_fluxes = np.squeeze(all_retr_fluxes)

    # Ensure dimensions are correct for all_retr_fluxes if # of different KL modes == 1
    if all_retr_fluxes.ndim == 1:
        all_retr_fluxes = all_retr_fluxes[:, np.newaxis]

    return all_seps, all_pas, all_inj_fluxes, all_retr_fluxes
