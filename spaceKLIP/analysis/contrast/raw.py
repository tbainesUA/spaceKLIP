from __future__ import division

import logging

# =============================================================================
# IMPORTS
# =============================================================================
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np

# -----
from astropy.io import fits
from astropy.table import Table
from numpy.typing import NDArray
from pyklip import klip
from stpsf.constants import JWST_CIRCUMSCRIBED_DIAMETER

from spaceKLIP import utils as ut
from spaceKLIP.psf import get_offsetpsf
from spaceKLIP.starphot import get_stellar_magnitudes
from spaceKLIP.utils import set_surrounded_pixels, write_starfile

from .masking import apply_instrument_mask
from .plotting import plot_masked_data, plot_raw_contrast

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


#
# Helpers
#

from typing import Final

# =============================================================================
# DOMAIN CONSTANTS
# =============================================================================

MIRI_4QPM_ROTATION_OFFSET_DEG: Final[float] = 4.83544897

MIRI_4QPM_WIDE_BAR_WIDTH_PIX: Final[int] = 10
MIRI_4QPM_THIN_BAR_WIDTH_PIX: Final[int] = 2
MIRI_4QPM_INNER_CLEAR_RADIUS_PIX: Final[int] = 15

MIRI_ROTATION_REFERENCE_DEG: Final[float] = 90.0

MASK_OCCUPANCY_THRESHOLD: Final[float] = 0.5


# =============================================================================
# LOW-LEVEL GEOMETRIC KERNELS
# =============================================================================


# @dataclass(frozen=True)
# class ContrastResult:
#     """Results from a raw contrast measurement"""

#     separations: NDArray[np.floating]
#     raw_contrast: NDArray[np.floating]
#     throughput_corrected_contrast: NDArray[np.floating] | None


# @dataclass(frozen=True)
# class ContrastResult:
#     """Raw contrast curves measured for a set of KL modes."""

#     separations: NDArray[np.floating]
#     raw_contrast: NDArray[np.floating]
#     throughput_corrected_contrast: NDArray[np.floating] | None
#     # klmodes: tuple[int, ...]

#     # @property
#     # def n_klmodes(self) -> int:
#     #     return len(self.klmodes)

#     @property
#     def has_throughput_correction(self) -> bool:
#         return self.throughput_corrected_contrast is not None


@dataclass(frozen=True)
class ContrastResult:
    """Contrast curves measured from a reduced image cube."""

    separations_pix: NDArray[np.floating]
    raw_contrast: NDArray[np.floating]
    throughput_corrected_contrast: NDArray[np.floating] | None = None

    @property
    def has_throughput_correction(self) -> bool:
        return self.throughput_corrected_contrast is not None


def rotate_coordinates(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    rotation_angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate detector coordinates using broadcasted rotation matrices.

    Returns
    -------
    x_rotated, y_rotated
        Arrays with shape (n_rolls, ny, nx)
    """
    theta_rad = np.deg2rad(rotation_angles_deg)

    cos_theta = np.cos(theta_rad)[:, None, None]
    sin_theta = np.sin(theta_rad)[:, None, None]

    x_rotated = x_coords[None, :, :] * cos_theta - y_coords[None, :, :] * sin_theta

    y_rotated = x_coords[None, :, :] * sin_theta + y_coords[None, :, :] * cos_theta

    return x_rotated, y_rotated


def evaluate_cross_occultation(
    x_coords_pix: np.ndarray,
    y_coords_pix: np.ndarray,
    wide_bar_width_pix: float,
    thin_bar_width_pix: float,
    inner_clear_radius_pix: float,
) -> np.ndarray:
    """
    Evaluate analytic 4QPM occultation geometry.
    """
    half_wide = wide_bar_width_pix / 2.0
    half_thin = thin_bar_width_pix / 2.0

    radius_squared = x_coords_pix**2 + y_coords_pix**2

    central_clear_region = radius_squared < inner_clear_radius_pix**2

    vertical_wide = np.abs(x_coords_pix) <= half_wide
    horizontal_wide = np.abs(y_coords_pix) <= half_wide

    vertical_thin = np.abs(x_coords_pix) <= half_thin
    horizontal_thin = np.abs(y_coords_pix) <= half_thin

    occulted_region = (vertical_wide | horizontal_wide) & ~central_clear_region

    structural_cross = vertical_thin | horizontal_thin

    return occulted_region | structural_cross


# =============================================================================
# HIGH-LEVEL DOMAIN API
# =============================================================================


def generate_miri_4qpm_mask(
    detector_shape: Tuple[int, int],
    coronagraph_center_pix: Tuple[float, float],
    roll_reference_angles_deg: np.ndarray,
    upsample_factor: int = 1,
    padding_pix: int = 0,
) -> np.ndarray:
    """
    Generate the MIRI 4QPM occultation mask analytically.

    Parameters
    ----------
    detector_shape
        Detector shape as (ny, nx).

    coronagraph_center_pix
        Coronagraph center coordinates as (x_center, y_center).

    roll_reference_angles_deg
        Telescope roll angles in degrees.

    upsample_factor
        Integer detector upsampling factor.

    Returns
    -------
    np.ndarray
        Float mask where occulted pixels are NaN.
    """
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")

    if np.any(~np.isfinite(roll_reference_angles_deg)):
        raise ValueError("roll_reference_angles_deg contains NaN or Inf values.")

    ny, nx = detector_shape

    # Here we're preserving the padding and upsampling
    padded_ny = ny + 2 * padding_pix
    padded_nx = nx + 2 * padding_pix

    x_center_pix, y_center_pix = coronagraph_center_pix
    x_center_pix += padding_pix
    y_center_pix += padding_pix

    sampled_ny = padded_ny * upsample_factor
    sampled_nx = padded_nx * upsample_factor

    y_coords_pix = (
        np.arange(sampled_ny, dtype=np.float32) / upsample_factor - y_center_pix
    )

    x_coords_pix = (
        np.arange(sampled_nx, dtype=np.float32) / upsample_factor - x_center_pix
    )

    x_grid_pix, y_grid_pix = np.meshgrid(
        x_coords_pix,
        y_coords_pix,
        indexing="xy",
        sparse=False,
    )

    rotation_angles_deg = (
        MIRI_ROTATION_REFERENCE_DEG
        - roll_reference_angles_deg
        + MIRI_4QPM_ROTATION_OFFSET_DEG
    )

    x_rotated_pix, y_rotated_pix = rotate_coordinates(
        x_grid_pix,
        y_grid_pix,
        rotation_angles_deg,
    )

    occultation_stack = evaluate_cross_occultation(
        x_coords_pix=x_rotated_pix,
        y_coords_pix=y_rotated_pix,
        wide_bar_width_pix=MIRI_4QPM_WIDE_BAR_WIDTH_PIX,
        thin_bar_width_pix=MIRI_4QPM_THIN_BAR_WIDTH_PIX,
        inner_clear_radius_pix=MIRI_4QPM_INNER_CLEAR_RADIUS_PIX,
    )

    combined_occultation = np.any(occultation_stack, axis=0)

    if upsample_factor > 1:
        combined_occultation = combined_occultation[
            ::upsample_factor,
            ::upsample_factor,
        ]

    combined_occultation = combined_occultation[
        padding_pix : padding_pix + ny,
        padding_pix : padding_pix + nx,
    ]

    final_mask = np.ones(
        detector_shape,
        dtype=np.float32,
    )

    final_mask[combined_occultation] = np.nan

    return set_surrounded_pixels(final_mask)


# =============================================================================
# HIGH-LEVEL Contrast calculation API
# =============================================================================


def calc_single_contrast_curve(
    normalized_frame: np.ndarray,
    spatial_resolution_pix: float,
    center_pix: tuple[float, float],
    inner_working_angle_pix: int | float,
    outer_working_angle_pix: int | float,
    coronagraph_transmission_mask: Optional[np.ndarray] = None,
    low_pass_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute raw and throughput-corrected contrast curve for single detector frame"""

    separations_pix, raw_contrast = klip.meas_contrast(
        dat=normalized_frame,
        center=center_pix,
        iwa=inner_working_angle_pix,
        owa=outer_working_angle_pix,
        resolution=spatial_resolution_pix,
        low_pass_filter=low_pass_filter,
    )

    corrected_contrast = None
    if coronagraph_transmission_mask is not None:
        corrected_frame = normalized_frame / coronagraph_transmission_mask

        _, corrected_contrast = klip.meas_contrast(
            dat=corrected_frame,
            center=center_pix,
            iwa=inner_working_angle_pix,
            owa=outer_working_angle_pix,
            resolution=spatial_resolution_pix,
            low_pass_filter=low_pass_filter,
        )

    return (separations_pix, raw_contrast, corrected_contrast)


MIN_CORONAGRAPH_TRANSMISSION = 1e-12


def normalize_contrast_frame(
    frame_data: np.ndarray,
    normalization_factor: float,
) -> np.ndarray:
    """
    Normalize detector frame into contrast units.
    """
    return frame_data * normalization_factor


def apply_throughput_correction(
    normalized_frame: np.ndarray,
    transmission_mask: np.ndarray,
    minimum_transmission: float = MIN_CORONAGRAPH_TRANSMISSION,
) -> np.ndarray:
    """
    Apply bounded coronagraph throughput correction.

    Pixels below minimum transmission are masked to NaN.
    """
    corrected_frame = np.full_like(
        normalized_frame,
        np.nan,
        dtype=np.float32,
    )

    valid_transmission = transmission_mask >= minimum_transmission

    np.divide(
        normalized_frame,
        transmission_mask,
        out=corrected_frame,
        where=valid_transmission,
    )

    return corrected_frame


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

#     # return (
#     #     radial_separations_pix,
#     #     raw_contrast_curves,
#     #     throughput_corrected_array,
#     # )
#     return ContrastResult(
#         separations=radial_separations_pix,
#         raw_contrast=raw_contrast_curves,
#         throughput_corrected_contrast=throughput_corrected_array,
#     )


def compute_contrast_curves(
    data_cube: np.ndarray,
    pixel_area_sr: float,
    stellar_flux_peak: float,
    spatial_resolution_pix: float,
    center_pix: tuple[float, float],
    inner_working_angle_pix: int | float = 1,
    outer_working_angle_pix: int | float | None = None,
    coronagraph_transmission_mask: np.ndarray | None = None,
) -> ContrastResult:
    """
    Compute raw and throughput-corrected contrast curves.

    Parameters
    ----------
    data_cube
        Input detector cube with shape (n_frames, ny, nx).
    pixel_area_sr
        Pixel solid angle in steradians.
    stellar_flux_peak
        Stellar peak normalization flux.
    spatial_resolution_pix
        Resolution element diameter in pixels.
    center_pix
        PSF center coordinates as (x, y).
    inner_working_angle_pix
        Inner working angle in pixels.
    outer_working_angle_pix
        Outer working angle in pixels.
    coronagraph_transmission_mask
        Optional coronagraph throughput transmission map.

    Returns
    -------
    ContrastResult
        Contrast measurements sharing a common radial-separation grid.
    """
    if data_cube.ndim != 3:
        raise ValueError("data_cube must have shape (n_frames, ny, nx)")

    if stellar_flux_peak <= 0:
        raise ValueError("stellar_flux_peak must be positive.")

    if (
        coronagraph_transmission_mask is not None
        and coronagraph_transmission_mask.shape != data_cube.shape[1:]
    ):
        raise ValueError("coronagraph_transmission_mask shape mismatch.")

    if outer_working_angle_pix is None:
        outer_working_angle_pix = min(data_cube.shape[1:]) // 2

    normalization_factor = pixel_area_sr / stellar_flux_peak

    normalized_cube = (data_cube * normalization_factor).astype(np.float32, copy=False)

    raw_contrast_curves = []
    throughput_corrected_curves = []

    radial_separations_pix = None

    contrast_kwargs = {
        "inner_working_angle_pix": inner_working_angle_pix,
        "outer_working_angle_pix": outer_working_angle_pix,
        "spatial_resolution_pix": spatial_resolution_pix,
        "center_pix": center_pix,
        "low_pass_filter": False,
        "coronagraph_transmission_mask": coronagraph_transmission_mask,
    }

    for normalized_frame in normalized_cube:
        (
            separations_pix,
            raw_contrast,
            corrected_contrast,
        ) = calc_single_contrast_curve(
            normalized_frame=normalized_frame, **contrast_kwargs
        )

        if radial_separations_pix is None:
            radial_separations_pix = separations_pix

        raw_contrast_curves.append(raw_contrast)

        if corrected_contrast is not None:
            throughput_corrected_curves.append(corrected_contrast)

    raw_contrast_array = np.asarray(raw_contrast_curves, dtype=np.float32)

    throughput_corrected_array = None

    if coronagraph_transmission_mask is not None:
        throughput_corrected_array = np.asarray(
            throughput_corrected_curves, dtype=np.float32
        )

    return ContrastResult(
        separations_pix=np.asarray(radial_separations_pix, dtype=np.float32),
        raw_contrast=raw_contrast_array,
        throughput_corrected_contrast=(throughput_corrected_array),
    )


# =============================================================================
# Adding in the companion masks now
# =============================================================================


# =============================================================================
# ATOMIC MATHEMATICAL KERNEL (Pure Point Logic)
# =============================================================================


def compute_single_companion_mask(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    center_pix: Tuple[float, float],
    pixscale_arcsec: float,
    spatial_resolution_pix: float,
    ra_offset_arcsec: float,
    dec_offset_arcsec: float,
    mask_radius_ld: float,
) -> NDArray[np.bool_]:
    """
    Compute a 2D boolean exclusion mask for a single celestial companion.

    This function acts as a pure stateless coordinate-space kernel. It does
    not know about data frames, arrays of companions, or file paths.

    Parameters
    ----------
    x_grid : NDArray[np.float64]
        2D matrix containing the horizontal (X) pixel coordinates.
    y_grid : NDArray[np.float64]
        2D matrix containing the vertical (Y) pixel coordinates.
    center_pix : Tuple[float, float]
        0-indexed host star coordinate array (X_center, Y_center).
    pixscale_arcsec : float
        Pixel scale of the instrument detector in arcseconds/pixel.
    spatial_resolution_pix : float
        Calculated resolution element size (lambda/D) expressed in pixels.
    ra_offset_arcsec : float
        Right Ascension angular offset relative to host star in arcseconds.
    dec_offset_arcsec : float
        Declination angular offset relative to host star in arcseconds.
    mask_radius_ld : float
        Exclusion zone cutoff radius in units of lambda/D.

    Returns
    -------
    NDArray[np.bool_]
        A 2D boolean mask of shape matching the input grid, where True
        denotes coordinates inside the companion exclusion boundary.
    """
    # Transform physical angular coordinates into raw pixel displacements
    delta_x_pix = ra_offset_arcsec / pixscale_arcsec
    delta_y_pix = dec_offset_arcsec / pixscale_arcsec
    cutoff_radius_pix = mask_radius_ld * spatial_resolution_pix

    # Calculate radial Euclidean distance from the companion's shifted center
    radial_distance_map = np.sqrt(
        (x_grid - center_pix[0] + delta_x_pix) ** 2
        + (y_grid - center_pix[1] - delta_y_pix) ** 2
    )

    return radial_distance_map <= cutoff_radius_pix


# =============================================================================
# GEOMETRIC EXCLUSION FACTORY (State Aggregator Layer)
# =============================================================================


def generate_companion_spatial_mask(
    spatial_shape: Tuple[int, int],
    center_pix: Tuple[float, float],
    pixscale_arcsec: float,
    spatial_resolution_pix: float,
    companions: List[List[float]],
) -> NDArray[np.bool_]:
    """
    Generate a unified 2D binary mask indicating multi-companion exclusion zones.

    Parameters
    ----------
    spatial_shape : Tuple[int, int]
        The (Y, X) pixel dimensions of the target frame layout.
    center_pix : Tuple[float, float]
        0-indexed target coordinate array (X_center, Y_center).
    pixscale_arcsec : float
        Pixel scale of the instrument detector in arcseconds/pixel.
    spatial_resolution_pix : float
        Calculated resolution element size (lambda/D) expressed in pixels.
    companions : List[List[float]]
        Matrix of target companions where each row represents
        [RA_offset, Dec_offset, radius_ld].

    Returns
    -------
    NDArray[np.bool_]
        2D unified spatial boolean mask layout.
    """
    ny, nx = spatial_shape
    y_indices, x_indices = np.indices((ny, nx), dtype=np.float64)

    # Initialize a master empty mask
    combined_mask = np.zeros((ny, nx), dtype=bool)

    # Map the companion list across our mathematical kernel
    for companion in companions:
        ra_off, dec_off, radius_ld = companion

        single_mask = compute_single_companion_mask(
            x_grid=x_indices,
            y_grid=y_indices,
            center_pix=center_pix,
            pixscale_arcsec=pixscale_arcsec,
            spatial_resolution_pix=spatial_resolution_pix,
            ra_offset_arcsec=ra_off,
            dec_offset_arcsec=dec_off,
            mask_radius_ld=radius_ld,
        )

        # Accumulate the mask footprint using bitwise OR
        combined_mask |= single_mask

    return combined_mask


# =============================================================================
# math
# =============================================================================
ARCSEC_PER_RADIAN = 180.0 * 3600.0 / np.pi


def calculate_pixel_area_sr(pixel_scale_arcsec: float) -> float:
    """
    Calculate the solid angle subtended by a single detector pixel in steradians.

    Parameters
    ----------
    pixel_scale_arcsec : float
        The physical size of a single detector pixel in arcseconds.

    Returns
    -------
    float
        The solid angle area equivalent expressed in units of steradians (sr).
    """
    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be a positive non-zero value.")

    # Convert linear pixel edge dimension from arcseconds to radians
    pixel_scale_rad = pixel_scale_arcsec / ARCSEC_PER_RADIAN

    # Compute 2D area component (square radians is structurally equivalent to steradians)
    pixel_area_sr = pixel_scale_rad**2

    return float(pixel_area_sr)


# Official JWST Entrance Pupil/Primary Mirror Diameter Constants (Meters)
# JWST_CIRCUMSCRIBED_DIAMETER = 6.6  # Outer edge boundary
MICRON_TO_METERS = 1e-6
JWST_CORONAGRAPH_DIAMETER = (
    5.2  # Effective clearance diameter for specialized coronagraph masking
)


def calculate_spatial_resolution_pix(
    wavelength_um: float,
    pixel_scale_rad: float,
    telescope_name: str,
    exposure_type: str,
    blur_fwhm_pix: float = 0.0,
    factor: float = 1.0,
) -> float:
    """
    Calculate the effective structural resolution element in pixels, accounting
    for optical diffraction limitations and instrument-level blurring.

    Parameters
    ----------
    wavelength_um : float
        The observation center wavelength in micrometers (CWAVEL).
    pixel_scale_rad : float
        The pixel scale size expressed in radians per pixel (pxsc_rad).
    telescope_name : str
        Name identifier of the origin observatory (e.g., 'JWST').
    exposure_type : str
        Telemetry exposure type block code (e.g., 'NRC_CORON').
    blur_fwhm_pix : float, optional
        Additional atmospheric/instrument blur FWHM in pixel units (BLURFWHM). Defaults to 0.0.
    factor : float, optional
        Diffraction element scaling modifier. Defaults to 1.0 (lambda/D). Change to 1.22
        if strictly computing a Rayleigh criteria resolution bound.

    Returns
    -------
    float
        The effective physical resolution element scaled into detector pixels.
    """
    # 1. Enforce strict telescope type gating
    if str(telescope_name).upper() != "JWST":
        raise ValueError(
            f"Unsupported or unknown telescope platform: '{telescope_name}'"
        )

    if wavelength_um <= 0 or pixel_scale_rad <= 0:
        raise ValueError(
            "Wavelength and pixel scale inputs must evaluate to positive non-zero floats."
        )

    # 2. Assign effective optical diameter based on specialized pupil masking profiles
    if str(exposure_type).upper() in ["NRC_CORON", "NRC_TACONFIRM", "NRC_TACQ"]:
        # NIRCam coronagraph masks constrain the open entrance pupil to an effective 5.2m
        aperture_diameter_m = JWST_CORONAGRAPH_DIAMETER
    else:
        aperture_diameter_m = JWST_CIRCUMSCRIBED_DIAMETER

    # 3. Compute structural diffraction footprint (meters to micrometers cancel via 1e-6)
    wavelength_m = wavelength_um * 1e-6  # MICRON_TO_METERS
    diffraction_limit_rad = (factor * wavelength_m) / aperture_diameter_m

    # Map the spatial angular bounds directly into detector pixel grids
    spatial_resolution_pix = diffraction_limit_rad / pixel_scale_rad

    # 4. Apply additional instrumental/smearing blurring adjustments via quadrature addition
    if not np.isnan(blur_fwhm_pix) and blur_fwhm_pix > 0:
        spatial_resolution_pix = np.hypot(spatial_resolution_pix, blur_fwhm_pix)

    return float(spatial_resolution_pix)


# -----
# Data Loading


@dataclass
class ReducedData:
    """Reduced SpaceKLIP image product and associated metadata."""

    data: NDArray
    primary_header: fits.Header
    science_header: fits.Header
    is_2d: bool
    transmission_mask: NDArray | None


def load_reduced_data(fitsfile: str | Path, maskfile: str | Path) -> ReducedData:
    """
    Load a reduced SpaceKLIP product and optional transmission mask.

    Parameters
    ----------
    fitsfile
        Reduced FITS product.

    maskfile
        Optional coronagraph transmission-mask file.

    Returns
    -------
    ReducedData
        Loaded image data, FITS headers, dimensionality flag, and
        optional coronagraph transmission mask.
    """
    fitsfile = Path(fitsfile)

    data, primary_header, science_header, is_2d = ut.read_red(str(fitsfile))

    transmission_mask = None

    if maskfile is not None:
        maskfile = Path(maskfile)
        transmission_mask = ut.read_msk(str(maskfile))

    if transmission_mask is None:
        log.warning(f"No coronagraph transmission mask was loaded for {fitsfile.name}")

    return ReducedData(
        data=data,
        primary_header=primary_header,
        science_header=science_header,
        is_2d=is_2d,
        transmission_mask=transmission_mask,
    )


# =============================================================================
# MAIN
# =============================================================================


def validate_companions(companions):
    """Validation check to ensure that a list of companions (objects) have
    3 elements (ra, dec, size lambda/D units) want a list of lists"""
    if companions is None:
        return None

    if not companions:
        return []

    if not isinstance(companions[0], (list, tuple)):
        companions = [companions]

    if any(len(c) != 3 for c in companions):
        raise ValueError("Each companion must contain exactly 3 elements")

    return companions


def create_output_directory(
    base_directory: str | Path,
    subdirectory: str | Path | None = None,
) -> Path:
    """
    Create and return an output directory.

    Parameters
    ----------
    base_directory
        Base output directory.

    subdirectory
        Optional directory beneath ``base_directory``.

    Returns
    -------
    pathlib.Path
        Created output directory.
    """
    output_directory = Path(base_directory)

    if subdirectory is not None:
        output_directory /= subdirectory

    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def stage_stellar_calibration_file(
    starfile: str | Path,
    output_directory: str | Path,
    spectral_type: str,
) -> Path:
    """
    Stage the stellar calibration input used for raw-contrast analysis.

    Parameters
    ----------
    starfile
        Stellar photometry or spectrum file.

    output_directory
        Raw-contrast output directory.

    spectral_type
        Stellar spectral type used when interpreting VOTable photometry.

    Returns
    -------
    pathlib.Path
        Path of the staged stellar calibration file.
    """
    starfile = Path(starfile)
    output_directory = Path(output_directory)

    staged_starfile = output_directory / starfile.name

    if starfile.suffix.lower() == ".vot":
        spectral_type_label = f"Spectral Type: {spectral_type}"
    else:
        spectral_type_label = "Spectral Type: N/A"

    calibration_info = f"#{starfile.name} /// {spectral_type_label}\n"

    info_file = output_directory / "contrast_curve_info.txt"

    info_file.write_text(calibration_info, encoding="utf-8")

    log.info("Copying starfile %s to %s", starfile, staged_starfile)

    write_starfile(str(starfile), str(staged_starfile))

    log.info("Syncing starfile assets: %s to %s", str(starfile), str(staged_starfile))

    return staged_starfile


def run_raw_contrast(
    database,
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
    """
    Compute the raw contrast relative to the provided host star flux.

    Parameters
    ----------
    starfile : path
        Path of VizieR VOTable containing host star photometry or two
        column TXT file with wavelength (micron) and flux (Jy).
    spectral_type : str, optional
        Host star spectral type for the stellar model SED. The default is
        'G2V'.
    companions : list of list of three float, optional
        List of companions to be masked before computing the raw contrast.
        For each companion, there should be a three element list containing
        [RA offset (arcsec), Dec offset (arcsec), mask radius (lambda/D)].
        The default is None.
    overwrite_crpix : tuple of two float, optional
        Overwrite the PSF center with the (CRPIX1, CRPIX2) values provided
        here (in 1-indexed coordinates). This is required for Coron3 data!
        The default is None.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'rawcon'.
    output_filetype : str
        File type to save the raw contrast information to. Options are 'ecsv'
        or 'npy'.
    save_figures : bool, optional
        Save the plots in a PDF?

    Returns
    -------
    None.

    """

    # Check input.
    companions = validate_companions(companions)

    # Set output directory.
    output_dir = create_output_directory(database.output_dir, subdir)

    star_path = stage_stellar_calibration_file(
        starfile, output_directory=output_dir, spectral_type=spectral_type
    )

    # Loop through concatenations.
    # The looping signature here can be updated to be more clearer
    # Example:
    #
    # for key, observation in database.obs.items():
    #     for obs in observation:
    #
    # looking of the keys and table and then iterate per row in the table
    for i, key in enumerate(database.red.keys()):
        log.info("--> Concatenation " + key)
        # Loop through FITS files.
        for j in range(len(database.red[key])):
            log.info("Analyzing file " + database.red[key]["FITSFILE"][j])

            # Fetch data
            instrument = database.red[key]["INSTRUME"][j]
            subarray = database.red[key]["SUBARRAY"][j]
            filt = database.red[key]["FILTER"][j]
            exp_type = database.red[key]["EXP_TYPE"][j]
            pixscale = database.red[key]["PIXSCALE"][j]
            c_wavelength = database.red[key]["CWAVEL"][j]

            fitsfile = database.red[key]["FITSFILE"][j]

            maskfile = database.red[key]["MASKFILE"][j]

            reduced_data = load_reduced_data(fitsfile, maskfile=maskfile)

            data = reduced_data.data
            head_pri = reduced_data.primary_header
            head_sci = reduced_data.science_header
            is2d = reduced_data.is_2d
            mask = reduced_data.transmission_mask

            # Compute the pixel area in steradian.
            # 1. Resolve pixel scaling and solid angles uniformly
            pxsc_arcsec = database.red[key]["PIXSCALE"][j]
            pxsc_rad = pxsc_arcsec / ARCSEC_PER_RADIAN

            # Establish a consistent, standardized tracking variable
            pxar_telemetry = database.red[key]["PIXAR_SR"][j]

            pixel_area_sr = (
                pxar_telemetry
                if not np.isnan(pxar_telemetry)
                else calculate_pixel_area_sr(pxsc_arcsec)
            )

            # Keep old pix area variable temporarily so that things continue to work
            pxar = pixel_area_sr  # database.red[key]["PIXAR_SR"][j]  # sr

            # if np.isnan(pxar):
            #     # log.warning(
            #     #     "PIXAR_SR not found in database, falling back to use PIXSCALE"
            #     # )
            #     log.warning(
            #         f"[{key}][Index {j}] PIXAR_SR not found in telescope telemetry database. "
            #         f"Falling back to geometric pixel scale calculation."
            #     )
            #     pxsc_arcsec = database.red[key]["PIXSCALE"][j]  # arcsec
            #     # pxar = pxsc_rad**2  # sr
            #     # pxsc_rad = pxsc_arcsec / 3600.0 / 180.0 * np.pi  # rad
            #     pixel_area_sr = calculate_pixel_area_sr(
            #         pixel_scale_arcsec=pxsc_arcsec
            #     )
            #     print(f"DEGUG -- original: {pxar} | new: {pixel_area_sr}")
            #     pxar = pixel_area_sr
            # else:
            #     log.info(
            #         f"[{key}][Index {j}] PIXAR_SR found in telescope telemetry database. "
            #     )

            # Convert the host star brightness from vegamag to MJy. Use an
            # unocculted model PSF whose integrated flux is normalized to
            # one in order to obtain the theoretical peak count of the
            # star.
            log.info("Implementing extracted stellar flux calculation")
            filt = database.red[key]["FILTER"][j]
            fstar = get_stellar_peak_flux(
                starfile=star_path,
                spectral_type=spectral_type,
                instrument=instrument,
                filter_name=filt,
                observations=database.obs[key],
                output_dir=output_dir,
                **kwargs,
            )

            # Set the inner and outer working angle and compute the
            # resolution element. Account for possible blurring.
            iwa = 1  # pix
            owa = data.shape[1] // 2  # pix

            # 2. Extract instrument-level configurations safely
            telescop = database.red[key]["TELESCOP"][j]
            exp_type = database.red[key]["EXP_TYPE"][j]
            cwave_um = database.red[key]["CWAVEL"][j]
            blur_fwhm = database.obs[key]["BLURFWHM"][j]

            # 3. Call the standardized resolution tracking function
            resolution = calculate_spatial_resolution_pix(
                wavelength_um=cwave_um,
                pixel_scale_rad=pxsc_rad,  # Directly matching your linear baseline scale variable
                telescope_name=telescop,
                exposure_type=exp_type,
                blur_fwhm_pix=blur_fwhm,
            )

            print(f"Standardized Resolution tracking element: {resolution:.4f} pixels")

            # keep original config
            # resolution = resolution

            # Get the star position.
            center = get_image_center(head_pri, overwrite_crpix=overwrite_crpix)

            # Mask coronagraph spiders, 4QPM edges, etc.
            # print(center)
            # print(" --------- I am here -------")
            # print("Before mask data shape:", data.shape)
            data = apply_instrument_mask(
                data,
                center=center,
                exposure_type=database.red[key]["EXP_TYPE"][j],
                coronagraph_mask=database.red[key]["CORONMSK"][j],
                observation_types=database.obs[key]["TYPE"],
                roll_reference_angles=database.obs[key]["ROLL_REF"],
            )
            print(data.shape)  # data shape should be the same after masking

            ####################
            # Mask companions Step
            ####################
            if companions is not None:
                log.info(
                    f"  Masking out {len(companions)} known companions using provided parameters."
                )
                print(f"Resolution: {resolution}")
                print(f"pixel Scale (arcsec): {pxsc_arcsec}")
                print(f"After maskdata shape: {data.shape}")
                companion_mask = generate_companion_spatial_mask(
                    spatial_shape=data.shape[1:],
                    # spatial_shape=data.shape,
                    center_pix=center,
                    pixscale_arcsec=pxsc_arcsec,
                    spatial_resolution_pix=resolution,
                    companions=companions,
                )
                # apply the mask and fill with nans
                data[:, companion_mask] = np.nan

            ####################
            # injecting new contrast calculations:
            ####################
            log.info("  Measuring raw contrast in annuli")
            contrast_results = compute_contrast_curves(
                data_cube=data,
                pixel_area_sr=pxar,
                stellar_flux_peak=fstar,
                spatial_resolution_pix=resolution,
                center_pix=center,
                inner_working_angle_pix=iwa,
                outer_working_angle_pix=owa,
                coronagraph_transmission_mask=mask,
            )

            # NOTES: The raw contrast calculation like should be migrated to a table
            # or dataclass result structure rather than floating variables, its a result.
            # can keep things more orgnanized.

            radial_separations_pix = contrast_results.separations_pix
            raw_contrast_curves = contrast_results.raw_contrast
            throughput_corrected_array = contrast_results.throughput_corrected_contrast

            # assuming pixel scale is the same between all data.
            radial_separations_pix *= database.red[key]["PIXSCALE"][0]

            # map back to orignal setup
            # ideally we shouldnt need this because we should only need one coorindate
            # The radial separations.
            seps = np.tile(radial_separations_pix, (len(data), 1))
            cons = raw_contrast_curves
            cons_mask = throughput_corrected_array

            # PLOTTING DATA

            # Plot masked data.

            # Maksed Data plotting

            logging.info("Creating figures")

            klmodes = database.red[key]["KLMODES"][j].split(",")
            fitsfile = os.path.join(output_dir, os.path.basename(fitsfile))

            # Get PSF subtraction strategy used, for use in plot labels below.
            psfsub_strategy = (
                f"{head_pri['MODE']} with {head_pri['ANNULI']} annuli."
                if head_pri["ANNULI"] > 1
                else head_pri["MODE"]
            )
            plot_masked_data(
                data=data,
                center=center,
                pixel_scale=pxsc_arcsec,
                klmodes=klmodes,
                filter_name=filt,
                psfsub_strategy=psfsub_strategy,
                fitsfile=fitsfile,
                bunit=database.red[key]["BUNIT"][j],
            )

            plot_raw_contrast(
                separations=seps,
                contrasts=cons,
                masked_contrasts=cons_mask if mask is not None else None,
                klmodes=klmodes,
                filter_name=filt,
                psfsub_strategy=psfsub_strategy,
                fitsfile=fitsfile,
                plot_xlim=plot_xlim,
                # plot_ylim=plot_ylim,
                save_figure=save_figures,
                plot_style=plot_style,
            )

            # Exporting Data
            write_contrast_results(
                result=contrast_results,
                fitsfile=fitsfile,
                pixel_scale_arcsec=pxsc_arcsec,
                klmodes=klmodes,
                output_filetype=output_filetype,
            )


def get_image_center(primary_header, overwrite_crpix=None) -> tuple[float, float]:
    """Return the zero-indexed image center."""
    if overwrite_crpix is None:
        crpix1 = primary_header["CRPIX1"]
        crpix2 = primary_header["CRPIX2"]
    else:
        crpix1, crpix2 = overwrite_crpix

    return crpix1 - 1.0, crpix2 - 1.0


def calculate_stellar_peak_flux(
    stellar_magnitude: float,
    zero_point_flux_jy: float,
    psf_peak: float,
) -> float:
    """
    Calculate the stellar peak flux used for contrast normalization.

    Parameters
    ----------
    stellar_magnitude
        Stellar magnitude in the relevant filter.

    zero_point_flux_jy
        Photometric zero-point flux in Jy.

    psf_peak
        Peak value of the normalized unocculted model PSF.

    Returns
    -------
    float
        Stellar peak flux in MJy.
    """
    stellar_flux_jy = zero_point_flux_jy / 10.0 ** (stellar_magnitude / 2.5)

    stellar_flux_mjy = stellar_flux_jy / 1e6

    return stellar_flux_mjy * psf_peak


def get_stellar_peak_flux(
    starfile, spectral_type, instrument, filter_name, observations, output_dir, **kwargs
) -> float:
    """Determine the peak stellar flux used to normalize contrast measurements"""

    mstar, fzero = get_stellar_magnitudes(
        str(starfile),
        spectral_type,
        instrument,
        output_dir=str(output_dir),
        **kwargs,
    )

    offset_psf = get_offsetpsf(obs=observations)

    return calculate_stellar_peak_flux(
        stellar_magnitude=mstar[filter_name],
        zero_point_flux_jy=fzero[filter_name],
        psf_peak=np.nanmax(offset_psf),
    )


def write_contrast_results(
    result: ContrastResult,
    fitsfile: str | Path,
    pixel_scale_arcsec: float,
    klmodes: list[str] | tuple[str, ...] | list[int] | tuple[int, ...],
    output_filetype: str = "npy",
) -> None:
    """
    Write raw contrast results to disk.

    Parameters
    ----------
    result
        ContrastResult containing radial separations in pixels,
        raw contrast curves, and optional throughput-corrected curves.

    fitsfile
        FITS filename used to construct the output filenames.

    pixel_scale_arcsec
        Pixel scale in arcseconds per pixel.

    klmodes
        KL modes corresponding to the first axis of the contrast arrays.

    output_filetype
        Output format. Supported options are ``"npy"`` and ``"ecsv"``.

    Returns
    -------
    None
    """
    fitsfile = Path(fitsfile)
    output_stem = fitsfile.with_suffix("")

    separations_arcsec = result.separations_pix * pixel_scale_arcsec

    filetype = output_filetype.lower()

    if filetype == "npy":
        # Preserve the legacy shape: one separation array per KL mode.
        separations = np.tile(separations_arcsec, (result.raw_contrast.shape[0], 1))

        np.save(f"{output_stem}_seps.npy", separations)

        np.save(f"{output_stem}_cons.npy", result.raw_contrast)

        if result.has_throughput_correction:
            np.save(
                f"{output_stem}_cons_mask.npy", result.throughput_corrected_contrast
            )

        print(
            "Contrast results and plots saved to "
            f"{output_stem}_seps.npy, "
            f"{output_stem}_cons.npy"
        )

        return

    if filetype == "ecsv":
        columns = [separations_arcsec]
        names = ["separation"]

        for index, klmode in enumerate(klmodes):
            columns.append(result.raw_contrast[index])
            names.append(f"contrast, N_kl={klmode}")

            if result.has_throughput_correction:
                columns.append(result.throughput_corrected_contrast[index])
                names.append(f"contrast+mask, N_kl={klmode}")

        results_table = Table(columns, names=names)

        results_table["separation"].unit = u.arcsec

        output_file = Path(f"{output_stem}_contrast.ecsv")

        results_table.write(
            output_file,
            overwrite=True,
        )

        print(f"Contrast results and plots saved to {output_file}")

        return

    raise ValueError('File save format not supported, options are "npy" or "ecsv".')
