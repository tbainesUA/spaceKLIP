# TODO: TB 2024-06-05: This module is a work in progress. The legacy implementation is
# preserved for reference, but the new implementation is being developed to improve
# clarity and maintainability. The legacy code is commented out, and the new functions
# are being built incrementally.
#
# Masking features should return boolean/binary arrays and should not modify the input
# data in place. A caller should be responsible for applying the mask to the data.


import logging

import numpy as np
from scipy.ndimage import rotate

from spaceKLIP.utils import set_surrounded_pixels

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def apply_instrument_mask(
    data,
    center,
    exposure_type,
    coronagraph_mask,
    observation_types,
    roll_reference_angles,
):
    """building and applying the instrument masks to the data. This function takes the
    legacy implementation, and extracted it contents into their own functions. Unmutated
    data is also returned.
    """

    # Extract the roll angles of the science frames.
    science_indices = np.where(observation_types == "SCI")[0]
    science_roll_angles = roll_reference_angles[science_indices]

    # Mask coronagraph spiders, 4QPM edges, etc.
    if exposure_type in ["NRC_CORON"]:
        return build_nrc_coron_bar_mask_legacy(
            data=data,
            coronagraph_mask=coronagraph_mask,
            center=center,
            roll_reference_angles=science_roll_angles,
        )
    elif exposure_type in ["MIR_4QPM"]:
        log.info("  Masking out areas for MIRI 4QPM coronagraph")

        instrument_mask = build_miri_4qpm_mask_legacy(
            data=data, center=center, roll_reference_angles=science_roll_angles
        )

        data *= instrument_mask
        return data
    elif exposure_type in ["MIR_LYOT"]:
        raise NotImplementedError()

    return data


def build_nrc_coron_bar_mask_legacy(
    data, coronagraph_mask, center, roll_reference_angles
):
    """
    Extracted from the exposure_type == "NRC_CORON" / "WB" in
    coronagraph_mask branch, with the observation_types dependency
    removed. Caller is responsible for passing only the roll angles
    that should be masked (e.g. pre-filtered to SCI frames).
    """
    if "WB" in coronagraph_mask:
        log.info("  Masking out areas for NIRCam bar coronagraph")
        xr = np.arange(data.shape[-1]) - center[0]
        yr = np.arange(data.shape[-2]) - center[1]
        xx, yy = np.meshgrid(xr, yr)
        pa = -np.rad2deg(np.arctan2(xx, yy))
        pa[pa < 0.0] += 360.0
        for roll_ref in roll_reference_angles:
            pa1 = (90.0 - 15.0 + roll_ref) % 360.0
            pa2 = (90.0 + 15.0 + roll_ref) % 360.0
            if pa1 > pa2:
                temp = (pa > pa1) | (pa < pa2)
            else:
                temp = (pa > pa1) & (pa < pa2)
            data[:, temp] = np.nan
            pa1 = (270.0 - 15.0 + roll_ref) % 360.0
            pa2 = (270.0 + 15.0 + roll_ref) % 360.0
            if pa1 > pa2:
                temp = (pa > pa1) | (pa < pa2)
            else:
                temp = (pa > pa1) & (pa < pa2)
            data[:, temp] = np.nan
        return data


def build_miri_4qpm_mask_legacy(data, center, roll_reference_angles):
    """
    Extracted as-is from the exposure_type == "MIR_4QPM" branch,
    with the observation_types dependency removed. Caller is
    responsible for passing only the roll angles that should be
    masked (e.g. pre-filtered to SCI frames).

    Depends on `rotate` (scipy.ndimage) and `set_surrounded_pixels`
    being available/imported exactly as in the original module.
    """
    # This is MIRI 4QPM data, want to mask edges. However, close
    # to the center you don't have a choice. So, want to use
    # rectangles with a gap in the center.
    print("  Masking out areas for MIRI 4QPM coronagraph")

    # Create array and pad slightly
    nanmask = np.zeros_like(data[0])
    pad = 5
    nanmask = np.pad(nanmask, pad)

    # Upsample array to improve centering.
    samp = 1  # Upsampling factor
    nanmask = nanmask.repeat(samp, axis=0).repeat(samp, axis=1)

    # Define rectangle edges
    rect_width = 10 * samp  # pixels
    thinrect_width = 2 * samp  # pixels

    cent_rect = [
        (center[0] + pad) * samp,
        (center[0] + pad) * samp,
        (center[1] + pad) * samp,
        (center[1] + pad) * samp,
    ]
    rect = [int(cent_rect[i] - (rect_width / 2 * (-1) ** (i % 2))) for i in range(4)]
    thinrect = [
        int(cent_rect[i] - (thinrect_width / 2 * (-1) ** (i % 2))) for i in range(4)
    ]

    # Define circle mask for center
    circ_rad = 15 * samp  # pixels
    yarr, xarr = np.ogrid[: nanmask.shape[0], : nanmask.shape[1]]
    rad_dist = np.sqrt(
        (xarr - (center[0] + pad) * samp) ** 2 + (yarr - (center[1] + pad) * samp) ** 2
    )
    circ = rad_dist < circ_rad

    # Loop over images
    for roll_ref in roll_reference_angles:
        # Apply cross
        temp = np.zeros_like(nanmask)
        temp[:, rect[0] : rect[1]] = 1  # Vertical
        temp[rect[2] : rect[3], :] = 1  # Horizontal

        # Now ensure center isn't completely masked
        temp[circ] = 0

        # Apply thin cross
        temp[:, thinrect[0] : thinrect[1]] = 1  # Vertical
        temp[thinrect[2] : thinrect[3], :] = 1  # Horizontal

        # Rotate the array, include fixed rotation of FQPM edges
        # temp = rotate(temp, 90 - roll_ref + 4.83544897, reshape=False)
        temp = rotate(
            temp,
            90 - roll_ref + 4.83544897,
            reshape=False,
            order=0,
            # mode="contant",
            # cval=0.0,
            # prefilter=False,
        )
        nanmask += temp

    # If pixel value too high, should be masked, else set to 1.
    nanmask[nanmask >= 0.5] = np.nan
    nanmask[nanmask < 0.5] = 1

    # Downsample, remove padding, and mask data
    nanmask = nanmask[::samp, ::samp]
    nanmask = nanmask[pad:-pad, pad:-pad]
    nanmask = set_surrounded_pixels(nanmask)
    # data *= nanmask
    return nanmask


# # new miri masking function
# MIRI_4QPM_PADDING_PIXELS = 5
# MIRI_4QPM_UPSAMPLE_FACTOR = 1
# MIRI_4QPM_WIDE_CROSS_WIDTH_PIXELS = 10
# MIRI_4QPM_THIN_CROSS_WIDTH_PIXELS = 2
# MIRI_4QPM_CENTER_RADIUS_PIXELS = 15
# MIRI_4QPM_ROTATION_REFERENCE_DEG = 90.0
# MIRI_4QPM_ROTATION_OFFSET_DEG = 4.83544897
# MIRI_4QPM_MASK_THRESHOLD = 0.5


# def radial_distance(x, y):
#     return np.hypot(x, y)


# def create_miri_4qpm_mask_legacy(
#     image_shape: tuple[int, int],
#     center: tuple[float, float],
#     roll_reference_angles: Sequence[float] | NDArray[np.floating],
# ) -> NDArray[np.floating]:
#     """
#     Construct the legacy MIRI 4QPM geometric instrument mask.

#     This function preserves the original raster-based implementation used
#     inside ``raw_contrast``. A cross-shaped mask is constructed, rotated for
#     each science roll angle, accumulated, thresholded, cropped, and processed
#     with ``set_surrounded_pixels``.

#     Parameters
#     ----------
#     image_shape
#         Shape of one reduced image as ``(ny, nx)``.
#     center
#         Stellar center as ``(x, y)`` in zero-indexed pixel coordinates.
#     roll_reference_angles
#         Science observation roll-reference angles in degrees.

#     Returns
#     -------
#     numpy.ndarray
#         Two-dimensional floating-point mask with shape ``image_shape``.
#         Usable pixels are 1 and masked pixels are NaN.

#     Notes
#     -----
#     This is a frozen compatibility implementation. It intentionally preserves
#     the original use of raster rotation, interpolation, accumulation before
#     thresholding, and rotation about the padded array center.
#     """
#     ny, nx = image_shape

#     if ny <= 0 or nx <= 0:
#         raise ValueError(
#             f"image_shape must contain positive dimensions; received {image_shape!r}."
#         )

#     if len(center) != 2:
#         raise ValueError("center must contain exactly two values as (x, y).")

#     center_x, center_y = center

#     pad = MIRI_4QPM_PADDING_PIXELS
#     sample = MIRI_4QPM_UPSAMPLE_FACTOR

#     # Preserve the original construction based on a zero-valued image.
#     accumulated_mask = np.zeros(image_shape, dtype=float)
#     accumulated_mask = np.pad(accumulated_mask, pad)

#     # Preserve the original upsampling strategy.
#     accumulated_mask = accumulated_mask.repeat(sample, axis=0).repeat(sample, axis=1)

#     wide_width = MIRI_4QPM_WIDE_CROSS_WIDTH_PIXELS * sample
#     thin_width = MIRI_4QPM_THIN_CROSS_WIDTH_PIXELS * sample

#     padded_center_x = (center_x + pad) * sample
#     padded_center_y = (center_y + pad) * sample

#     center_coordinates = [
#         padded_center_x,
#         padded_center_x,
#         padded_center_y,
#         padded_center_y,
#     ]

#     wide_edges = [
#         int(center_coordinates[index] - wide_width / 2 * (-1) ** (index % 2))
#         for index in range(4)
#     ]

#     thin_edges = [
#         int(center_coordinates[index] - thin_width / 2 * (-1) ** (index % 2))
#         for index in range(4)
#     ]

#     center_radius = MIRI_4QPM_CENTER_RADIUS_PIXELS * sample

#     y_grid, x_grid = np.ogrid[: accumulated_mask.shape[0], : accumulated_mask.shape[1]]

#     r_grid = radial_distance(x_grid - padded_center_x, y_grid - padded_center_y)

#     center_region = r_grid < center_radius

#     for roll_reference_angle in np.asarray(
#         roll_reference_angles,
#         dtype=float,
#     ):
#         roll_mask = np.zeros_like(accumulated_mask)

#         # Wide vertical and horizontal cross.
#         roll_mask[:, wide_edges[0] : wide_edges[1]] = 1
#         roll_mask[wide_edges[2] : wide_edges[3], :] = 1

#         # Remove the wide cross near the center.
#         roll_mask[center_region] = 0

#         # Restore a thin cross through the center.
#         roll_mask[:, thin_edges[0] : thin_edges[1]] = 1
#         roll_mask[thin_edges[2] : thin_edges[3], :] = 1

#         # calculate rotation angle
#         rotation_angle = (
#             MIRI_4QPM_ROTATION_REFERENCE_DEG
#             - roll_reference_angle
#             + MIRI_4QPM_ROTATION_OFFSET_DEG
#         )

#         roll_mask = rotate(roll_mask, rotation_angle, reshape=False)

#         accumulated_mask += roll_mask

#     accumulated_mask[accumulated_mask >= MIRI_4QPM_MASK_THRESHOLD] = np.nan

#     accumulated_mask[accumulated_mask < MIRI_4QPM_MASK_THRESHOLD] = 1

#     # Preserve the original downsampling operation.
#     accumulated_mask = accumulated_mask[::sample, ::sample]

#     # Remove the padding.
#     accumulated_mask = accumulated_mask[pad:-pad, pad:-pad]

#     accumulated_mask = set_surrounded_pixels(accumulated_mask)

#     return accumulated_mask


# UNCOMMENT CODE FOR LEGACY IMPLEMENTATION BELOW.
# def apply_instrument_mask_legacy(
#     data,
#     center,
#     exposure_type,
#     coronagraph_mask,
#     observation_type,
#     roll_reference_angles,
# ):
#     # Mask coronagraph spiders, 4QPM edges, etc.
#     if exposure_type in ["NRC_CORON"]:
#         if "WB" in coronagraph_mask:
#             log.info("  Masking out areas for NIRCam bar coronagraph")
#             xr = np.arange(data.shape[-1]) - center[0]
#             yr = np.arange(data.shape[-2]) - center[1]
#             xx, yy = np.meshgrid(xr, yr)
#             pa = -np.rad2deg(np.arctan2(xx, yy))
#             pa[pa < 0.0] += 360.0
#             ww_sci = np.where(observation_type == "SCI")[0]
#             for ww in ww_sci:
#                 roll_ref = roll_reference_angles[ww]  # deg
#                 pa1 = (90.0 - 15.0 + roll_ref) % 360.0
#                 pa2 = (90.0 + 15.0 + roll_ref) % 360.0
#                 if pa1 > pa2:
#                     temp = (pa > pa1) | (pa < pa2)
#                 else:
#                     temp = (pa > pa1) & (pa < pa2)
#                 data[:, temp] = np.nan
#                 pa1 = (270.0 - 15.0 + roll_ref) % 360.0
#                 pa2 = (270.0 + 15.0 + roll_ref) % 360.0
#                 if pa1 > pa2:
#                     temp = (pa > pa1) | (pa < pa2)
#                 else:
#                     temp = (pa > pa1) & (pa < pa2)
#                 data[:, temp] = np.nan
#             return data
#     elif exposure_type in ["MIR_4QPM"]:
#         # This is MIRI 4QPM data, want to mask edges. However, close
#         # to the center you don't have a choice. So, want to use
#         # rectangles with a gap in the center.
#         log.info("  Masking out areas for MIRI 4QPM coronagraph")

#         # Create array and pad slightly
#         nanmask = np.zeros_like(data[0])
#         pad = 5
#         nanmask = np.pad(nanmask, pad)

#         # Upsample array to improve centering.
#         samp = 1  # Upsampling factor
#         nanmask = nanmask.repeat(samp, axis=0).repeat(samp, axis=1)

#         # Define rectangle edges
#         rect_width = 10 * samp  # pixels
#         thinrect_width = 2 * samp  # pixels

#         cent_rect = [
#             (center[0] + pad) * samp,
#             (center[0] + pad) * samp,
#             (center[1] + pad) * samp,
#             (center[1] + pad) * samp,
#         ]
#         rect = [
#             int(cent_rect[i] - (rect_width / 2 * (-1) ** (i % 2))) for i in range(4)
#         ]
#         thinrect = [
#             int(cent_rect[i] - (thinrect_width / 2 * (-1) ** (i % 2))) for i in range(4)
#         ]

#         # Define circle mask for center
#         circ_rad = 15 * samp  # pixels
#         yarr, xarr = np.ogrid[: nanmask.shape[0], : nanmask.shape[1]]
#         rad_dist = np.sqrt(
#             (xarr - (center[0] + pad) * samp) ** 2
#             + (yarr - (center[1] + pad) * samp) ** 2
#         )
#         circ = rad_dist < circ_rad

#         # Loop over images
#         ww_sci = np.where(observation_type == "SCI")[0]
#         for ww in ww_sci:
#             # Apply cross
#             roll_ref = roll_reference_angles[ww]  # deg
#             temp = np.zeros_like(nanmask)
#             temp[:, rect[0] : rect[1]] = 1  # Vertical
#             temp[rect[2] : rect[3], :] = 1  # Horizontal

#             # Now ensure center isn't completely masked
#             temp[circ] = 0

#             # Apply thin cross
#             temp[:, thinrect[0] : thinrect[1]] = 1  # Vertical
#             temp[thinrect[2] : thinrect[3], :] = 1  # Horizontal

#             # Rotate the array, include fixed rotation of FQPM edges
#             temp = rotate(temp, 90 - roll_ref + 4.83544897, reshape=False)
#             nanmask += temp

#         # If pixel value too high, should be masked, else set to 1.
#         nanmask[nanmask >= 0.5] = np.nan
#         nanmask[nanmask < 0.5] = 1

#         # Downsample, remove padding, and mask data
#         nanmask = nanmask[::samp, ::samp]
#         nanmask = nanmask[pad:-pad, pad:-pad]
#         nanmask = set_surrounded_pixels(nanmask)
#         data *= nanmask
#         return data
#     elif exposure_type in ["MIR_LYOT"]:
#         raise NotImplementedError()

#     return data
