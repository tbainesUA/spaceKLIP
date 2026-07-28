# data = apply_instrument_mask(
#     data,
#     center=center,
#     exposure_type=exposure_type,
#     coronagraph_mask=self.database.red[key]["CORONMSK"][j],
#     observation_types=observation_type,
#     roll_reference_angles=self.database.obs[key]["ROLL_REF"],
# )

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
    observation_type,
    roll_reference_angles,
):
    # Mask coronagraph spiders, 4QPM edges, etc.
    if exposure_type in ["NRC_CORON"]:
        if "WB" in coronagraph_mask:
            log.info("  Masking out areas for NIRCam bar coronagraph")
            xr = np.arange(data.shape[-1]) - center[0]
            yr = np.arange(data.shape[-2]) - center[1]
            xx, yy = np.meshgrid(xr, yr)
            pa = -np.rad2deg(np.arctan2(xx, yy))
            pa[pa < 0.0] += 360.0
            ww_sci = np.where(observation_type == "SCI")[0]
            for ww in ww_sci:
                roll_ref = roll_reference_angles[ww]  # deg
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
    elif exposure_type in ["MIR_4QPM"]:
        # This is MIRI 4QPM data, want to mask edges. However, close
        # to the center you don't have a choice. So, want to use
        # rectangles with a gap in the center.
        log.info("  Masking out areas for MIRI 4QPM coronagraph")

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
        rect = [
            int(cent_rect[i] - (rect_width / 2 * (-1) ** (i % 2))) for i in range(4)
        ]
        thinrect = [
            int(cent_rect[i] - (thinrect_width / 2 * (-1) ** (i % 2))) for i in range(4)
        ]

        # Define circle mask for center
        circ_rad = 15 * samp  # pixels
        yarr, xarr = np.ogrid[: nanmask.shape[0], : nanmask.shape[1]]
        rad_dist = np.sqrt(
            (xarr - (center[0] + pad) * samp) ** 2
            + (yarr - (center[1] + pad) * samp) ** 2
        )
        circ = rad_dist < circ_rad

        # Loop over images
        ww_sci = np.where(observation_type == "SCI")[0]
        for ww in ww_sci:
            # Apply cross
            roll_ref = roll_reference_angles[ww]  # deg
            temp = np.zeros_like(nanmask)
            temp[:, rect[0] : rect[1]] = 1  # Vertical
            temp[rect[2] : rect[3], :] = 1  # Horizontal

            # Now ensure center isn't completely masked
            temp[circ] = 0

            # Apply thin cross
            temp[:, thinrect[0] : thinrect[1]] = 1  # Vertical
            temp[thinrect[2] : thinrect[3], :] = 1  # Horizontal

            # Rotate the array, include fixed rotation of FQPM edges
            temp = rotate(temp, 90 - roll_ref + 4.83544897, reshape=False)
            nanmask += temp

        # If pixel value too high, should be masked, else set to 1.
        nanmask[nanmask >= 0.5] = np.nan
        nanmask[nanmask < 0.5] = 1

        # Downsample, remove padding, and mask data
        nanmask = nanmask[::samp, ::samp]
        nanmask = nanmask[pad:-pad, pad:-pad]
        nanmask = set_surrounded_pixels(nanmask)
        data *= nanmask
        return data
    elif exposure_type in ["MIR_LYOT"]:
        raise NotImplementedError()
