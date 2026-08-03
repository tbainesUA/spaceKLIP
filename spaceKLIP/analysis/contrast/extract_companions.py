from __future__ import division

import copy
import logging

# =============================================================================
# IMPORTS
# =============================================================================
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pyklip.fakes as fakes
from pyklip import parallelized

from spaceKLIP.starphot import get_stellar_magnitudes

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def run_extract_companions(
    database,
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

    # Check input.
    kwargs_temp = {}

    # Set output directory.
    output_dir = os.path.join(database.database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through concatenations.
    for i, key in enumerate(database.database.red.keys()):
        log.info("--> Concatenation " + key)

        # Loop through FITS files.
        nfitsfiles = len(database.database.red[key])
        for j in range(nfitsfiles):
            # Get stellar magnitudes and filter zero points.
            mstar, fzero, fzero_si = get_stellar_magnitudes(
                starfile,
                spectral_type,
                database.database.red[key]["INSTRUME"][j],
                return_si=True,
                output_dir=output_dir,
                **kwargs,
            )  # vegamag, Jy, erg/cm^2/s/A

            # Get COM substrate throughput.
            tp_comsubst = ut.get_tp_comsubst(
                database.database.red[key]["INSTRUME"][j],
                database.database.red[key]["SUBARRAY"][j],
                database.database.red[key]["FILTER"][j],
            )

            # Compute the pixel area in steradian.
            pxsc_arcsec = database.database.red[key]["PIXSCALE"][j]  # arcsec
            pxsc_rad = pxsc_arcsec / 3600.0 / 180.0 * np.pi  # rad
            pxar = database.database.red[key]["PIXAR_SR"][j]  # sr
            if np.isnan(pxar):
                log.warning(
                    "PIXAR_SR not found in database, falling back to use PIXSCALE"
                )
                pxar = pxsc_rad**2  # sr

            # Compute the resolution element. Account for possible
            # blurring.
            if database.database.red[key]["TELESCOP"][j] == "JWST":
                if database.database.red[key]["EXP_TYPE"][j] in ["NRC_CORON"]:
                    diam = 5.2
                else:
                    diam = JWST_CIRCUMSCRIBED_DIAMETER
            else:
                raise UserWarning("Data originates from unknown telescope")
            resolution = (
                1e-6 * database.database.red[key]["CWAVEL"][j] / diam / pxsc_rad
            )  # pix
            if not np.isnan(database.database.obs[key]["BLURFWHM"][j]):
                resolution = np.hypot(
                    resolution, database.database.obs[key]["BLURFWHM"][j]
                )

            # Find science and reference files.
            filepaths, psflib_filepaths, maxnumbasis = get_pyklip_filepaths(
                database.database, key, return_maxbasis=True
            )
            if (
                "maxnumbasis" not in kwargs_temp.keys()
                or kwargs_temp["maxnumbasis"] is None
            ):
                kwargs_temp["maxnumbasis"] = maxnumbasis

            # Initialize pyKLIP dataset.
            pop_pxar_kw(np.append(filepaths, psflib_filepaths))
            dataset = JWSTData(filepaths, psflib_filepaths, highpass=highpass)
            kwargs_temp["dataset"] = dataset
            kwargs_temp["aligned_center"] = dataset._centers[0]
            kwargs_temp["psf_library"] = dataset.psflib

            # Make copy of the original pyKLIP dataset.
            dataset_orig = copy.deepcopy(dataset)

            # Get index of desired KL mode.
            klmodes = database.database.red[key]["KLMODES"][j].split(",")
            klmodes = np.array([int(temp) for temp in klmodes])
            if klmode == "max":
                klindex = np.argmax(klmodes)
            else:
                klindex = klmodes.tolist().index(klmode)

            # Set output directories.
            output_dir_kl = os.path.join(output_dir, "KL%.0f" % klmodes[klindex])
            if not os.path.exists(output_dir_kl):
                os.makedirs(output_dir_kl)

            # Initialize a function that can generate model offset PSFs.
            inst = database.database.red[key]["INSTRUME"][j]
            filt = database.database.red[key]["FILTER"][j]
            apername = database.database.red[key]["APERNAME"][j]
            if database.database.red[key]["TELESCOP"][j] == "JWST":
                if inst == "NIRCAM":
                    pass
                    # image_mask = database.database.red[key]['CORONMSK'][j]
                    # image_mask = image_mask[:4] + image_mask[5:]
                elif inst == "NIRISS":
                    raise NotImplementedError()
                elif inst == "MIRI":
                    pass
                    # image_mask = database.database.red[key]['CORONMSK'][j].replace('4QPM_', 'FQPM')
                else:
                    raise UserWarning("Data originates from unknown JWST instrument")
            else:
                raise UserWarning("Data originates from unknown telescope")
            if planetfile is None:
                # Keep backward compatibility where planetfile was part of
                # the kwargs.
                if "planetfile" in kwargs.keys() and kwargs["planetfile"] is not None:
                    planetfile = kwargs["planetfile"]
            if planetfile is not None:
                sed = read_spec_file(planetfile)
            else:
                sed = None
            ww_sci = np.where(database.database.obs[key]["TYPE"] == "SCI")[0]
            if date is not None:
                if date == "auto":
                    date = fits.getheader(
                        database.database.obs[key]["FITSFILE"][ww_sci[0]], 0
                    )["DATE-BEG"]
            offsetpsf_func = JWST_PSF(
                apername,
                filt,
                date=date,
                fov_pix=65,
                oversample=2,
                sp=sed,
                use_coeff=False,
            )

            # NOTE: if minmethod not None, it will split the fit into a fitmethod (e.g. mcmc) for the estimation
            # of position and flux, and a minmethod (e.g. Powell) to fit the extension of the source using a
            # psf convolved by a 2D Gaussian and a minimization approach.
            if minmethod is not None:
                split_fit = True
            else:
                split_fit = False

            if split_fit:
                if not all(
                    x in kwargs.keys()
                    for x in [
                        "sigma_xguess",
                        "sigma_yguess",
                        "scale_guess",
                        "theta_guess",
                    ]
                ):
                    gauss_param_guesses = [0.3, 0.3, 0, 0]
                else:
                    gauss_param_guesses = [
                        kwargs["sigma_xguess"],
                        kwargs["sigma_yguess"],
                        kwargs["scale_guess"],
                        kwargs["theta_guess"],
                    ]

                # Loop through companions.
                tab = Table(
                    names=(
                        "ID",
                        "RA",
                        "RA_ERR",
                        "DEC",
                        "DEC_ERR",
                        "FLUX_JY",
                        "FLUX_JY_ERR",
                        "FLUX_SI",
                        "FLUX_SI_ERR",
                        "FLUX_SI_ALT",
                        "FLUX_SI_ALT_ERR",
                        "CON",
                        "CON_ERR",
                        "DELMAG",
                        "DELMAG_ERR",
                        "APPMAG",
                        "APPMAG_ERR",
                        "MSTAR",
                        "MSTAR_ERR",
                        "SNR",
                        "LN(Z/Z0)",
                        "TP_CORONMSK",
                        "TP_COMSUBST",
                        "FITSFILE",
                        "GSCALE",
                        "GSCALE_ERROR",
                        "SIGMA_X",
                        "SIGMA_X_ERROR",
                        "SIGMA_Y",
                        "SIGMA_Y_ERROR",
                        "THETA",
                        "THETA_ERROR",
                    ),
                    dtype=(
                        "int",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "object",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                    ),
                )
            else:
                # Loop through companions.
                tab = Table(
                    names=(
                        "ID",
                        "RA",
                        "RA_ERR",
                        "DEC",
                        "DEC_ERR",
                        "FLUX_JY",
                        "FLUX_JY_ERR",
                        "FLUX_SI",
                        "FLUX_SI_ERR",
                        "FLUX_SI_ALT",
                        "FLUX_SI_ALT_ERR",
                        "CON",
                        "CON_ERR",
                        "DELMAG",
                        "DELMAG_ERR",
                        "APPMAG",
                        "APPMAG_ERR",
                        "MSTAR",
                        "MSTAR_ERR",
                        "SNR",
                        "LN(Z/Z0)",
                        "TP_CORONMSK",
                        "TP_COMSUBST",
                        "FITSFILE",
                    ),
                    dtype=(
                        "int",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "object",
                    ),
                )
            for k in range(len(companions)):
                output_dir_comp = os.path.join(output_dir_kl, "C%.0f" % (k + 1))
                if not os.path.exists(output_dir_comp):
                    os.makedirs(output_dir_comp)
                output_dir_fm = os.path.join(output_dir_comp, "KLIP_FM")
                if not os.path.exists(output_dir_fm):
                    os.makedirs(output_dir_fm)
                if save_preklip:
                    output_dir_pk = os.path.join(output_dir_comp, "PREKLIP")
                    if not os.path.exists(output_dir_pk):
                        os.makedirs(output_dir_pk)

                # Offset PSF that is not affected by the coronagraphic
                # mask, but only the Lyot stop.
                psf_no_coronmsk = offsetpsf_func.psf_off

                # Initial guesses for the fit parameters.
                guess_dx = companions[k][0] / pxsc_arcsec  # pix
                guess_dy = companions[k][1] / pxsc_arcsec  # pix
                guess_flux = companions[k][2]  # contrast
                guess_spec = np.array([1.0])
                guess_sep = np.sqrt(guess_dx**2 + guess_dy**2)  # pix
                guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy))  # deg

                # The initial guesses are made in RA/Dec space, but the
                # model PSFs are defined by the offset between the
                # coronagraphic mask center and the companion. Hence, we
                # need to generate a separate model PSF for each roll.
                rot_offsetpsfs = []
                sci_totinttime = []
                all_offsetpsfs = []
                all_offsetpsfs_nohpf = []
                all_pas = []
                scale_factor_avg = []
                for ww in ww_sci:
                    roll_ref = database.database.obs[key]["ROLL_REF"][ww]  # deg

                    # Get shift between star and coronagraphic mask
                    # position. If positive, the coronagraphic mask center
                    # is to the left/bottom of the star position.
                    _, _, _, _, _, _, _, _, _, _, maskoffs = ut.read_obs(
                        database.database.obs[key]["FITSFILE"][ww]
                    )

                    # NIRCam.
                    if maskoffs is not None:
                        mask_xoff = -maskoffs[:, 0]  # pix
                        mask_yoff = -maskoffs[:, 1]  # pix

                        # Need to rotate by the roll angle (CCW) and flip
                        # the x-axis so that positive RA is to the left.
                        mask_raoff = -(
                            mask_xoff * np.cos(np.deg2rad(roll_ref))
                            - mask_yoff * np.sin(np.deg2rad(roll_ref))
                        )  # pix
                        mask_deoff = mask_xoff * np.sin(
                            np.deg2rad(roll_ref)
                        ) + mask_yoff * np.cos(np.deg2rad(roll_ref))  # pix

                        # Compute the true offset between the companion and
                        # the coronagraphic mask center.
                        sim_dx = guess_dx - mask_raoff  # pix
                        sim_dy = guess_dy - mask_deoff  # pix
                        sim_sep = np.sqrt(sim_dx**2 + sim_dy**2) * pxsc_arcsec  # arcsec
                        sim_pa = np.rad2deg(np.arctan2(sim_dx, sim_dy))  # deg

                        # Take median of observation. Typically, each
                        # dither position is a separate observation.
                        sim_sep = np.median(sim_sep)
                        sim_pa = np.median(sim_pa)

                    # Otherwise.
                    else:
                        sim_sep = (
                            np.sqrt(guess_dx**2 + guess_dy**2) * pxsc_arcsec
                        )  # arcsec
                        sim_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy))  # deg

                    # Generate offset PSF for this roll angle. Do not add
                    # the V3Yidl angle as it has already been added to the
                    # roll angle by spaceKLIP. This is only for estimating
                    # the coronagraphic mask throughput!
                    offsetpsf_coronmsk = offsetpsf_func.gen_psf(
                        [sim_sep, sim_pa],
                        mode="rth",
                        PA_V3=roll_ref,
                        do_shift=False,
                        quick=True,
                        addV3Yidl=False,
                    )

                    # Coronagraphic mask throughput is not incorporated
                    # into the flux calibration of the JWST pipeline so
                    # that the companion flux from the detector pixels will
                    # be underestimated. Therefore, we need to scale the
                    # model offset PSF to account for the coronagraphic
                    # mask throughput (it becomes fainter). Compute scale
                    # factor by comparing a model PSF with and without
                    # coronagraphic mask.
                    scale_factor = np.sum(offsetpsf_coronmsk) / np.sum(psf_no_coronmsk)
                    scale_factor_avg += [scale_factor]

                    # Normalize model offset PSF to a total integrated flux
                    # of 1 at infinity. Generates a new webbpsf model with
                    # PSF normalization set to 'exit_pupil'.
                    offsetpsf = offsetpsf_func.gen_psf(
                        [sim_sep, sim_pa],
                        mode="rth",
                        PA_V3=roll_ref,
                        do_shift=False,
                        quick=False,
                        addV3Yidl=False,
                        normalize="exit_pupil",
                    )

                    # Normalize model offset PSF by the flux of the star.
                    offsetpsf *= (
                        fzero[filt] / 10 ** (mstar[filt] / 2.5) / 1e6 / pxar
                    )  # MJy/sr

                    # Apply scale factor to incorporate the coronagraphic
                    # mask througput.
                    # NOTE: There is no need to apply a correction for the substrate
                    # as this is accounted for by the pipeline.
                    offsetpsf *= scale_factor

                    # Flip model PSF in x or y direction if requested
                    if flip_fmpsf_xy is not None:
                        if flip_fmpsf_xy == "x":
                            offsetpsf = np.fliplr(offsetpsf)
                        elif flip_fmpsf_xy == "y":
                            offsetpsf = np.flipud(offsetpsf)
                        elif flip_fmpsf_xy == "xy" or flip_fmpsf_xy == "yx":
                            offsetpsf = np.flipud(np.fliplr(offsetpsf))
                        else:
                            raise ValueError(
                                'flip_fmpsf_xy must be "x", "y", "xy", or "yx".'
                            )

                    # Blur frames with a Gaussian filter.
                    if not np.isnan(database.database.obs[key]["BLURFWHM"][ww]):
                        gauss_sigma = database.database.obs[key]["BLURFWHM"][
                            j
                        ] / np.sqrt(8.0 * np.log(2.0))
                        offsetpsf = gaussian_filter(offsetpsf, gauss_sigma)

                    # Apply high-pass filter.
                    offsetpsf_nohpf = copy.deepcopy(offsetpsf)
                    if not isinstance(highpass, bool):
                        highpass = float(highpass)
                        fourier_sigma_size = (offsetpsf.shape[0] / highpass) / (
                            2.0 * np.sqrt(2.0 * np.log(2.0))
                        )
                        offsetpsf = parallelized.high_pass_filter_imgs(
                            np.array([offsetpsf]),
                            numthreads=None,
                            filtersize=fourier_sigma_size,
                        )[0]
                    else:
                        if highpass:
                            raise NotImplementedError()

                    # Save rotated model offset PSFs in case we do not end
                    # up using FM.
                    nints = database.database.obs[key]["NINTS"][ww]
                    effinttm = database.database.obs[key]["EFFINTTM"][ww]
                    rot_offsetpsf = rotate(
                        offsetpsf,
                        -roll_ref,
                        reshape=False,
                        mode="constant",
                        cval=0.0,
                    )
                    rot_offsetpsfs.extend([rot_offsetpsf])  # do not duplicate
                    sci_totinttime.extend([nints * effinttm])

                    # Save non-rotated model offset PSFs for the FM.
                    all_offsetpsfs.extend([offsetpsf for ni in range(nints)])
                    all_offsetpsfs_nohpf.extend(
                        [offsetpsf_nohpf for ni in range(nints)]
                    )
                    all_pas.extend([roll_ref for ni in range(nints)])
                scale_factor_avg = np.sum(
                    [
                        scale_factor_avg[l] * sci_totinttime[l] / np.sum(sci_totinttime)
                        for l in range(len(scale_factor_avg))
                    ]
                )

                # Compute the FM dataset if it does not exist yet, or if
                # overwrite is True.
                # Compute the FM dataset.
                mode = database.database.red[key]["MODE"][j]
                annuli = int(database.database.red[key]["ANNULI"][j])
                subsections = int(database.database.red[key]["SUBSECTS"][j])
                fmdataset = os.path.join(
                    output_dir_fm,
                    "FM-"
                    + mode
                    + "_NANNU"
                    + str(annuli)
                    + "_NSUBS"
                    + str(subsections)
                    + "_"
                    + key
                    + "-fmpsf-KLmodes-all.fits",
                )
                klipdataset = os.path.join(
                    output_dir_fm,
                    "FM-"
                    + mode
                    + "_NANNU"
                    + str(annuli)
                    + "_NSUBS"
                    + str(subsections)
                    + "_"
                    + key
                    + "-klipped-KLmodes-all.fits",
                )
                if overwrite or (
                    not os.path.exists(fmdataset) or not os.path.exists(klipdataset)
                ):
                    # Initialize the pyKLIP FM class. Use sep/pa relative
                    # to the star and not the coronagraphic mask center.
                    input_wvs = np.unique(dataset.wvs)
                    if len(input_wvs) != 1:
                        raise NotImplementedError(
                            "Only implemented for broadband photometry"
                        )
                    fm_class = fmpsf.FMPlanetPSF(
                        inputs_shape=dataset.input.shape,
                        numbasis=klmodes,
                        sep=guess_sep,
                        pa=guess_pa,
                        dflux=guess_flux,
                        input_psfs=np.array(all_offsetpsfs),
                        input_wvs=input_wvs,
                        spectrallib=[guess_spec],
                        spectrallib_units="contrast",
                        field_dependent_correction=None,
                        input_psfs_pas=all_pas,
                    )

                    if not isinstance(highpass, bool):
                        if k == 0:
                            highpass_temp = float(highpass)
                        else:
                            highpass_temp = False
                    else:
                        if highpass:
                            raise NotImplementedError()
                        else:
                            highpass_temp = False
                    fm.klip_dataset(
                        dataset=dataset,
                        fm_class=fm_class,
                        mode=mode,
                        outputdir=output_dir_fm,
                        fileprefix="FM-"
                        + mode
                        + "_NANNU"
                        + str(annuli)
                        + "_NSUBS"
                        + str(subsections)
                        + "_"
                        + key,
                        annuli=annuli,
                        subsections=subsections,
                        movement=1.0,
                        numbasis=klmodes,
                        maxnumbasis=maxnumbasis,
                        calibrate_flux=False,
                        aligned_center=dataset._centers[0],
                        psf_library=dataset.psflib,
                        highpass=highpass_temp,
                        mute_progression=True,
                    )

                # Open the FM dataset.
                with fits.open(fmdataset) as hdul:
                    fm_frame = hdul[0].data[klindex]
                    fm_centx = hdul[0].header["PSFCENTX"]
                    fm_centy = hdul[0].header["PSFCENTY"]
                with fits.open(klipdataset) as hdul:
                    data_frame = hdul[0].data[klindex]
                    data_centx = hdul[0].header["PSFCENTX"]
                    data_centy = hdul[0].header["PSFCENTY"]

                # If use_fm_psf is False, then replace the FM PSF in the
                # fm_frame with an integration time-averaged model offset
                # PSF.
                if use_fm_psf == False:
                    av_offsetpsf = np.average(
                        rot_offsetpsfs, weights=sci_totinttime, axis=0
                    )
                    sx = av_offsetpsf.shape[1]
                    sy = av_offsetpsf.shape[0]

                    # Make sure that the model offset PSF has odd shape and
                    # perform the required subpixel shift before inserting
                    # it into the fm_frame.
                    if (sx % 2 != 1) or (sy % 2 != 1):
                        raise UserWarning("Model offset PSF must be of odd shape")
                    xshift = (fm_centx - int(fm_centx)) - (guess_dx - int(guess_dx))
                    yshift = (fm_centy - int(fm_centy)) + (guess_dy - int(guess_dy))
                    stamp = spline_shift(
                        av_offsetpsf,
                        (yshift, xshift),
                        order=3,
                        mode="constant",
                        cval=0.0,
                    )

                    # Also need to scale the model offset PSF by the
                    # guessed flux.
                    stamp *= guess_flux

                    # Insert the model offset PSF into the fm_frame.
                    fm_frame[:, :] = 0.0
                    fm_frame[
                        int(fm_centy) + int(guess_dy) - sy // 2 : int(fm_centy)
                        + int(guess_dy)
                        + sy // 2
                        + 1,
                        int(fm_centx) - int(guess_dx) - sx // 2 : int(fm_centx)
                        - int(guess_dx)
                        + sx // 2
                        + 1,
                    ] = stamp

                # assign forward model kwargs
                if "boxsize" not in kwargs.keys() or kwargs["boxsize"] is None:
                    boxsize = 35
                else:
                    boxsize = kwargs["boxsize"]
                if "dr" not in kwargs.keys() or kwargs["dr"] is None:
                    dr = 5
                else:
                    dr = kwargs["dr"]
                if "exclr" not in kwargs.keys() or kwargs["exclr"] is None:
                    exclr = 3 * resolution
                else:
                    exclr = kwargs["exclr"] * resolution
                if "xrange" not in kwargs.keys() or kwargs["xrange"] is None:
                    xrange = 3.0
                else:
                    xrange = kwargs["xrange"]
                if "yrange" not in kwargs.keys() or kwargs["yrange"] is None:
                    yrange = 3.0
                else:
                    yrange = kwargs["yrange"]
                if "frange" not in kwargs.keys() or kwargs["frange"] is None:
                    frange = 2.0  # i.e. bounds=[guess_flux/(10.**frange),guess_flux*(10**frange)]
                else:
                    frange = kwargs["frange"]
                if (
                    "corr_len_range" not in kwargs.keys()
                    or kwargs["corr_len_range"] is None
                ):
                    corr_len_range = 1.0
                else:
                    corr_len_range = kwargs["corr_len_range"]
                if (
                    "corr_len_guess" not in kwargs.keys()
                    or kwargs["corr_len_guess"] is None
                ):
                    corr_len_guess = 3.0
                else:
                    corr_len_guess = kwargs["corr_len_guess"]

                # Fit the FM PSF to the KLIP-subtracted data.
                if inject == False:
                    # Remove a constant background level from the
                    # KLIP-subtracted data before fitting the FM PSF?
                    if remove_background:
                        # Initialize pyKLIP FMAstrometry class.
                        fma = fitpsf.FMAstrometry(
                            guess_sep=guess_sep,
                            guess_pa=guess_pa,
                            fitboxsize=boxsize,
                        )
                        fma.generate_fm_stamp(
                            fm_image=fm_frame,
                            fm_center=[fm_centx, fm_centy],
                            padding=5,
                        )
                        fma.generate_data_stamp(
                            data=data_frame,
                            data_center=[data_centx, data_centy],
                            dr=dr,
                            exclusion_radius=exclr,
                        )
                        corr_len_label = r"$l$"
                        fma.set_kernel(fitkernel, [corr_len_guess], [corr_len_label])
                        fma.set_bounds(xrange, yrange, frange, [corr_len_range])

                        # Make sure that the noise map is invertible.
                        noise_map_max = np.nanmax(fma.noise_map)
                        fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
                        fma.noise_map[fma.noise_map == 0.0] = noise_map_max

                        # Set MCMC parameters from kwargs.
                        if (
                            "nwalkers" not in kwargs.keys()
                            or kwargs["nwalkers"] is None
                        ):
                            nwalkers = 50
                        else:
                            nwalkers = kwargs["nwalkers"]
                        if "nburn" not in kwargs.keys() or kwargs["nburn"] is None:
                            nburn = 100
                        else:
                            nburn = kwargs["nburn"]
                        if "nsteps" not in kwargs.keys() or kwargs["nsteps"] is None:
                            nsteps = 100
                        else:
                            nsteps = kwargs["nsteps"]
                        if (
                            "nthreads" not in kwargs.keys()
                            or kwargs["nthreads"] is None
                        ):
                            nthreads = 4
                        else:
                            nthreads = kwargs["nthreads"]

                        # Run the MCMC fit.
                        chain_output = os.path.join(
                            output_dir_comp,
                            mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                            + "-bka_chain_c%.0f" % (k + 1)
                            + ".pkl",
                        )
                        fma.fit_astrometry(
                            nwalkers=nwalkers,
                            nburn=nburn,
                            nsteps=nsteps,
                            numthreads=nthreads,
                            chain_output=chain_output,
                        )

                        # Estimate the background level from those pixels
                        # in the KLIP-subtracted data which have a small
                        # flux in the best fit FM PSF.
                        if "hsz" not in kwargs.keys() or kwargs["hsz"] is None:
                            hsz = 35
                        else:
                            hsz = kwargs["hsz"]
                        stamp = data_frame.copy()
                        xp = int(round(data_centx - guess_dx))
                        yp = int(round(data_centy + guess_dy))
                        stamp = stamp[yp - hsz : yp + hsz + 1, xp - hsz : xp + hsz + 1]
                        psf = fm_frame.copy()
                        xp = int(round(fm_centx - guess_dx))
                        yp = int(round(fm_centy + guess_dy))
                        psf = psf[yp - hsz : yp + hsz + 1, xp - hsz : xp + hsz + 1]
                        xshift = -(fma.raw_RA_offset.bestfit - guess_dx)
                        yshift = fma.raw_Dec_offset.bestfit - guess_dy
                        yxshift = np.array([yshift, xshift])
                        psf_shift = np.fft.ifftn(
                            fourier_shift(np.fft.fftn(psf), yxshift)
                        ).real
                        con = fma.fit_flux.bestfit * guess_flux
                        res = stamp - fma.fit_flux.bestfit * psf_shift
                        thresh = np.nanmax(psf_shift) / 150.0
                        bg = np.nanmedian(stamp[np.abs(psf_shift) < thresh])
                        data_frame -= bg

                    # MCMC.
                    if fitmethod == "mcmc":
                        # Initialize pyKLIP FMAstrometry class.
                        fma = fitpsf.FMAstrometry(
                            guess_sep=guess_sep,
                            guess_pa=guess_pa,
                            fitboxsize=boxsize,
                        )
                        fma.generate_fm_stamp(
                            fm_image=fm_frame,
                            fm_center=[fm_centx, fm_centy],
                            padding=5,
                        )
                        fma.generate_data_stamp(
                            data=data_frame,
                            data_center=[data_centx, data_centy],
                            dr=dr,
                            exclusion_radius=exclr,
                        )
                        corr_len_label = r"$l$"
                        fma.set_kernel(fitkernel, [corr_len_guess], [corr_len_label])
                        fma.set_bounds(xrange, yrange, frange, [corr_len_range])

                        # Make sure that the noise map is invertible.
                        noise_map_max = np.nanmax(fma.noise_map)
                        fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
                        fma.noise_map[fma.noise_map == 0.0] = noise_map_max

                        # Set MCMC parameters from kwargs.
                        if (
                            "nwalkers" not in kwargs.keys()
                            or kwargs["nwalkers"] is None
                        ):
                            nwalkers = 50
                        else:
                            nwalkers = kwargs["nwalkers"]
                        if "nburn" not in kwargs.keys() or kwargs["nburn"] is None:
                            nburn = 100
                        else:
                            nburn = kwargs["nburn"]
                        if "nsteps" not in kwargs.keys() or kwargs["nsteps"] is None:
                            nsteps = 100
                        else:
                            nsteps = kwargs["nsteps"]
                        if (
                            "nthreads" not in kwargs.keys()
                            or kwargs["nthreads"] is None
                        ):
                            nthreads = 4
                        else:
                            nthreads = kwargs["nthreads"]

                        # Run the MCMC fit.
                        chain_output = os.path.join(
                            output_dir_comp,
                            mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                            + "-bka_chain_c%.0f" % (k + 1)
                            + ".pkl",
                        )
                        fma.fit_astrometry(
                            nwalkers=nwalkers,
                            nburn=nburn,
                            nsteps=nsteps,
                            numthreads=nthreads,
                            chain_output=chain_output,
                        )

                        # Plot the MCMC fit results.
                        fig = fma.make_corner_plot()
                        if save_figures:
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-corner_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            fig.suptitle(
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                            )
                            fig.savefig(path)
                        plt.show()
                        plt.close(fig)
                        fig = fma.best_fit_and_residuals()
                        if save_figures:
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-model_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            fig.suptitle(
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                            )
                            fig.savefig(path)
                        plt.show()
                        plt.close(fig)

                        # Write the MCMC fit results into a table.
                        flux_jy = fma.fit_flux.bestfit * guess_flux
                        flux_jy *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                        flux_jy_err = fma.fit_flux.error * guess_flux
                        flux_jy_err *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                        flux_si = fma.fit_flux.bestfit * guess_flux
                        flux_si *= fzero_si[filt] / 10 ** (
                            mstar[filt] / 2.5
                        )  # erg/cm^2/s/A
                        flux_si *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                        flux_si_err = fma.fit_flux.error * guess_flux
                        flux_si_err *= fzero_si[filt] / 10 ** (
                            mstar[filt] / 2.5
                        )  # erg/cm^2/s/A
                        flux_si_err *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                        flux_si_alt = (
                            flux_jy
                            * 1e-26
                            * 299792458.0
                            / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                            * 1e-6
                        )  # W/m^2/um
                        flux_si_alt_err = (
                            flux_jy_err
                            * 1e-26
                            * 299792458.0
                            / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                            * 1e-6
                        )  # W/m^2/um
                        delmag = -2.5 * np.log10(
                            fma.fit_flux.bestfit * guess_flux
                        )  # mag
                        delmag_err = (
                            2.5
                            / np.log(10.0)
                            * fma.fit_flux.error
                            / fma.fit_flux.bestfit
                        )  # mag
                        if isinstance(mstar_err, dict):
                            mstar_err_temp = mstar_err[filt]
                        else:
                            mstar_err_temp = mstar_err
                        appmag = mstar[filt] + delmag  # vegamag
                        appmag_err = np.sqrt(mstar_err_temp**2 + delmag_err**2)
                        fitsfile = os.path.join(
                            output_dir_comp,
                            mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                            + "-fitpsf_c%.0f" % (k + 1)
                            + ".fits",
                        )

                        if split_fit:
                            # fit the sources with a 2D gaussian only to evaluate the sigma_x, sigma_y and theta
                            fig, result = best_convfit_and_residuals(
                                fma,
                                minmethod=minmethod,
                                initial_params=gauss_param_guesses,
                            )

                            if save_figures:
                                path = os.path.join(
                                    output_dir_comp,
                                    mode
                                    + "_NANNU"
                                    + str(annuli)
                                    + "_NSUBS"
                                    + str(subsections)
                                    + "_"
                                    + key
                                    + "-model_conv_c%.0f" % (k + 1)
                                    + ".pdf",
                                )
                                fig.suptitle(
                                    mode
                                    + "_NANNU"
                                    + str(annuli)
                                    + "_NSUBS"
                                    + str(subsections)
                                    + "_"
                                    + key
                                )
                                fig.savefig(path)
                            plt.show()
                            plt.close(fig)

                            tab.add_row(
                                (
                                    k + 1,
                                    fma.raw_RA_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_RA_offset.error * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.error * pxsc_arcsec,  # arcsec
                                    flux_jy,
                                    flux_jy_err,
                                    flux_si,
                                    flux_si_err,
                                    flux_si_alt,
                                    flux_si_alt_err,
                                    fma.raw_flux.bestfit * guess_flux,
                                    fma.raw_flux.error * guess_flux,
                                    delmag,  # mag
                                    delmag_err,  # mag
                                    appmag,  # mag
                                    appmag_err,  # mag
                                    mstar[filt],  # mag
                                    mstar_err_temp,  # mag
                                    np.nan,
                                    np.nan,
                                    scale_factor_avg,
                                    tp_comsubst,
                                    fitsfile,
                                    result.x[3],
                                    np.nan,
                                    result.x[0],
                                    np.nan,
                                    result.x[1],
                                    np.nan,
                                    result.x[2],
                                    np.nan,
                                )
                            )
                        else:
                            tab.add_row(
                                (
                                    k + 1,
                                    fma.raw_RA_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_RA_offset.error * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.error * pxsc_arcsec,  # arcsec
                                    flux_jy,
                                    flux_jy_err,
                                    flux_si,
                                    flux_si_err,
                                    flux_si_alt,
                                    flux_si_alt_err,
                                    fma.raw_flux.bestfit * guess_flux,
                                    fma.raw_flux.error * guess_flux,
                                    delmag,  # mag
                                    delmag_err,  # mag
                                    appmag,  # mag
                                    appmag_err,  # mag
                                    mstar[filt],  # mag
                                    mstar_err_temp,  # mag
                                    np.nan,
                                    np.nan,
                                    scale_factor_avg,
                                    tp_comsubst,
                                    fitsfile,
                                )
                            )

                        # Write the FM PSF to a file for future plotting.
                        ut.write_fitpsf_images(fma, fitsfile, tab[-1])

                    # Nested sampling.
                    elif fitmethod == "nested":
                        output_dir_ns = os.path.join(output_dir_comp, "temp-multinest/")

                        # Initialize PlanetEvidence module.
                        try:
                            fit = fitpsf.PlanetEvidence(
                                guess_sep, guess_pa, boxsize, output_dir_ns
                            )
                        except ModuleNotFoundError:
                            raise ModuleNotFoundError(
                                'Pymultinest is not installed, try\n"conda install -c conda-forge pymultinest"'
                            )
                        log.info("  --> Initialized PlanetEvidence module")

                        # Generate FM and data stamps.
                        fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)
                        fit.generate_data_stamp(
                            data_frame,
                            [data_centx, data_centy],
                            dr=dr,
                            exclusion_radius=exclr,
                        )
                        log.info("  --> Generated FM and data stamps")

                        # Set fit kernel.
                        corr_len_label = "l"
                        fit.set_kernel(fitkernel, [corr_len_guess], [corr_len_label])
                        log.info("  --> Set fit kernel to " + fitkernel)

                        # Set fit bounds.
                        fit.set_bounds(xrange, yrange, frange, [corr_len_range])
                        log.info("  --> Set fit bounds")

                        # Run the pymultinest fit.
                        fit.multifit()
                        log.info("  --> Finished pymultinest fit")

                        # Get model evidence and posteriors.
                        evidence = fit.fit_stats()
                        fm_evidence = evidence[0][
                            "nested sampling global log-evidence"
                        ]  # FM evidence
                        fm_posteriors = evidence[0]["marginals"]  # FM posteriors
                        null_evidence = evidence[1][
                            "nested sampling global log-evidence"
                        ]  # null evidence
                        null_posteriors = evidence[1]["marginals"]  # null posteriors
                        evidence_ratio = fm_evidence - null_evidence

                        # Plot the pymultinest fit results.
                        H1, H0 = fit.fit_plots()
                        if save_figures:
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-corner_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            H1.savefig(path)
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-noise_corner_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            H0.savefig(path)
                        plt.show()
                        plt.close(H1)
                        plt.close(H0)

                        fig, _ = fit.fm_residuals()
                        if save_figures:
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-model_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            plt.savefig(path)
                        plt.show()
                        plt.close(fig)

                        # Write the pymultinest fit results into a table.
                        flux_jy = fit.fit_flux.bestfit * guess_flux
                        flux_jy *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                        flux_jy_err = fit.fit_flux.error * guess_flux
                        flux_jy_err *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                        flux_si = fit.fit_flux.bestfit * guess_flux
                        flux_si *= fzero_si[filt] / 10 ** (
                            mstar[filt] / 2.5
                        )  # erg/cm^2/s/A
                        flux_si *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                        flux_si_err = fit.fit_flux.error * guess_flux
                        flux_si_err *= fzero_si[filt] / 10 ** (
                            mstar[filt] / 2.5
                        )  # erg/cm^2/s/A
                        flux_si_err *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                        flux_si_alt = (
                            flux_jy
                            * 1e-26
                            * 299792458.0
                            / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                            * 1e-6
                        )  # W/m^2/um
                        flux_si_alt_err = (
                            flux_jy_err
                            * 1e-26
                            * 299792458.0
                            / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                            * 1e-6
                        )  # W/m^2/um
                        delmag = -2.5 * np.log10(
                            fit.fit_flux.bestfit * guess_flux
                        )  # mag
                        delmag_err = (
                            2.5
                            / np.log(10.0)
                            * fit.fit_flux.error
                            / fit.fit_flux.bestfit
                        )  # mag
                        if isinstance(mstar_err, dict):
                            mstar_err_temp = mstar_err[filt]
                        else:
                            mstar_err_temp = mstar_err
                        appmag = mstar[filt] + delmag  # vegamag
                        appmag_err = np.sqrt(mstar_err_temp**2 + delmag_err**2)
                        fitsfile = os.path.join(
                            output_dir_comp,
                            mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                            + "-fitpsf_c%.0f" % (k + 1)
                            + ".fits",
                        )
                        tab.add_row(
                            (
                                k + 1,
                                -(fit.fit_x.bestfit - data_centx)
                                * pxsc_arcsec,  # arcsec
                                fit.fit_x.error * pxsc_arcsec,  # arcsec
                                (fit.fit_y.bestfit - data_centy)
                                * pxsc_arcsec,  # arcsec
                                fit.fit_y.error * pxsc_arcsec,  # arcsec
                                flux_jy,
                                flux_jy_err,
                                flux_si,
                                flux_si_err,
                                flux_si_alt,
                                flux_si_alt_err,
                                fit.fit_flux.bestfit * guess_flux,
                                fit.fit_flux.error * guess_flux,
                                delmag,  # mag
                                delmag_err,  # mag
                                appmag,  # mag
                                appmag_err,  # mag
                                mstar[filt],  # mag
                                mstar_err_temp,  # mag
                                np.nan,
                                evidence_ratio,
                                scale_factor_avg,
                                tp_comsubst,
                                fitsfile,
                            )
                        )

                        # Write the FM PSF to a file for future plotting.
                        ut.write_fitpsf_images(fit, fitsfile, tab[-1])

                    # Otherwise.
                    else:
                        if split_fit:
                            # Mocking the MCMC fit results using the initial guesses and perform only the Gaussian fit.
                            log.info(
                                "  --> Skipping  mcmc and pymultinest fit, just fitting for extended source."
                            )
                            # Initialize pyKLIP FMAstrometry class.
                            fma = fitpsf.FMAstrometry(
                                guess_sep=guess_sep,
                                guess_pa=guess_pa,
                                fitboxsize=boxsize,
                            )
                            fma.generate_fm_stamp(
                                fm_image=fm_frame,
                                fm_center=[fm_centx, fm_centy],
                                padding=5,
                            )
                            fma.generate_data_stamp(
                                data=data_frame,
                                data_center=[data_centx, data_centy],
                                dr=dr,
                                exclusion_radius=exclr,
                            )

                            fma.fit_flux = fitpsf.ParamRange(1, [0, 0])
                            fma.fit_x = fitpsf.ParamRange(
                                fma.data_stamp_x_center, [0, 0]
                            )
                            fma.fit_y = fitpsf.ParamRange(
                                fma.data_stamp_y_center, [0, 0]
                            )
                            fma.raw_RA_offset = fitpsf.ParamRange(
                                -(fma.fit_x.bestfit - fma.data_center[0]),
                                fma.fit_x.error_2sided[::-1],
                            )
                            fma.raw_Dec_offset = fitpsf.ParamRange(
                                fma.fit_y.bestfit - fma.data_center[1],
                                fma.fit_y.error_2sided[::-1],
                            )
                            fma.raw_flux = fma.fit_flux

                            flux_jy = fma.fit_flux.bestfit * guess_flux
                            flux_jy *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                            flux_jy_err = fma.fit_flux.error * guess_flux
                            flux_jy_err *= fzero[filt] / 10 ** (mstar[filt] / 2.5)  # Jy
                            flux_si = fma.fit_flux.bestfit * guess_flux
                            flux_si *= fzero_si[filt] / 10 ** (
                                mstar[filt] / 2.5
                            )  # erg/cm^2/s/A
                            flux_si *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                            flux_si_err = fma.fit_flux.error * guess_flux
                            flux_si_err *= fzero_si[filt] / 10 ** (
                                mstar[filt] / 2.5
                            )  # erg/cm^2/s/A
                            flux_si_err *= 1e-7 * 1e4 * 1e4  # W/m^2/um
                            flux_si_alt = (
                                flux_jy
                                * 1e-26
                                * 299792458.0
                                / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                                * 1e-6
                            )  # W/m^2/um
                            flux_si_alt_err = (
                                flux_jy_err
                                * 1e-26
                                * 299792458.0
                                / (1e-6 * database.database.red[key]["CWAVEL"][j]) ** 2
                                * 1e-6
                            )  # W/m^2/um
                            delmag = -2.5 * np.log10(
                                fma.fit_flux.bestfit * guess_flux
                            )  # mag
                            delmag_err = (
                                2.5
                                / np.log(10.0)
                                * fma.fit_flux.error
                                / fma.fit_flux.bestfit
                            )  # mag
                            if isinstance(mstar_err, dict):
                                mstar_err_temp = mstar_err[filt]
                            else:
                                mstar_err_temp = mstar_err
                            appmag = mstar[filt] + delmag  # vegamag
                            appmag_err = np.sqrt(mstar_err_temp**2 + delmag_err**2)
                            fitsfile = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-fitpsf_c%.0f" % (k + 1)
                                + ".fits",
                            )

                            # fit the sources with a 2D gaussian only to evaluate the sigma_x, sigma_y and theta
                            fig, result = best_convfit_and_residuals(
                                fma,
                                minmethod=minmethod,
                                initial_params=gauss_param_guesses,
                            )

                            if save_figures:
                                path = os.path.join(
                                    output_dir_comp,
                                    mode
                                    + "_NANNU"
                                    + str(annuli)
                                    + "_NSUBS"
                                    + str(subsections)
                                    + "_"
                                    + key
                                    + "-model_conv_c%.0f" % (k + 1)
                                    + ".pdf",
                                )
                                fig.suptitle(
                                    mode
                                    + "_NANNU"
                                    + str(annuli)
                                    + "_NSUBS"
                                    + str(subsections)
                                    + "_"
                                    + key
                                )
                                fig.savefig(path)
                            plt.show()
                            plt.close(fig)

                            tab.add_row(
                                (
                                    k + 1,
                                    fma.raw_RA_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_RA_offset.error * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.bestfit * pxsc_arcsec,  # arcsec
                                    fma.raw_Dec_offset.error * pxsc_arcsec,  # arcsec
                                    flux_jy,
                                    flux_jy_err,
                                    flux_si,
                                    flux_si_err,
                                    flux_si_alt,
                                    flux_si_alt_err,
                                    fma.raw_flux.bestfit * guess_flux,
                                    fma.raw_flux.error * guess_flux,
                                    delmag,  # mag
                                    delmag_err,  # mag
                                    appmag,  # mag
                                    appmag_err,  # mag
                                    mstar[filt],  # mag
                                    mstar_err_temp,  # mag
                                    np.nan,
                                    np.nan,
                                    scale_factor_avg,
                                    tp_comsubst,
                                    fitsfile,
                                    result.x[3],
                                    np.nan,
                                    result.x[0],
                                    np.nan,
                                    result.x[1],
                                    np.nan,
                                    result.x[2],
                                    np.nan,
                                )
                            )
                        else:
                            raise NotImplementedError()

                    # Plot estimated background level.
                    if remove_background:
                        f, ax = plt.subplots(1, 4, figsize=(4 * 6.4, 4.8))
                        p0 = ax[0].imshow(res, origin="lower")
                        c0 = plt.colorbar(p0, ax=ax[0])
                        c0.set_label(
                            "Flux (arbitrary units)", rotation=270, labelpad=20
                        )
                        text = ax[0].text(
                            0.99,
                            0.99,
                            "Contrast = %.3e" % con,
                            ha="right",
                            va="top",
                            transform=ax[0].transAxes,
                        )
                        text.set_path_effects(
                            [PathEffects.withStroke(linewidth=3, foreground="white")]
                        )
                        ax[0].set_title("Residuals before bg. subtraction")
                        p1 = ax[1].imshow(np.abs(psf_shift) < thresh, origin="lower")
                        c1 = plt.colorbar(p1, ax=ax[1])
                        ax[1].set_title("Pixels used for bg. estimation")
                        ax[2].hist(stamp[np.abs(psf_shift) < thresh], bins=20)
                        ax[2].axvline(
                            bg, ls="--", color="black", label="bg. = %.2f" % bg
                        )
                        ax[2].set_xlabel("Pixel value")
                        ax[2].set_ylabel("Occurrence")
                        ax[2].legend(loc="upper right")
                        ax[2].set_title("Distribution of bg. pixels")
                        con = fma.fit_flux.bestfit * guess_flux
                        res = stamp - fma.fit_flux.bestfit * psf_shift
                        imgs = ax[0].get_images()
                        if len(imgs) > 0:
                            vmin, vmax = imgs[0].get_clim()
                        p3 = ax[3].imshow(res, origin="lower", vmin=vmin, vmax=vmax)
                        c3 = plt.colorbar(p3, ax=ax[3])
                        c3.set_label(
                            "Flux (arbitrary units)", rotation=270, labelpad=20
                        )
                        text = ax[3].text(
                            0.99,
                            0.99,
                            "Contrast = %.3e" % con,
                            ha="right",
                            va="top",
                            transform=ax[3].transAxes,
                        )
                        text.set_path_effects(
                            [PathEffects.withStroke(linewidth=3, foreground="white")]
                        )
                        ax[3].set_title("Residuals after bg. subtraction")
                        plt.show()
                        if save_figures:
                            path = os.path.join(
                                output_dir_comp,
                                mode
                                + "_NANNU"
                                + str(annuli)
                                + "_NSUBS"
                                + str(subsections)
                                + "_"
                                + key
                                + "-bgest_c%.0f" % (k + 1)
                                + ".pdf",
                            )
                            plt.savefig(path)
                            log.info(f" Plot saved in {path}")
                            plt.close()

                # Subtract companion before fitting the next one.
                if subtract or inject:
                    # Subtract companion from pyKLIP dataset. Use offset
                    # PSFs w/o high-pass filtering because this will be
                    # applied by the klip_dataset routine below.
                    if inject:
                        ra = companions[k][0]  # arcsec
                        dec = companions[k][1]  # arcsec
                        con = companions[k][2]
                        inputflux = con * np.array(
                            all_offsetpsfs_nohpf
                        )  # positive to inject companion
                        fileprefix = (
                            "INJECTED-"
                            + mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                        )
                    else:
                        ra = tab[-1]["RA"]  # arcsec
                        dec = tab[-1]["DEC"]  # arcsec
                        con = tab[-1]["CON"]
                        inputflux = -con * np.array(
                            all_offsetpsfs_nohpf
                        )  # negative to remove companion
                        fileprefix = (
                            "KILLED-"
                            + mode
                            + "_NANNU"
                            + str(annuli)
                            + "_NSUBS"
                            + str(subsections)
                            + "_"
                            + key
                        )
                    sep = np.sqrt(ra**2 + dec**2) / pxsc_arcsec  # pix
                    pa = np.rad2deg(np.arctan2(ra, dec))  # deg
                    thetas = [pa + 90.0 - all_pa for all_pa in all_pas]
                    fakes.inject_planet(
                        frames=dataset_orig.input,
                        centers=dataset_orig.centers,
                        inputflux=inputflux,
                        astr_hdrs=dataset_orig.wcs,
                        radius=sep,
                        pa=pa,
                        thetas=np.array(thetas),
                        field_dependent_correction=None,
                    )

                    if save_preklip:
                        # Copy pre-KLIP files.
                        for filepath in filepaths:
                            src = filepath
                            dst = os.path.join(output_dir_pk, os.path.split(src)[1])
                            shutil.copy(src, dst)
                        for psflib_filepath in psflib_filepaths:
                            src = psflib_filepath
                            dst = os.path.join(output_dir_pk, os.path.split(src)[1])
                            shutil.copy(src, dst)

                        # Update content of pre-KLIP files.
                        filenames = dataset_orig.filenames.copy()
                        for l, filename in enumerate(filenames):
                            filenames[l] = filename[: filename.find("_INT")]
                        for filepath in filepaths:
                            ww_file = filenames == os.path.split(filepath)[1]
                            file = os.path.join(
                                output_dir_pk, os.path.split(filepath)[1]
                            )
                            hdul = fits.open(file)
                            hdul["SCI"].data = dataset_orig.input[ww_file]
                            hdul.writeto(file, output_verify="fix", overwrite=True)
                            hdul.close()

                        # Update and write observations database.
                        temp = database.database.obs.copy()
                        for l in range(len(database.database.obs[key])):
                            file = os.path.split(
                                database.database.obs[key]["FITSFILE"][l]
                            )[1]
                            database.database.obs[key]["FITSFILE"][l] = os.path.join(
                                output_dir_pk, file
                            )
                        file = os.path.split(database.database.red[key]["FITSFILE"][j])[
                            1
                        ]
                        file = file[file.find("JWST") : file.find("-KLmodes-all")]
                        file = os.path.join(output_dir_fm, file + ".dat")
                        database.database.obs[key].write(
                            file, format="ascii", overwrite=True
                        )
                        database.database.obs = temp

                    # Reduce companion-subtracted data.
                    mode = database.database.red[key]["MODE"][j]
                    annuli = database.database.red[key]["ANNULI"][j]
                    subsections = database.database.red[key]["SUBSECTS"][j]
                    parallelized.klip_dataset(
                        dataset=dataset_orig,
                        mode=mode,
                        outputdir=output_dir_fm,
                        fileprefix=fileprefix,
                        annuli=annuli,
                        subsections=subsections,
                        movement=1.0,
                        numbasis=klmodes,
                        maxnumbasis=maxnumbasis,
                        calibrate_flux=False,
                        aligned_center=dataset_orig._centers[0],
                        psf_library=dataset_orig.psflib,
                        highpass=highpass_temp,
                        verbose=False,
                    )
                    head = fits.getheader(database.database.red[key]["FITSFILE"][j], 0)
                    temp = os.path.join(output_dir_fm, fileprefix + "-KLmodes-all.fits")
                    hdul = fits.open(temp)
                    hdul[0].header = head
                    hdul.writeto(temp, output_verify="fix", overwrite=True)
                    hdul.close()

                # Restore original pyKLIP dataset.
                if subtract:
                    dataset = dataset_orig

            # Update source database.
            database.database.update_src(key, j, tab)

            # Save the results table.
            output_ecsv_path = os.path.join(
                output_dir_comp,
                mode
                + "_NANNU"
                + str(annuli)
                + "_NSUBS"
                + str(subsections)
                + "_"
                + key
                + "-results_c%.0f" % (k + 1)
                + ".ecsv",
            )
            tab.write(output_ecsv_path, format="ascii.ecsv", overwrite=True)
            print(f"Table saved to {output_ecsv_path}")
    pass
