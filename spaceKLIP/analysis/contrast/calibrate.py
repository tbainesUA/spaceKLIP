from __future__ import division

import copy
import logging

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
from io import StringIO

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pyklip.fakes as fakes
from astropy.table import Table
from cycler import cycler
from pyklip import parallelized
from pyklip.instruments.JWST import JWSTData
from scipy.interpolate import interp1d
from stpsf.constants import JWST_CIRCUMSCRIBED_DIAMETER
from tqdm.auto import trange

from spaceKLIP import utils as ut
from spaceKLIP.plotting import load_plt_style
from spaceKLIP.psf import get_offsetpsf
from spaceKLIP.pyklippipeline import get_pyklip_filepaths
from spaceKLIP.starphot import get_stellar_magnitudes
from spaceKLIP.utils import pop_pxar_kw

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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


def run_calibrate_contrast(
    database,
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

    print(type(database))

    # Check input.
    companions = validate_companions(companions)

    # Set output directory.
    output_dir = os.path.join(database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get raw contrast directory
    rawcon_dir = os.path.join(database.output_dir, rawcon_subdir)
    if not os.path.exists(rawcon_dir):
        raise TypeError(
            'Raw contrast must be calculated first. "rawcon" subdirectory not found.'
        )

    # Loop through concatenations.
    for i, key in enumerate(database.red.keys()):
        log.info("--> Concatenation " + key)

        # Need to generate the offset PSF we'll be injecting. Best to do
        # this per concatenation to save time.
        if use_saved == True:
            # Don't need to bother generating the PSF, use dummy value
            offsetpsf = 1
        else:
            offsetpsf = get_offsetpsf(database.obs[key])

        # Loop through FITS files.
        nfitsfiles = len(database.red[key])
        for j in range(nfitsfiles):
            # Read FITS file and PSF mask.
            fitsfile = database.red[key]["FITSFILE"][j]

            data, head_pri, head_sci, is2d = ut.read_red(fitsfile)
            maskfile = database.red[key]["MASKFILE"][j]
            mask = ut.read_msk(maskfile)

            log.info("Analyzing file " + fitsfile)

            # Get the raw contrast information with and without mask correction
            file_str = fitsfile.split("/")[-1]
            if rawcon_filetype == "npy":
                seps_file = file_str.replace(".fits", "_seps.npy")  # Arcseconds
                rawcons_file = file_str.replace(".fits", "_cons.npy")
                maskcons_file = file_str.replace(".fits", "_cons_mask.npy")

                rawseps = np.load(os.path.join(rawcon_dir, seps_file))
                rawcons = np.load(os.path.join(rawcon_dir, rawcons_file))
                maskcons = np.load(os.path.join(rawcon_dir, maskcons_file))
            elif rawcon_filetype == "ecsv":
                raise NotImplementedError(
                    ".ecsv save format not currently supported for \
                        calibrated contrasts. Please use .npy raw contrasts as input."
                )
                # contrast_file = file_str.replace('.fits', '_contrast.ecsv')
                # contrast_path =  os.path.join(rawcon_dir, contrast_file)

                # rawcon_data = Table.read(contrast_path, format='ascii.ecsv')

            print("HERE")

            # Read Stage 2 files and make pyKLIP dataset
            filepaths, psflib_filepaths = get_pyklip_filepaths(database, key)
            pop_pxar_kw(np.append(filepaths, psflib_filepaths))
            pyklip_dataset = JWSTData(filepaths, psflib_filepaths)

            # Compute the resolution element. Account for possible blurring.
            pxsc_arcsec = database.red[key]["PIXSCALE"][j]  # arcsec
            pxsc_rad = pxsc_arcsec / 3600.0 / 180.0 * np.pi  # rad
            if database.red[key]["TELESCOP"][j] == "JWST":
                if database.red[key]["EXP_TYPE"][j] in [
                    "NRC_CORON",
                    "NRC_TACONFIRM",
                    "NRC_TACQ",
                ]:
                    diam = 5.2
                else:
                    diam = JWST_CIRCUMSCRIBED_DIAMETER
            else:
                raise UserWarning("Data originates from unknown telescope")
            resolution = 1e-6 * database.red[key]["CWAVEL"][j] / diam / pxsc_rad  # pix
            if not np.isnan(database.obs[key]["BLURFWHM"][j]):
                resolution *= database.obs[key]["BLURFWHM"][j]
            resolution_fwhm = 1.025 * resolution

            # Get stellar magnitudes and filter zero points, but use the same file as rawcon
            ccinfo = os.path.join(rawcon_dir, "contrast_curve_info.txt")
            with open(ccinfo) as cci:
                starfile, spectral_type_info = cci.readline().strip("\n").split(" /// ")
                spectral_type = spectral_type_info.split(": ")[1]
                starfile = os.path.join(rawcon_dir, starfile.replace("#", ""))
            mstar, fzero = get_stellar_magnitudes(
                starfile,
                spectral_type,
                database.red[key]["INSTRUME"][j],
                output_dir=output_dir,
                **kwargs,
            )  # vegamag, Jy
            filt = database.red[key]["FILTER"][j]
            fstar = (
                fzero[filt] / 10.0 ** (mstar[filt] / 2.5) / 1e6 * np.nanmax(offsetpsf)
            )  # MJy
            fstar *= ((180.0 / np.pi) * 3600.0) ** 2 / pxsc_arcsec**2  # MJy/sr
            # Get PSF subtraction strategy used, for use in plot labels below.
            psfsub_strategy = (
                f"{head_pri['MODE']} with {head_pri['ANNULI']} annuli."
                if head_pri["ANNULI"] > 1
                else head_pri["MODE"]
            )

            ### Now want to perform the injection and recovery of companions.
            # Define the seps and PAs to inject companions at
            if injection_seps == "default":
                inj_seps = [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.2,
                    1.4,
                    1.6,
                    1.8,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    5.0,
                ]
            else:
                inj_seps = injection_seps
            inj_seps_pix = inj_seps / pxsc_arcsec  # Convert separation to pixels

            if injection_pas == "default":
                if "4QPM" in database.red[key]["CORONMSK"][j]:
                    inj_pas = [57.5, 147.5, 237.5, 327.5]
                elif "WB" in database.red[key]["CORONMSK"][j]:
                    inj_pas = [45.0, 135.0, 225.0, 315.0]
                else:
                    inj_pas = [0, 60, 120, 180, 240, 300]
            else:
                inj_pas = injection_pas

            # Determine the fluxes we want to inject the companions at.
            # Base it on the contrast for the desired separations. Use the
            # contrast with the ~median KL modes.
            median_KL_index = int(len(rawseps) / 2)
            cons_cleaned = np.nan_to_num(rawcons[median_KL_index], nan=1)
            contrast_interp = interp1d(
                rawseps[median_KL_index],
                cons_cleaned,
                kind="linear",
                bounds_error=False,
                fill_value=(1, cons_cleaned[-1]),
            )
            inj_cons = contrast_interp(inj_seps)
            inj_fluxes = inj_cons * fstar  # MJy/sr
            inj_fluxes *= injection_flux_sigma / 5  # Scale to an N sigma peak flux

            # Going to redefine companion locations in terms of pixels
            companions_pix = []
            if companions is not None:
                for k in range(len(companions)):
                    ra, dec, rad = companions[k]  # arcsec, arcsec, lambda/D
                    ra_pix = ra / pxsc_arcsec
                    dec_pix = dec / pxsc_arcsec
                    rad_pix = rad * resolution  # pix
                    companions_pix.append([ra_pix, dec_pix, rad_pix])
            else:
                companions_pix = None

            # Redefine the multi_injection_spacing in terms of pixels
            if multi_injection_spacing is not None:
                injection_spacing_pix = multi_injection_spacing * resolution
            else:
                injection_spacing_pix = multi_injection_spacing

            # Need to get exactly the same KLIP arguments that were used for this subtraction.
            klip_args = {}
            klip_args["mode"] = database.red[key]["MODE"][j]
            klip_args["annuli"] = database.red[key]["ANNULI"][j]
            klip_args["subsections"] = database.red[key]["SUBSECTS"][j]
            klip_args["numbasis"] = [
                int(nb) for nb in database.red[key]["KLMODES"][j].split(",")
            ]
            klip_args["algo"] = (
                "klip"  # Currently not logged, may need changing in future.
            )
            _, _, maxnumbasis = get_pyklip_filepaths(
                database, key, return_maxbasis=True
            )  # ensure maxnumbasis is same as for rawcon / klipsub reduction
            klip_args["maxnumbasis"] = maxnumbasis
            inj_subdir = (
                klip_args["mode"]
                + "_NANNU"
                + str(klip_args["annuli"])
                + "_NSUBS"
                + str(klip_args["subsections"])
                + "_"
                + key
                + "/"
            )
            klip_args["movement"] = 1  # Currently not logged, fix later.
            klip_args["calibrate_flux"] = False
            klip_args["highpass"] = False
            klip_args["verbose"] = False
            inj_output_dir = os.path.join(output_dir, inj_subdir)
            if not os.path.exists(inj_output_dir):
                os.makedirs(inj_output_dir)
            klip_args["outputdir"] = inj_output_dir

            save_string = output_dir + "/" + file_str[:-5]
            if use_saved:
                log.info("Retrieving saved companion injection and recovery results.")
                all_inj_seps = np.load(save_string + "_injrec_seps.npy")
                all_inj_pas = np.load(save_string + "_injrec_pas.npy")
                all_inj_fluxes = np.load(save_string + "_injrec_inj_fluxes.npy")
                all_retr_fluxes = np.load(save_string + "_injrec_retr_fluxes.npy")
            else:
                # Run the injection and recovery process
                log.info(
                    "Injecting and recovering synthetic companions. This may take a while..."
                )
                inj_rec = inject_and_recover(
                    pyklip_dataset,
                    injection_psf=offsetpsf,
                    injection_seps=inj_seps_pix,
                    injection_pas=inj_pas,
                    injection_spacing=injection_spacing_pix,
                    injection_fluxes=inj_fluxes,
                    klip_args=klip_args,
                    retrieve_fwhm=resolution_fwhm,
                    true_companions=companions_pix,
                )

                # Unpack everything from the injection and recovery
                all_inj_seps, all_inj_pas, all_inj_fluxes, all_retr_fluxes = inj_rec

                # Save these arrays
                np.save(save_string + "_injrec_seps.npy", all_inj_seps)
                np.save(save_string + "_injrec_pas.npy", all_inj_pas)
                np.save(save_string + "_injrec_inj_fluxes.npy", all_inj_fluxes)
                np.save(save_string + "_injrec_retr_fluxes.npy", all_retr_fluxes)

            # Need to add a point at a separation of zero pixels, assume
            # basically no flux retrieved at zero separation.
            all_inj_seps = np.append([0], all_inj_seps)
            all_inj_pas = np.append([0], all_inj_pas)
            all_inj_fluxes = np.append([1], all_inj_fluxes)
            zero_sep_retr_flux = 1e-10 * np.ones_like(all_retr_fluxes[0])
            all_retr_fluxes = np.vstack([zero_sep_retr_flux, all_retr_fluxes])

            # Separation returned in pixels but we want arcseconds
            all_inj_seps *= pxsc_arcsec

            # Need to loop over each KL mode used to compute a correction
            # for each.
            rawcons_corr = []
            maskcons_corr = []
            all_corrections = []
            for k in range(len(rawseps)):
                # Get the raw separation and contrast for this KL mode
                this_KL_rawseps = rawseps[k]
                this_KL_rawcons = rawcons[k]
                this_KL_maskcons = maskcons[k]

                # Get fluxes for this KL mode subtracted image
                this_KL_retr_fluxes = all_retr_fluxes[:, k]

                # Make a table to make things easier
                results = Table(
                    [
                        all_inj_seps,
                        all_inj_pas,
                        all_inj_fluxes,
                        this_KL_retr_fluxes,
                    ],
                    names=("inj_seps", "inj_pas", "inj_fluxes", "retr_fluxes"),
                )

                # Determine throughput of klip process on the injected flux
                results["klip_thrputs"] = np.divide(
                    results["retr_fluxes"], results["inj_fluxes"]
                )

                # Calculate the median across all position angles
                med_results = results.group_by("inj_seps").groups.aggregate(
                    np.nanmedian
                )

                # Need to interpolate or model to determine throughput at actual
                # separations of the contrast curve.
                if thrput_fit_method == "median":
                    med_interp = interp1d(
                        med_results["inj_seps"],
                        med_results["klip_thrputs"],
                        fill_value=(1e-10, med_results["klip_thrputs"][-1]),
                        bounds_error=False,
                        kind="slinear",
                    )
                    contrast_correction = med_interp(this_KL_rawseps)
                elif thrput_fit_method == "log_grow":
                    raise NotImplementedError()
                else:
                    raise ValueError(
                        "Invalid thrput_fit_method: "
                        + "{}, options are 'median' or 'log_grow'".format(
                            thrput_fit_method
                        )
                    )

                # Apply contrast correction
                rawcons_corr.append(rawcons[k] / contrast_correction)
                maskcons_corr.append(maskcons[k] / contrast_correction)
                all_corrections.append(contrast_correction)

            all_corrections = np.squeeze(all_corrections)  # Tidy array
            # Need to make sure its not 1D for number of different KL modes == 1 case
            if all_corrections.ndim == 1:
                all_corrections = all_corrections[np.newaxis, :]

            # Save the corrected contrasts, as well as the separations for convenience.
            np.save(save_string + "_cal_seps.npy", rawseps)
            np.save(save_string + "_cal_cons.npy", rawcons_corr)
            np.save(save_string + "_cal_maskcons.npy", maskcons_corr)

            # Define some local utilty functions for plot setup.
            # This makes the plotting code below less repetitive and more consistent

            def standardize_plots_setup(plot_style=None):
                # Intialize the matplotlib style.
                load_plt_style(plot_style)

                fig = plt.figure(figsize=(6.4, 4.8))
                ax = plt.gca()
                color = plt.cm.tab10(np.linspace(0, 1, 10))
                cc = cycler(linestyle=["-", ":", "--"]) * cycler(color=color)
                ax.set_prop_cycle(cc)
                return fig, ax

            def standardize_plots_annotate_save(
                ax,
                title="",
                ylabel="Throughput",
                xlim=plot_xlim,
                filename=None,
                plot_style=None,
            ):
                # Intialize the matplotlib style.
                load_plt_style(plot_style)

                ax.set_xlabel('Separation (")')
                ax.set_title(title, fontsize=11)
                if ylabel == "Throughput":
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Throughput")
                else:  # or else it's contrast on a log scale
                    ax.set_yscale("log")
                    ax.set_ylabel(r"5-$\sigma$ contrast")
                    ax.set_ylim(None, 1)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                ax.grid(axis="both", alpha=0.15)
                if filename is not None:
                    plt.savefig(filename, bbox_inches="tight", dpi=300)

            # Plot measured KLIP throughputs, for all KL modes
            fig, ax = standardize_plots_setup(plot_style=plot_style)

            for ci, corr in enumerate(all_corrections):
                KLmodes = klip_args["numbasis"][ci]
                ax.plot(rawseps[ci], corr, label="KL = {}".format(KLmodes))
            ax.legend(ncol=3, fontsize=10)
            standardize_plots_annotate_save(
                ax,
                title=f"Injected companions in {filt}, {psfsub_strategy}, all KL modes",
                ylabel="Throughput",
                filename=save_string + "_allKL_throughput.pdf",
                plot_style=plot_style,
            )
            plt.close(fig)

            # Plot individual measurements for median KL mode
            fig, ax = standardize_plots_setup(plot_style=plot_style)

            ax.plot(
                rawseps[median_KL_index],
                all_corrections[median_KL_index],
                label="Applied Correction",
                color="#0B5345",
                zorder=100,
            )
            ax.scatter(
                all_inj_seps,
                all_retr_fluxes[:, median_KL_index] / all_inj_fluxes,
                s=75,
                color="mediumaquamarine",
                alpha=0.5,
                label="Individual Injections",
            )
            ax.legend(fontsize=10)
            standardize_plots_annotate_save(
                ax,
                title=f"Injected companions in {filt}, {psfsub_strategy}, for KL={klip_args['numbasis'][median_KL_index]}",
                ylabel="Throughput",
                filename=save_string + "_medKL_throughput.pdf",
                plot_style=plot_style,
            )
            plt.close(fig)

            # Plot calibrated contrast curves
            fig, ax = standardize_plots_setup(plot_style=plot_style)
            for si, seps in enumerate(rawseps):
                KLmodes = klip_args["numbasis"][si]
                ax.plot(
                    seps, maskcons_corr[si], label=f"KL = {KLmodes}", color=f"C{si}"
                )
                ax.plot(seps, rawcons_corr[si], alpha=0.3, ls="--", color=f"C{si}")
            ax.legend(
                loc="upper right",
                ncols=3,
                fontsize=10,
                title="Dashed lines exclude coronagraph mask throughput",
                title_fontsize=10,
            )
            standardize_plots_annotate_save(
                ax,
                title=f"Calibrated contrast in {filt}, {psfsub_strategy}",
                ylabel="Contrast",
                filename=save_string + "_calcon.pdf",
                plot_style=plot_style,
            )
            plt.close(fig)

            # Plot calibrated contrast curves compared to raw
            fig, ax = standardize_plots_setup(plot_style=plot_style)
            for si, seps in enumerate(rawseps):
                KLmodes = klip_args["numbasis"][si]
                ax.plot(
                    seps, maskcons_corr[si], label=f"KL = {KLmodes}", color=f"C{si}"
                )
                ax.plot(seps, maskcons[si], alpha=0.3, ls=":", color=f"C{si}")
            ax.legend(
                loc="upper right",
                ncols=3,
                fontsize=10,
                title="Solid lines = calibrated, dotted lines = raw",
                title_fontsize=10,
            )
            standardize_plots_annotate_save(
                ax,
                title=f"Calibrated contrast vs Raw contrast in {filt}, {psfsub_strategy}",
                ylabel="Contrast",
                filename=save_string + "_calcon_vs_rawcon.pdf",
                plot_style=plot_style,
            )
            plt.close(fig)


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
