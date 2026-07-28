from __future__ import division

import logging

# =============================================================================
# IMPORTS
# =============================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from cycler import cycler
from pyklip.instruments.JWST import JWSTData
from scipy.interpolate import interp1d
from stpsf.constants import JWST_CIRCUMSCRIBED_DIAMETER

from spaceKLIP import utils as ut
from spaceKLIP.plotting import load_plt_style
from spaceKLIP.psf import get_offsetpsf
from spaceKLIP.pyklippipeline import get_pyklip_filepaths
from spaceKLIP.starphot import get_stellar_magnitudes
from spaceKLIP.utils import pop_pxar_kw

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def calibrate_contrast(
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

    # Check input.
    companions = validate_companions(companions)

    # Set output directory.
    output_dir = os.path.join(database.database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get raw contrast directory
    rawcon_dir = os.path.join(database.database.output_dir, rawcon_subdir)
    if not os.path.exists(rawcon_dir):
        raise TypeError(
            'Raw contrast must be calculated first. "rawcon" subdirectory not found.'
        )

    # Loop through concatenations.
    for i, key in enumerate(database.database.red.keys()):
        log.info("--> Concatenation " + key)

        # Need to generate the offset PSF we'll be injecting. Best to do
        # this per concatenation to save time.
        if use_saved == True:
            # Don't need to bother generating the PSF, use dummy value
            offsetpsf = 1
        else:
            offsetpsf = get_offsetpsf(database.database.obs[key])

        # Loop through FITS files.
        nfitsfiles = len(database.database.red[key])
        for j in range(nfitsfiles):
            # Read FITS file and PSF mask.
            fitsfile = database.database.red[key]["FITSFILE"][j]
            data, head_pri, head_sci, is2d = ut.read_red(fitsfile)
            maskfile = database.database.red[key]["MASKFILE"][j]
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

            # Read Stage 2 files and make pyKLIP dataset
            filepaths, psflib_filepaths = get_pyklip_filepaths(database.database, key)
            pop_pxar_kw(np.append(filepaths, psflib_filepaths))
            pyklip_dataset = JWSTData(filepaths, psflib_filepaths)

            # Compute the resolution element. Account for possible blurring.
            pxsc_arcsec = database.database.red[key]["PIXSCALE"][j]  # arcsec
            pxsc_rad = pxsc_arcsec / 3600.0 / 180.0 * np.pi  # rad
            if database.database.red[key]["TELESCOP"][j] == "JWST":
                if database.database.red[key]["EXP_TYPE"][j] in [
                    "NRC_CORON",
                    "NRC_TACONFIRM",
                    "NRC_TACQ",
                ]:
                    diam = 5.2
                else:
                    diam = JWST_CIRCUMSCRIBED_DIAMETER
            else:
                raise UserWarning("Data originates from unknown telescope")
            resolution = (
                1e-6 * database.database.red[key]["CWAVEL"][j] / diam / pxsc_rad
            )  # pix
            if not np.isnan(database.database.obs[key]["BLURFWHM"][j]):
                resolution *= database.database.obs[key]["BLURFWHM"][j]
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
                database.database.red[key]["INSTRUME"][j],
                output_dir=output_dir,
                **kwargs,
            )  # vegamag, Jy
            filt = database.database.red[key]["FILTER"][j]
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
                if "4QPM" in database.database.red[key]["CORONMSK"][j]:
                    inj_pas = [57.5, 147.5, 237.5, 327.5]
                elif "WB" in database.database.red[key]["CORONMSK"][j]:
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
            klip_args["mode"] = database.database.red[key]["MODE"][j]
            klip_args["annuli"] = database.database.red[key]["ANNULI"][j]
            klip_args["subsections"] = database.database.red[key]["SUBSECTS"][j]
            klip_args["numbasis"] = [
                int(nb) for nb in database.database.red[key]["KLMODES"][j].split(",")
            ]
            klip_args["algo"] = (
                "klip"  # Currently not logged, may need changing in future.
            )
            _, _, maxnumbasis = get_pyklip_filepaths(
                database.database, key, return_maxbasis=True
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
