from __future__ import annotations

import logging
import os
from os import PathLike

# from __future__ import annotations
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spaceKLIP.plotting import load_plt_style

log = logging.getLogger(__name__)


def plot_masked_data(
    data,
    center,
    pixel_scale,
    klmodes,
    filter_name,
    psfsub_strategy,
    fitsfile,
    bunit,
    *,
    save_figure=True,
    plot_style=None,
):
    """Plot the final masked PSF-subtracted image."""
    import textwrap

    fitsfile = os.fspath(fitsfile)

    load_plt_style(plot_style)

    image_data = np.asarray(data[-1], dtype=float)
    finite_values = image_data[np.isfinite(image_data)]

    if finite_values.size == 0:
        log.warning(
            "Masked image contains no finite values; using fallback normalization."
        )
        vmax = 1.0
    else:
        vmax = np.nanmax(np.abs(finite_values))

    if not np.isfinite(vmax) or vmax <= 0:
        log.warning(
            "Masked image has an invalid plotting range vmax=%s; "
            "using fallback normalization.",
            vmax,
        )
        vmax = 1.0

    linthresh = max(vmax / 100.0, np.finfo(float).eps)

    print(linthresh)

    norm = matplotlib.colors.SymLogNorm(
        linthresh=linthresh,
        vmin=-vmax,
        vmax=vmax,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    try:
        xx = np.arange(image_data.shape[1]) - center[0]
        yy = np.arange(image_data.shape[0]) - center[1]

        extent = (
            -(xx[0] - 0.5) * pixel_scale,
            -(xx[-1] + 0.5) * pixel_scale,
            (yy[0] - 0.5) * pixel_scale,
            (yy[-1] + 0.5) * pixel_scale,
        )

        image = ax.imshow(
            image_data,
            origin="lower",
            cmap="inferno",
            norm=norm,
            extent=extent,
        )

        ax.set_xlabel(r"$\Delta$RA [arcsec]")
        ax.set_ylabel(r"$\Delta$Dec [arcsec]")
        ax.set_title(
            f"Masked data in {filter_name}, {psfsub_strategy} ({klmodes[-1]} KL)"
        )

        for radius in (5, 10):
            ax.add_patch(
                matplotlib.patches.Circle(
                    (0, 0),
                    radius,
                    ls="--",
                    facecolor="none",
                    edgecolor="cyan",
                    clip_on=True,
                )
            )
            ax.text(radius, 0, f" {radius}''", color="cyan")

        ax.text(
            0.01,
            0.99,
            textwrap.fill(os.path.basename(fitsfile), width=40),
            transform=ax.transAxes,
            color="black",
            verticalalignment="top",
            fontsize=9,
        )

        fig.colorbar(image, ax=ax, label=bunit)
        fig.tight_layout()

        if save_figure:
            output_file = os.path.splitext(fitsfile)[0] + "_masked.pdf"
            fig.savefig(output_file)
            log.info("Plot saved in %s", output_file)

        plt.show()

    finally:
        plt.close(fig)


# def _robust_image_statistics(
#     image_data: np.ndarray,
#     *,
#     percentile: float = 99.5,
# ) -> tuple[float, float]:
#     """
#     Estimate a robust residual scale and symmetric display limit.

#     Parameters
#     ----------
#     image_data
#         Two-dimensional image.
#     percentile
#         Percentile of the absolute finite image values used to set the
#         symmetric display limit.

#     Returns
#     -------
#     sigma
#         Robust estimate of the residual scatter based on the median
#         absolute deviation.
#     vmax
#         Symmetric display limit.
#     """
#     finite_values = np.asarray(image_data, dtype=float)
#     finite_values = finite_values[np.isfinite(finite_values)]

#     if finite_values.size == 0:
#         raise ValueError("The image contains no finite pixel values.")

#     median = np.median(finite_values)
#     mad = np.median(np.abs(finite_values - median))
#     sigma = 1.4826 * mad

#     vmax = np.percentile(np.abs(finite_values), percentile)

#     if not np.isfinite(vmax) or vmax <= 0:
#         vmax = np.max(np.abs(finite_values))

#     if not np.isfinite(vmax) or vmax <= 0:
#         raise ValueError(
#             "The image does not contain a usable dynamic range for plotting."
#         )

#     if not np.isfinite(sigma) or sigma <= 0:
#         sigma = vmax / 100.0

#     return float(sigma), float(vmax)


# def _build_residual_norm(
#     image_data: np.ndarray,
#     *,
#     scale: str = "symlog",
#     percentile: float = 99.5,
#     vmax: float | None = None,
#     linthresh: float | None = None,
# ) -> tuple[Normalize, dict[str, float]]:
#     """
#     Construct a symmetric normalization for signed residual data.

#     Parameters
#     ----------
#     image_data
#         Two-dimensional residual image.
#     scale
#         Display scaling. Supported values are ``"linear"`` and
#         ``"symlog"``.
#     percentile
#         Percentile of absolute finite values used when ``vmax`` is not
#         supplied.
#     vmax
#         Optional fixed symmetric display limit. Supplying this is useful
#         when comparing multiple reductions.
#     linthresh
#         Linear threshold for the symmetric logarithmic normalization.
#         When not supplied, the robust image scatter is used.

#     Returns
#     -------
#     norm
#         Matplotlib normalization instance.
#     statistics
#         Dictionary containing the robust scatter, display limit, and
#         linear threshold.
#     """
#     sigma, robust_vmax = _robust_image_statistics(
#         image_data,
#         percentile=percentile,
#     )

#     if vmax is None:
#         vmax = robust_vmax

#     vmax = float(vmax)

#     if not np.isfinite(vmax) or vmax <= 0:
#         raise ValueError("vmax must be finite and greater than zero.")

#     if linthresh is None:
#         linthresh = sigma

#     linthresh = float(np.clip(linthresh, np.finfo(float).eps, vmax))

#     scale = scale.lower()

#     if scale == "linear":
#         norm = Normalize(vmin=-vmax, vmax=vmax)

#     elif scale == "symlog":
#         norm = SymLogNorm(
#             linthresh=linthresh,
#             linscale=1.0,
#             vmin=-vmax,
#             vmax=vmax,
#             base=10,
#         )

#     else:
#         raise ValueError(
#             f"Unsupported scale {scale!r}. Expected either 'linear' or 'symlog'."
#         )

#     statistics = {
#         "sigma": sigma,
#         "vmax": vmax,
#         "linthresh": linthresh,
#     }

#     return norm, statistics


# def _add_scale_circles(
#     ax: matplotlib.axes.Axes,
#     radii: Sequence[float],
#     *,
#     color: str = "white",
# ) -> None:
#     """Add reference-separation circles to an image."""
#     text_effect = [
#         patheffects.Stroke(linewidth=2.5, foreground="black"),
#         patheffects.Normal(),
#     ]

#     for radius in radii:
#         radius = float(radius)

#         if radius <= 0:
#             continue

#         circle = Circle(
#             (0.0, 0.0),
#             radius,
#             facecolor="none",
#             edgecolor=color,
#             linewidth=0.9,
#             linestyle="--",
#             alpha=0.85,
#             zorder=4,
#         )
#         ax.add_patch(circle)

#         label = ax.text(
#             radius / np.sqrt(2),
#             radius / np.sqrt(2),
#             f"{radius:g}″",
#             color=color,
#             fontsize=8,
#             ha="left",
#             va="bottom",
#             zorder=5,
#         )
#         label.set_path_effects(text_effect)


# def _add_orientation_arrows(
#     ax: matplotlib.axes.Axes,
#     *,
#     color: str = "white",
# ) -> None:
#     """
#     Add north and east orientation arrows.

#     Notes
#     -----
#     This annotation assumes that the image is already north-up and
#     east-left.
#     """
#     text_effect = [
#         patheffects.Stroke(linewidth=2.5, foreground="black"),
#         patheffects.Normal(),
#     ]

#     x0 = 0.90
#     y0 = 0.10
#     length = 0.08

#     arrow_properties = {
#         "arrowstyle": "-|>",
#         "color": color,
#         "linewidth": 1.4,
#         "mutation_scale": 11,
#     }

#     ax.annotate(
#         "",
#         xy=(x0, y0 + length),
#         xytext=(x0, y0),
#         xycoords="axes fraction",
#         arrowprops=arrow_properties,
#         zorder=6,
#     )

#     north_label = ax.text(
#         x0,
#         y0 + length + 0.015,
#         "N",
#         transform=ax.transAxes,
#         color=color,
#         fontsize=9,
#         fontweight="bold",
#         ha="center",
#         va="bottom",
#         zorder=6,
#     )
#     north_label.set_path_effects(text_effect)

#     # East points left for a conventional north-up astronomical image.
#     ax.annotate(
#         "",
#         xy=(x0 - length, y0),
#         xytext=(x0, y0),
#         xycoords="axes fraction",
#         arrowprops=arrow_properties,
#         zorder=6,
#     )

#     east_label = ax.text(
#         x0 - length - 0.015,
#         y0,
#         "E",
#         transform=ax.transAxes,
#         color=color,
#         fontsize=9,
#         fontweight="bold",
#         ha="right",
#         va="center",
#         zorder=6,
#     )
#     east_label.set_path_effects(text_effect)


# def plot_masked_data(
#     data,
#     center,
#     pixel_scale,
#     klmodes,
#     filter_name,
#     psfsub_strategy,
#     fitsfile,
#     bunit,
#     *,
#     image_index: int = -1,
#     save_figure: bool = True,
#     show_figure: bool = True,
#     close_figure: bool = True,
#     plot_style=None,
#     figure_mode: str = "publication",
#     scale: str = "symlog",
#     percentile: float = 99.5,
#     vmax: float | None = None,
#     linthresh: float | None = None,
#     cmap: str = "RdBu_r",
#     field_of_view: float | None = None,
#     separation_radii: Sequence[float] | None = None,
#     show_orientation: bool = True,
#     show_filename: bool | None = None,
#     output_directory: str | os.PathLike | None = None,
#     output_formats: Sequence[str] = ("pdf",),
#     dpi: int = 300,
# ):
#     """
#     Plot a masked PSF-subtracted residual image.

#     Parameters
#     ----------
#     data
#         Two-dimensional image or image cube indexed by KL mode.
#     center
#         Image center in zero-indexed ``(x, y)`` pixel coordinates.
#     pixel_scale
#         Pixel scale in arcseconds per pixel.
#     klmodes
#         Sequence of KL modes corresponding to the image cube.
#     filter_name
#         Instrument filter name.
#     psfsub_strategy
#         Name of the PSF-subtraction strategy.
#     fitsfile
#         Input or output FITS filename used to construct the figure name.
#     bunit
#         Image units used for the colorbar label.
#     image_index
#         Image-plane index to display.
#     save_figure
#         Save the figure to disk.
#     show_figure
#         Display the figure interactively.
#     close_figure
#         Close the figure before returning.
#     plot_style
#         Optional Matplotlib style accepted by ``plt.style.context``.
#     figure_mode
#         Either ``"publication"`` or ``"diagnostic"``.
#     scale
#         Either ``"linear"`` or ``"symlog"``.
#     percentile
#         Robust absolute-value percentile used for the display range.
#     vmax
#         Optional fixed symmetric display limit.
#     linthresh
#         Optional linear threshold for ``SymLogNorm``.
#     cmap
#         Diverging Matplotlib colormap.
#     field_of_view
#         Optional total displayed field of view in arcseconds.
#     separation_radii
#         Optional reference-circle radii in arcseconds.
#     show_orientation
#         Add north and east arrows. This assumes north-up and east-left.
#     show_filename
#         Include the FITS filename in the figure. By default, filenames are
#         shown only in diagnostic mode.
#     output_directory
#         Directory in which figures are saved. Defaults to the FITS-file
#         directory.
#     output_formats
#         Output extensions, such as ``("pdf", "png")``.
#     dpi
#         Resolution used for raster outputs.

#     Returns
#     -------
#     fig
#         Matplotlib figure.
#     ax
#         Image axes.
#     artists
#         Dictionary containing the image artist, colorbar, normalization,
#         display statistics, and saved output paths.

#     Notes
#     -----
#     The RA and Dec labels assume the input image is already transformed into
#     a north-up, east-left orientation. If that is not guaranteed, the labels
#     should instead describe generic relative image offsets.
#     """
#     fitsfile = Path(os.fspath(fitsfile))

#     figure_mode = figure_mode.lower()

#     if figure_mode not in {"publication", "diagnostic"}:
#         raise ValueError("figure_mode must be either 'publication' or 'diagnostic'.")

#     image_cube = np.asarray(data, dtype=float)

#     if image_cube.ndim == 2:
#         image_data = image_cube
#         selected_klmode = None

#     elif image_cube.ndim == 3:
#         image_data = image_cube[image_index]

#         if klmodes is None or len(klmodes) == 0:
#             selected_klmode = None
#         else:
#             selected_klmode = klmodes[image_index]

#     else:
#         raise ValueError(
#             "data must be either a two-dimensional image or a "
#             "three-dimensional image cube."
#         )

#     if len(center) != 2:
#         raise ValueError("center must contain exactly two values: (x, y).")

#     center_x, center_y = map(float, center)
#     pixel_scale = float(pixel_scale)

#     if not np.isfinite(pixel_scale) or pixel_scale <= 0:
#         raise ValueError("pixel_scale must be finite and greater than zero.")

#     if not np.any(np.isfinite(image_data)):
#         raise ValueError("The selected image contains no finite values.")

#     norm, display_statistics = _build_residual_norm(
#         image_data,
#         scale=scale,
#         percentile=percentile,
#         vmax=vmax,
#         linthresh=linthresh,
#     )

#     masked_image = np.ma.masked_invalid(image_data)

#     image_cmap = matplotlib.colormaps[cmap].copy()
#     image_cmap.set_bad(color="0.75", alpha=1.0)

#     # Pixel-edge coordinates relative to the supplied stellar center.
#     x_edges = (
#         np.arange(image_data.shape[1] + 1, dtype=float) - center_x - 0.5
#     ) * pixel_scale

#     y_edges = (
#         np.arange(image_data.shape[0] + 1, dtype=float) - center_y - 0.5
#     ) * pixel_scale

#     extent = (
#         x_edges[0],
#         x_edges[-1],
#         y_edges[0],
#         y_edges[-1],
#     )

#     if plot_style is None:
#         style_context = matplotlib.rc_context()
#     else:
#         style_context = plt.style.context(plot_style)

#     saved_files: list[Path] = []

#     with style_context:
#         fig, ax = plt.subplots(
#             figsize=(6.3, 5.3),
#             constrained_layout=True,
#         )

#         image_artist = ax.imshow(
#             masked_image,
#             origin="lower",
#             extent=extent,
#             cmap=image_cmap,
#             norm=norm,
#             interpolation="nearest",
#             aspect="equal",
#             rasterized=True,
#         )

#         # Astronomical convention: east is displayed toward the left.
#         ax.invert_xaxis()

#         ax.set_xlabel(r"$\Delta\mathrm{RA}$ [arcsec]")
#         ax.set_ylabel(r"$\Delta\mathrm{Dec}$ [arcsec]")

#         if selected_klmode is None:
#             title = f"{filter_name} residual image"
#         else:
#             title = f"{filter_name} residual image — {selected_klmode} KL mode"

#         ax.set_title(title, pad=10)

#         if field_of_view is not None:
#             half_width = float(field_of_view) / 2.0

#             if not np.isfinite(half_width) or half_width <= 0:
#                 raise ValueError("field_of_view must be finite and greater than zero.")

#             ax.set_xlim(half_width, -half_width)
#             ax.set_ylim(-half_width, half_width)

#         if separation_radii is not None:
#             _add_scale_circles(
#                 ax,
#                 separation_radii,
#                 color="white",
#             )

#         if show_orientation:
#             _add_orientation_arrows(
#                 ax,
#                 color="white",
#             )

#         if show_filename is None:
#             show_filename = figure_mode == "diagnostic"

#         if figure_mode == "diagnostic":
#             strategy_text = f"PSF subtraction: {psfsub_strategy}"

#             statistics_text = (
#                 f"{strategy_text}\n"
#                 f"Scale: {scale}\n"
#                 f"Robust σ: {display_statistics['sigma']:.3g}\n"
#                 f"|limit|: {display_statistics['vmax']:.3g}"
#             )

#             diagnostic_label = ax.text(
#                 0.02,
#                 0.98,
#                 statistics_text,
#                 transform=ax.transAxes,
#                 ha="left",
#                 va="top",
#                 fontsize=8,
#                 color="white",
#                 linespacing=1.35,
#                 bbox={
#                     "facecolor": "black",
#                     "edgecolor": "none",
#                     "alpha": 0.60,
#                     "boxstyle": "round,pad=0.35",
#                 },
#                 zorder=7,
#             )
#             diagnostic_label.set_in_layout(False)

#         if show_filename:
#             filename_label = ax.text(
#                 0.02,
#                 0.02,
#                 fitsfile.name,
#                 transform=ax.transAxes,
#                 ha="left",
#                 va="bottom",
#                 fontsize=7,
#                 color="white",
#                 bbox={
#                     "facecolor": "black",
#                     "edgecolor": "none",
#                     "alpha": 0.55,
#                     "boxstyle": "round,pad=0.30",
#                 },
#                 zorder=7,
#             )
#             filename_label.set_in_layout(False)

#         colorbar = fig.colorbar(
#             image_artist,
#             ax=ax,
#             pad=0.025,
#             fraction=0.05,
#         )

#         if bunit:
#             colorbar.set_label(f"Residual intensity [{bunit}]")
#         else:
#             colorbar.set_label("Residual intensity")

#         colorbar.ax.tick_params(direction="out")

#         if save_figure:
#             if output_directory is None:
#                 output_directory = fitsfile.parent

#             output_directory = Path(output_directory)
#             output_directory.mkdir(parents=True, exist_ok=True)

#             # Handles normal FITS names and compressed FITS files.
#             stem = fitsfile.name
#             for suffix in (".fits.gz", ".fits", ".fit.gz", ".fit"):
#                 if stem.lower().endswith(suffix):
#                     stem = stem[: -len(suffix)]
#                     break
#             else:
#                 stem = fitsfile.stem

#             output_stem = output_directory / f"{stem}_masked"

#             metadata = {
#                 "Title": title,
#                 "Subject": "Masked PSF-subtracted residual image",
#                 "Keywords": (
#                     f"{filter_name}, {psfsub_strategy}, "
#                     "PSF subtraction, high-contrast imaging"
#                 ),
#             }

#             for output_format in output_formats:
#                 extension = str(output_format).lower().lstrip(".")
#                 output_file = output_stem.with_suffix(f".{extension}")

#                 save_kwargs = {
#                     "bbox_inches": "tight",
#                     "facecolor": "white",
#                     "metadata": metadata,
#                 }

#                 if extension in {"png", "jpg", "jpeg", "tif", "tiff"}:
#                     save_kwargs["dpi"] = dpi

#                 fig.savefig(
#                     output_file,
#                     **save_kwargs,
#                 )

#                 saved_files.append(output_file)
#                 log.info("Plot saved to %s", output_file)

#         if show_figure:
#             plt.show()

#     artists = {
#         "image": image_artist,
#         "colorbar": colorbar,
#         "norm": norm,
#         "statistics": display_statistics,
#         "saved_files": saved_files,
#     }

#     if close_figure:
#         plt.close(fig)

#     return fig, ax, artists


def plot_raw_contrast(
    separations: NDArray,
    contrasts: NDArray,
    masked_contrasts: NDArray | None,
    klmodes: Sequence[str],
    filter_name: str,
    psfsub_strategy: str,
    fitsfile: str | PathLike,
    *,
    plot_xlim: tuple[float | None, float | None] | None = (0, 10),
    plot_ylim: tuple[float | None, float | None] | None = (None, 1),
    save_figure: bool = True,
    plot_style: str | None = None,
) -> None:
    """
    Plot raw contrast curves for each KL mode.

    When ``masked_contrasts`` is provided, the uncorrected curves are plotted
    as faint dashed lines and the mask-corrected curves are plotted as solid
    lines.

    Parameters
    ----------
    separations
        Separation arrays in arcseconds, with one row per KL mode.
    contrasts
        Raw contrast arrays, with one row per KL mode.
    masked_contrasts
        Contrast arrays corrected for coronagraph-mask throughput. Use None
        when no mask is available.
    klmodes
        KL mode labels.
    filter_name
        Instrument filter name used in the plot title.
    psfsub_strategy
        Description of the PSF-subtraction strategy.
    fitsfile
        Output FITS path used to construct the plot filename.
    plot_xlim
        Horizontal plotting limits.
    plot_ylim
        Vertical plotting limits.
    save_figure
        Save the figure as a PDF when True.
    plot_style
        Optional SpaceKLIP plotting style.
    """
    fitsfile = os.fspath(fitsfile)

    load_plt_style(plot_style)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ncolors = len(colors)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()

    for index in range(contrasts.shape[0]):
        color = colors[index % ncolors]
        label = f"{klmodes[index]} KL"

        if masked_contrasts is None:
            ax.plot(
                separations[index],
                contrasts[index],
                color=color,
                label=label,
            )
        else:
            ax.plot(
                separations[index],
                contrasts[index],
                color=color,
                alpha=0.3,
                ls="--",
            )
            ax.plot(
                separations[index],
                masked_contrasts[index],
                color=color,
                label=label,
            )

    ax.set_yscale("log")

    if plot_ylim is not None:
        ax.set_ylim(plot_ylim)

    if plot_xlim is not None:
        ax.set_xlim(plot_xlim)

    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel(r"5-$\sigma$ contrast")

    ax.legend(
        loc="upper right",
        ncols=3,
        title=(
            None
            if masked_contrasts is None
            else "Dashed lines exclude coronagraph mask throughput"
        ),
        title_fontsize=10,
    )

    ax.set_title(f"Raw contrast in {filter_name}, {psfsub_strategy}")

    plt.tight_layout()

    if save_figure:
        output_file = fitsfile[:-5] + "_rawcon.pdf"
        plt.savefig(output_file)
        log.info(" Plot saved in %s", output_file)

    plt.show()
    plt.close(fig)
