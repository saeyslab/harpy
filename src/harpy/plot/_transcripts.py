from __future__ import annotations

from pathlib import Path

from spatialdata import SpatialData

from harpy.plot import plot_shapes


def transcript_density(
    sdata: SpatialData,
    img_layer: tuple[str, str] = ["raw_image", "transcript_density"],
    channel: int = 0,
    crd: tuple[int, int, int, int] | None = None,
    figsize: tuple[int, int] | None = None,
    output: str | Path | None = None,
) -> None:
    """
    Visualize the transcript density layer.

    .. deprecated:: 0.3.0
       `harpy.pl.transcript_density` is deprecated and will be removed in 0.4.0.
       Prefer `harpy.pl.plot_sdata`.

    This function wraps around the :func:`harpy.pl.plot_shapes` function to showcase transcript density.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    img_layer
        A tuple where the first element indicates the base image layer and
        the second element indicates the transcript density.
    channel
        The channel of the image to be visualized.
        If the channel not in one of the images, the first available channel of the image will be plotted
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize
        The figure size for the visualization. If None, a default size will be used.
    output
        Path to save the output image. If None, the image will not be saved and will be displayed instead.

    Returns
    -------
    None

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> transcript_density(sdata, img_layer=["raw_img", "density"], crd=(2000,4000,2000,4000))

    See Also
    --------
    harpy.im.transcript_density
    harpy.pl.plot_shapes
    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        crd=crd,
        figsize=figsize,
        output=output,
    )
