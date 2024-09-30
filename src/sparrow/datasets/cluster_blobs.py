import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import skimage as ski
from dask_image import ndfilters
from numpy.random import default_rng
from scipy import ndimage as ndi
from scipy.stats import qmc
from spatialdata import SpatialData, concatenate
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Identity

from sparrow.table import add_regionprop_features
from sparrow.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def multisample_blobs(n_samples=4, prefix="sample_{i}", **kwargs):
    """Multisample blobs."""
    sdatas = [
        cluster_blobs(
            image_name=f"{prefix}_image".format(i=i),
            labels_name=f"{prefix}_labels".format(i=i),
            points_name=f"{prefix}_points".format(i=i),
            table_name=f"{prefix}_table".format(i=i),
            coordinate_system=f"{prefix}".format(i=i),
            seed=i,
            **kwargs,
        )
        for i in range(n_samples)
    ]
    sdata = concatenate(sdatas)
    sdata.tables["table"] = ad.concat(sdata.tables, merge="same")
    # sdata.tables["table"].var_names = list(sdata.images).c.values
    return sdata


def cluster_blobs(
    shape=None,
    n_cell_types=10,
    n_cells=20,
    noise_level_nuclei=None,
    noise_level_channels=None,
    region_key=_REGION_KEY,
    instance_key=_INSTANCE_KEY,
    image_name="blobs_image",
    labels_name="blobs_labels",
    points_name="blobs_points",
    table_name="table",
    coordinate_system="global",
    metadata_cycles=True,
    seed: int | None = None,
):
    """Differs from `spatialdata.datasets.make_blobs` in that it generates cells with multiple image channels and known ground truth cell types."""
    if shape is None:
        shape = (512, 512)
    if seed is None:
        seed = 42
    nuclei_centers = _generate_points(shape, n_points=n_cells, seed=seed)
    log.debug(f"{nuclei_centers.shape=}")
    log.debug(f"{nuclei_centers[0]=}")
    markers = _generate_markers(nuclei_centers, shape)
    log.debug(f"{markers.shape=}")
    nuclei_channel = _generate_blobs(nuclei_centers, shape)
    # assign each cell a random cell type
    assigned_cell_types = np.random.default_rng(seed).integers(0, n_cell_types, size=nuclei_centers.shape[0])
    log.debug(f"{assigned_cell_types.shape=}")
    lineage_channels = []
    for i in range(n_cell_types):
        selected_nuclei_centers = nuclei_centers[assigned_cell_types == i]
        if selected_nuclei_centers.shape[0] == 0:
            channel = da.zeros(shape)
        else:
            channel = _generate_blobs(selected_nuclei_centers, shape)
            channel = _add_noise(channel, noise_scale=noise_level_channels, seed=seed)
        lineage_channels.append(channel)
    log.debug(f"{lineage_channels[0].shape}")
    channel_names = ["nucleus"] + [f"lineage_{i}" for i in range(n_cell_types)]
    noisy_nuclei_channel = _add_noise(nuclei_channel, noise_scale=noise_level_nuclei, seed=seed)
    img = Image2DModel.parse(
        data=np.concatenate([noisy_nuclei_channel[np.newaxis], lineage_channels], axis=0),
        c_coords=channel_names,
        transformations={coordinate_system: Identity()},
    )
    img_segmented = _generate_segmentation(nuclei_channel, markers, watershed_line=True)
    labels = Labels2DModel.parse(img_segmented, transformations={coordinate_system: Identity()})
    points = PointsModel.parse(nuclei_centers, transformations={coordinate_system: Identity()})
    # generate table
    adata = aggregate(values=img, by=labels, target_coordinate_system=coordinate_system).tables["table"]
    # make X dense as markers are limited
    adata.X = adata.X.toarray()
    adata.obs[region_key] = pd.Categorical([labels_name] * len(adata))
    adata.obs[instance_key] = adata.obs_names.astype(int)
    adata.obs["phenotype"] = assigned_cell_types.astype(str)
    adata.obs.index.name = _CELL_INDEX
    adata.var_names = channel_names
    del adata.uns[TableModel.ATTRS_KEY]
    if metadata_cycles:
        # set the cycle to be per 2, so 1, 1, 2, 2, 3, 3, ...
        n_channels = n_cell_types + 1
        cycles = []
        for i in range(0, n_channels // 2):
            cycles.append(i)
            cycles.append(i)
        # if n_channels is odd, add one more cycle
        if n_channels % 2:
            cycles.append(i + 1)
        adata.var["cycle"] = cycles
    table = TableModel.parse(
        adata,
        region=labels_name,
        region_key=region_key,
        instance_key=instance_key,
    )

    sdata = SpatialData(
        images={
            image_name: img,
        },
        labels={
            labels_name: labels,
            # "blobs_markers": Labels2DModel.parse(data=markers),
        },
        points={points_name: points},
        tables={table_name: table},
    )
    add_regionprop_features(sdata, labels_layer=labels_name, table_layer=table_name)
    return sdata


def _generate_segmentation(image, markers, **kwargs):  # -> Any | ndarray[Any, dtype[Any]]:
    distance = ndi.distance_transform_edt(image)
    im_true = ski.segmentation.watershed(-distance, markers, compactness=0.8, mask=image > 10, watershed_line=True)
    im_true = ski.segmentation.expand_labels(im_true, distance=5)
    # im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]
    return im_true


def _generate_points(shape, n_points: int | None = None, seed: int | None = None):
    """Generate random points in a 2D image using dask.array."""
    rng = da.random.default_rng(seed)
    # TODO: make high depending on shape
    arr = rng.integers(0, high=shape[0], size=(n_points, len(shape)))
    log.debug(f"{arr.shape=}")
    return arr.compute()


def _generate_points_qmc(shape, n_points: int | None = None, seed: int | None = None):
    """Generate pseudo-random points in a 2D image using quasi-Monte Carlo sampling."""
    rng = default_rng(42) if seed is None else default_rng(seed)
    # from skimage
    engine = qmc.Sobol(d=len(shape), seed=rng)
    if n_points is None:
        n_points = np.mean(shape) / 20
    sample = engine.random(n_points)
    l_bounds = [0.0 for _ in shape]
    u_bounds = [float(d) for d in shape]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    return sample_scaled.astype(int)


def _generate_markers(points: np.ndarray, shape: tuple[int]):
    """Generate markers from points for initialising watershed segmentation."""
    m = points.shape[0]
    log.debug(f"{m=}")
    markers = np.zeros(shape)
    values = np.arange(m, dtype=int) + 1
    # generate image with values at coordinates of points
    for i, (x, y) in enumerate(points):
        markers[y, x] = values[i]
    # markers = ndi.grey_dilation(markers, size=(10, 10), output=markers)
    return markers.astype(int)


def _generate_blobs(points, shape, max_value=255, sigma: int | None = None) -> ArrayLike:
    """Generate image blobs from points."""
    if sigma is None:
        sigma = 10
    mask = da.zeros(shape)
    for x, y in points:
        mask[y, x] = 1
    mask = ndfilters.gaussian_filter(mask, sigma=sigma)
    mask *= max_value / mask.max()
    return mask


def _add_noise(image, noise_scale: float | None = None, seed: int | None = None):
    if noise_scale is None:
        noise_scale = 0.3 * image.max()
    rng = da.random.default_rng(seed)
    noise = da.abs(rng.normal(0, noise_scale, image.shape))
    output = image + noise
    # rescale output to original range
    output *= image.max() / output.max()
    return output
