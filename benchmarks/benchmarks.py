# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import uuid

import dask
import dask.array as da
import flowsom as fs
import pandas as pd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel

import harpy as hp
from harpy.image.segmentation.segmentation_models._baysor import _dummy
from harpy.utils._aggregate import RasterAggregator


class PixelClusteringSuite:
    """Benchmark FlowSOM pixel clustering."""

    timeout = 600
    image_layer = "image"

    params = ([1024],)
    param_names = [
        "chunks",
    ]

    def setup_cache(self):
        """Set up cache."""
        tile_repeat = (1, 10, 10)  # will results in image of size tile_repeat * ( 22, 512, 512 )

        sdata = hp.datasets.pixie_example(["fov0"])

        # create artificial large dataset
        arr = da.tile(sdata["raw_image_fov0"].data, tile_repeat)
        arr = arr.rechunk((1, 1024, 1024))

        sdata = SpatialData()

        se = Image2DModel.parse(arr, dims=("c", "y", "x"))

        _sdata = SpatialData()
        _sdata[self.image_layer] = se

        _sdata.write("sdata.zarr")

    def setup(self, chunks):
        """Set up."""
        self.sdata = read_zarr("sdata.zarr")

    def time_preprocess_pixel_clustering(self, chunks):
        """Peakmem clustering"""
        _pixel_clustering_preprocess(self.sdata, image_layer=self.image_layer, chunks=chunks)

    def peakmem_preprocess_pixel_clustering(self, chunks):
        """Peakmem clustering"""
        _pixel_clustering_preprocess(self.sdata, image_layer=self.image_layer, chunks=chunks)

    def time_pixel_clustering(self, chunks):
        """Peakmem clustering"""
        _pixel_clustering(self.sdata, image_layer=self.image_layer, chunks=chunks)

    def peakmem_pixel_clustering(self, chunks):
        """Peakmem clustering"""
        _pixel_clustering(self.sdata, image_layer=self.image_layer, chunks=chunks)


class RasterSuite:
    """Benchmark vectorization, rasterization, aggregation and (mock) segmentation."""

    timeout = 600
    labels_layer = "labels"
    image_layer = "image"

    params = ([4096],)
    param_names = [
        "chunks",
    ]

    def setup_cache(self):
        """Set up cache."""
        sdata = hp.datasets.merscope_segmentation_masks_example()

        _sdata = SpatialData()

        size = 50000
        se = Image2DModel.parse(sdata["clahe"].data[:, :size, :size].rechunk((1, 4096, 4096)), dims=("c", "y", "x"))
        assert se.data.chunksize == (1, 4096, 4096)

        _sdata[self.image_layer] = se

        se = Labels2DModel.parse(
            sdata["segmentation_mask_full"].data[:size, :size].rechunk((4096, 4096)), dims=("y", "x")
        )
        assert se.data.chunksize == (4096, 4096)

        _sdata[self.labels_layer] = se

        _sdata.write("sdata.zarr")

    def setup(self, chunks):
        """Set up."""
        self.sdata = read_zarr("sdata.zarr")

    def teardown(self, chunks):
        """Teardown"""
        del self.sdata

    # 1. vectorize, rasterize
    def time_vectorize_rasterize(self, chunks):
        """Time vectorize and rasterize."""
        _vectorize_rasterize(
            self.sdata,
            labels_layer=self.labels_layer,
            chunks=chunks,
        )

    def peakmem_vectorize_rasterize(self, chunks):
        """Peak mem vectorize and rasterize."""
        _vectorize_rasterize(
            self.sdata,
            labels_layer=self.labels_layer,
            chunks=chunks,
        )

    # 2. aggregate
    def time_aggregate_sum(self, chunks):
        """Aggregate mean."""
        _aggregate_sum(self.sdata, image_layer=self.image_layer, labels_layer=self.labels_layer, chunks=chunks)

    def peakmem_aggregate_sum(self, chunks):
        """Aggregate mean."""
        _aggregate_sum(self.sdata, image_layer=self.image_layer, labels_layer=self.labels_layer, chunks=chunks)

    # skip these
    # def time_aggregate_var(self, chunks):
    #    """Aggregate mean."""
    #    _aggregate_var(self.sdata, image_layer=self.image_layer, labels_layer=self.labels_layer, chunks=chunks)

    # def peakmem_aggregate_var(self, chunks):
    #    """Aggregate var."""
    #    _aggregate_var(self.sdata, image_layer=self.image_layer, labels_layer=self.labels_layer, chunks=chunks)

    def time_aggregate_area(self, chunks):
        """Aggregate area."""
        _aggregate_area(self.sdata, labels_layer=self.labels_layer, chunks=chunks)

    def peakmem_aggregate_area(self, chunks):
        """Aggregate area."""
        _aggregate_area(self.sdata, labels_layer=self.labels_layer, chunks=chunks)

    # 3. (mock) segmentation
    def peakmem_segment(self, chunks):
        """Segmentation."""
        # Note, that when using a client, dask spills to disk to reduce memory.
        # also, smaller chunks size can lead to higher mem consumption in this case (when not using a client),
        # due to more artefacts to solve/larger task graph.
        _mock_segment(self.sdata, labels_layer=self.labels_layer, chunks=chunks)

    def time_segment(self, chunks):
        """Segmentation."""
        _mock_segment(self.sdata, labels_layer=self.labels_layer, chunks=chunks)


def _vectorize_rasterize(sdata: SpatialData, labels_layer: str, chunks: int = 2000):
    """Rasterize and vectorize"""
    dask.config.set(scheduler="processes")

    shapes_layer = f"shapes_layer_{uuid.uuid4()}"
    sdata = hp.sh.vectorize(sdata, labels_layer=labels_layer, output_layer=shapes_layer, overwrite=True)

    dask.config.set(scheduler="threads")
    output_labels_layer = f"labels_layer_{uuid.uuid4()}"

    sdata = hp.im.rasterize(
        sdata, shapes_layer=shapes_layer, output_layer=output_labels_layer, chunks=chunks, overwrite=True
    )


def _aggregate_sum(sdata: SpatialData, image_layer: str, labels_layer: str, chunks: int):
    se_labels = sdata[labels_layer]
    se_image = sdata[image_layer]

    aggregator = RasterAggregator(
        mask_dask_array=se_labels.data[None, ...].rechunk(chunks),
        image_dask_array=se_image.data[:, None, ...].rechunk(chunks),
    )
    aggregator.aggregate_sum()


def _aggregate_var(sdata: SpatialData, image_layer: str, labels_layer: str, chunks: int):
    se_labels = sdata[labels_layer]
    se_image = sdata[image_layer]

    aggregator = RasterAggregator(
        mask_dask_array=se_labels.data[None, ...].rechunk(chunks),
        image_dask_array=se_image.data[:, None, ...].rechunk(chunks),
    )
    aggregator.aggregate_var()


def _aggregate_area(sdata: SpatialData, labels_layer: str, chunks: int):
    se_labels = sdata[labels_layer]

    aggregator = RasterAggregator(
        mask_dask_array=se_labels.data[None, ...].rechunk(chunks),
        image_dask_array=None,
    )
    aggregator.aggregate_area()


def _mock_segment(sdata: SpatialData, labels_layer: str, chunks: int):
    import dask.dataframe as dd
    from dask.dataframe import DataFrame

    # we use the mock segmentation callable _dummy to benchmark mem and time of resolving of chunk artefacts.
    data = {"x": [10], "y": [10], "gene": ["dummy_gene"]}

    df = pd.DataFrame(data)

    ddf = dd.from_pandas(df, npartitions=1)

    coordinates = {"x": "x", "y": "y"}

    sdata = hp.pt.add_points_layer(
        sdata,
        ddf=ddf,
        output_layer="dummy_transcripts",
        coordinates=coordinates,
        overwrite=True,
    )

    assert isinstance((sdata.points["dummy_transcripts"]), DataFrame)

    sdata = hp.im.segment_points(
        sdata,
        labels_layer=labels_layer,
        points_layer="dummy_transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        c_dim=2,
        output_labels_layer=[f"{labels_layer}_output_1", f"{labels_layer}_output_2"],
        output_shapes_layer=None,
        labels_layer_align=None,
        chunks=chunks,
        depth=300,
        overwrite=True,
    )


def _pixel_clustering_preprocess(sdata: SpatialData, image_layer: str, chunks: int):
    sdata = hp.im.pixel_clustering_preprocess(
        sdata,
        img_layer=[image_layer],
        output_layer=[f"{image_layer}_preprocessed"],
        chunks=chunks,
        persist_intermediate=False,  # set to False if you have multiple images, and if they are large.
        overwrite=True,
        sigma=2.0,
    )


def _pixel_clustering(sdata: SpatialData, image_layer: str, chunks: int):
    batch_model = fs.models.BatchFlowSOMEstimator

    sdata, _, _ = hp.im.flowsom(
        sdata,
        img_layer=[f"{image_layer}"],
        output_layer_clusters=[
            f"{image_layer}_clusters",
        ],
        output_layer_metaclusters=[f"{image_layer}_metaclusters"],
        n_clusters=20,
        chunks=chunks,
        client=None,
        model=batch_model,
        num_batches=10,
        fraction=0.05,
        xdim=10,
        ydim=10,
        z_score=True,
        z_cap=3,
        persist_intermediate=True,
        overwrite=True,
    )
