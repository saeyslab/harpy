# Quickstart

Get Harpy running quickly with a small, self-contained example.

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install "harpy-analysis[extra]"
```

## Example

```python
import harpy as hp
import scanpy as sc

# Download an example proteomics dataset, and write it to a local Zarr store.
# Harpy then writes the result of each step to that store.
sdata = hp.datasets.macsima_example()
sdata.write("sdata.zarr")
sdata = hp.io.read_zarr("sdata.zarr")

# Segment the DAPI stain with Cellpose, or any segmentation model of choice.
# Channel selection is lazy and does not create an intermediate image.
sdata = hp.im.segment(
    sdata,
    image_name="HumanLiverH35",
    image_channels="R0_DAPI",
    model = hp.im.cellpose_callable,
    # keywords passed to Cellpose
    diameter=50,
    flow_threshold=0.8,
    cellprob_threshold=-4,
    output_labels_name="segmentation_mask",
    )

channel = "R0_DAPI"
render_images_kwargs = {"cmap": "viridis",}
render_labels_kwargs = {"fill_alpha": 0.6, "outline_alpha": 0.4}
show_kwargs = {"title": channel, "colorbar": False}

# Visualize
hp.pl.plot_sdata(
    sdata,
    image_name="HumanLiverH35",
    channel=channel,
    labels_name="segmentation_mask",
    show_kwargs=show_kwargs,
    render_images_kwargs=render_images_kwargs,
    render_labels_kwargs=render_labels_kwargs,
 )

# Create the AnnData table, written to the store and attached with lazy matrices
sdata = hp.tb.aggregate_image(
    sdata,
    image_name="HumanLiverH35",
    labels_name="segmentation_mask",
    output_table_name="table_intensities",
    mode="mean",
    obs_stats="var",
)
```

## Analyse the table with scanpy

The table is lazy: its matrices stay in the Zarr store, and scanpy works on them
as Dask arrays. Run scanpy as usual, then write its results back to the store.

```python
adata = sdata.tables["table_intensities"]

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# covariance_eigh: scanpy's PCA for Dask arrays. Fitting computes a small
# channels × channels covariance in one pass over the data; the projection stays lazy.
sc.pp.pca(adata, n_comps=10, svd_solver="covariance_eigh")
# The projection is small: compute it once, rather than in every step that uses it.
adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

# Write what scanpy added or changed. The normalised matrix goes to a new layer, so
# that the stored X keeps the raw intensities. overwrite=True: leiden added a column
# to obs, which already exists in the store.
sdata = hp.tb.io.add_table_updates(
    sdata, table_name="table_intensities", x_to=("layers", "log1p"), overwrite=True
)
```

`add_table_updates` writes only what changed: here the layer `log1p`, `obs`,
`X_pca`, the neighbour graphs and the `uns` entries, but not `X`. To write
chosen components by name, use `hp.tb.io.add_table_components`. When cells or
genes are removed, for example with `sc.pp.filter_cells`, the result is a new
table: write it with `hp.tb.io.add_table`.

For interactive cell typing, compute per-instance feature matrices with
`hp.tb.add_feature_matrix`, and classify the cells in napari with
[spatiato](https://github.com/vibspatial/spatiato), installed with
`harpy-analysis[napari]`.

Next, explore the introductory tutorials for [transcriptomics](./tutorials/general/Harpy_xenium_transcriptomics_subset.ipynb) and [proteomics](./tutorials/general/Harpy_feature_calculation.ipynb).
