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

# Download an example proteomics dataset
sdata = hp.datasets.macsima_example()

# Get the DAPI stain, and add it to a new slot.
sdata = hp.im.add_image_layer(
    sdata,
    arr=sdata["HumanLiverH35"].sel(c="R0_DAPI").data[None, ...],
    output_layer="image",
    overwrite=True
)

# Segment the DAPI stain with Cellpose, or any segmentation model of choice.
sdata = hp.im.segment(
    sdata,
    img_layer="image",
    model = hp.im.cellpose_callable,
    # keywords passed to Cellpose
    diameter=50,
    flow_threshold=0.8,
    cellprob_threshold=-4,
    output_labels_layer="segmentation_mask",
    )

channel = "R0_DAPI"
render_images_kwargs = {"cmap": "viridis",}
render_labels_kwargs = {"fill_alpha": 0.6, "outline_alpha": 0.4}
show_kwargs = {"title": channel, "colorbar": False}

# Visualize
hp.pl.plot_sdata(
    sdata,
    img_layer="HumanLiverH35",
    channel=channel,
    labels_layer="segmentation_mask",
    show_kwargs=show_kwargs,
    render_images_kwargs=render_images_kwargs,
    render_labels_kwargs=render_labels_kwargs,
 )

# Create the AnnData table
sdata = hp.tb.allocate_intensity(
    sdata,
    img_layer="HumanLiverH35",
    labels_layer="segmentation_mask",
    output_layer="table_intensities",
    mode="mean",
    obs_stats="var",
)
```

Next steps: explore the [tutorials](tutorials/index.md) for end-to-end pipelines.
