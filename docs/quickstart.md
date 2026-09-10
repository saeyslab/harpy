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

# Create the AnnData table
sdata = hp.tb.aggregate_image(
    sdata,
    image_name="HumanLiverH35",
    labels_name="segmentation_mask",
    output_table_name="table_intensities",
    mode="mean",
    obs_stats="var",
)
```

Next, explore the introductory tutorials for [transcriptomics](./tutorials/general/Harpy_xenium_transcriptomics_subset.ipynb) and [proteomics](./tutorials/general/Harpy_feature_calculation.ipynb).
