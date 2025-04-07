import matplotlib.pyplot as plt
import pandas as pd

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def calculate_segmentation_coverage(sdata):
    # Calculate the segmentation coverage
    labels = sdata.labels
    data = []
    for name, label in labels.items():
        coverage = (label != 0).sum() / label.size
        data.append([name, coverage.compute().item()])
    return pd.DataFrame(data, columns=["name", "coverage"])


def calculate_segments_per_area(sdata, sample_key="sample_id"):
    table = sdata.table.obs.copy()
    # for labels in 'fov_labels', look up in sdata.labels and calculate the area
    if "image_width_px" in table.columns and "image_height_px" in table.columns:
        df = table.groupby(sample_key).agg({sample_key: "count", "image_width_px": "mean", "image_height_px": "mean"})
        # TODO: use pixel size from metadata
        df["cells_per_mm2"] = df[sample_key] / (df["image_width_px"] * df["image_height_px"] / 1e6)
    if "fov_labels" in table.columns:
        df = table.groupby(sample_key).agg({sample_key: "count"})
        area_map = dict.fromkeys(table["fov_labels"].unique().tolist(), 1000)
        log.debug(area_map)
        df["cells_per_mm2"] = df.index.map(area_map)
    df.sort_values("cells_per_mm2", inplace=True)
    return df


def segmentation_coverage(sdata, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    df = calculate_segmentation_coverage(sdata)
    df.sort_values("coverage").plot.barh(x="name", y="coverage", xlabel="Percentile of covered area", ax=ax, **kwargs)
    return ax


def segmentation_size_boxplot(sdata, area_key="area", sample_key="sample_id", ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    sdata.table.obs[[area_key, sample_key]].plot.box(by=sample_key, rot=45, ax=ax)
    return ax


def segments_per_area(sdata, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    df = calculate_segments_per_area(sdata, **kwargs)
    return df.plot.bar(y="cells_per_mm2", rot=45, ax=ax)
