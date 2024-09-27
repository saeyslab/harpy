import matplotlib.pyplot as plt
import pandas as pd


def calculate_segmentation_coverage(sdata):
    # Calculate the segmentation coverage
    labels = sdata.labels
    data = []
    for name, label in labels.items():
        coverage = (label != 0).sum() / label.size
        data.append([name, coverage.compute().item()])
    return pd.DataFrame(data, columns=["name", "coverage"])


def calculate_segments_per_area(sdata):
    df = sdata.table.obs.groupby("sample_id").agg(
        {"sample_id": "count", "image_width_px": "mean", "image_height_px": "mean"}
    )
    # TODO: use pixel size from metadata
    df["cells_per_mm2"] = df["sample_id"] / (df["image_width_px"] * df["image_height_px"] / 1e6)
    df.sort_values("cells_per_mm2", inplace=True)
    return df


def segmentation_coverage(sdata, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    df = calculate_segmentation_coverage(sdata)
    df.sort_values("coverage").plot.barh(x="name", y="coverage", xlabel="Percentile of covered area", ax=ax, **kwargs)
    return ax


def segmentation_size_boxplot(sdata, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    sdata.table.obs[["area", "sample_id"]].plot.box(by="sample_id", rot=45, ax=ax)
    return ax


def segments_per_area(sdata, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    df = calculate_segments_per_area(sdata)
    return df.plot.bar(y="cells_per_mm2", rot=45, ax=ax, **kwargs)
