"""Calculate various image quality metrics"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage as ski
import textalloc as ta
from loguru import logger

from sparrow.image import normalize


def calculate_snr(img, nbins=65536):
    """Calculate the signal to noise ratio of an image.

    The threshold is calculated using the Otsu method.
    The signal is the mean intensity of the pixels above the threshold and the noise is the mean of the pixels below the threshold.
    """
    thres = ski.filters.threshold_otsu(img, nbins=nbins)
    mask = img > thres
    signal = img[mask].mean()
    noise = img[~mask].mean()
    snr = signal / noise
    return snr, signal


def calculate_snr_ratio(
    sdata, image="raw_image", block_size=10000, channel_names=None, cycles=None, signal_threshold=None
):
    logger.debug("Calculating SNR ratio")
    data = []
    for image in sdata.images:
        if channel_names is None:
            channel_names = sdata.table.var_names
        if cycles is None:
            cycles = [None] * len(channel_names)

        for cycle, channel_name in zip(cycles, channel_names):
            float_block = sdata[image].sel(c=channel_name).data.rechunk(block_size)
            img = float_block.compute()
            snr, signal = calculate_snr(img)
            if signal_threshold and signal < signal_threshold:
                continue
            data += [(image, cycle, channel_name, snr, signal)]
            del img
    df_img = pd.DataFrame(data, columns=["image", "cycle", "channel", "snr", "signal"])
    return df_img


def snr_ratio(sdata, ax=None, loglog=True, **kwargs):
    """Plot the signal to noise ratio. On the x-axis is the signal intensity and on the y-axis is the SNR-ratio"""
    logger.debug("Plotting SNR ratio")
    if ax is None:
        fig, ax = plt.subplots()
    df_img = calculate_snr_ratio(sdata, **kwargs)
    if loglog:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    # group by "channel" and take the mean of "image" and "cycle"
    df_img = df_img.groupby(["channel"]).mean(numeric_only=True).reset_index()
    # sort by channel
    df_img = df_img.sort_values("channel")
    # do a scatter plot
    for _i, row in df_img.iterrows():
        ax.scatter(row["signal"], row["snr"], color="black")
        # use textalloc to add channel names
    x = df_img["signal"]
    y = df_img["snr"]
    ta.allocate(ax, x=x, y=y, text_list=sorted(sdata.table.var_names), x_scatter=x, y_scatter=y)

    ax.set_xlabel("Signal intensity")
    ax.set_ylabel("Signal-to-noise ratio")
    ax.legend()
    return ax


def arc_transform(df):
    divider = 5 * np.quantile(df, 0.2, axis=0)
    divider[divider == 0] = df.max(axis=0)[divider == 0]
    scaled = np.arcsinh(df / divider)
    return scaled


def calculate_mean_norm(sdata, overwrite=False, c_mask=None, key="normalized_", func_transform=np.arcsinh, **kwargs):
    """Calculate the mean of the normalized images and return a DataFrame with the mean for each image channel"""
    data = []
    metadata = []
    for image_name in [x for x in sdata.images if key not in x]:
        norm_image_name = key + image_name
        if overwrite or norm_image_name not in sdata.images:
            normalize(sdata, image_name, output_layer=norm_image_name, overwrite=True, **kwargs)
        # caluculate the mean of the normalized image for each channel
        c_means = sdata[norm_image_name].mean(["x", "y"]).compute().data
        data.append(c_means)
        metadata.append(image_name)
    df = pd.DataFrame(data, columns=sdata.table.var_names)
    if func_transform is not None:
        df = func_transform(df)
    # remove c_mask columns if it is not None
    if c_mask is not None:
        df: pd.DataFrame = df.drop(columns=c_mask)
    df.index = pd.Index(metadata, name="image_name")
    # sort by index
    df = df.sort_index()
    return df


def get_hexes(col, palette="Set1"):
    pallete = sns.color_palette(palette, n_colors=len(col.unique()))
    lut = dict(zip(col.unique(), pallete.as_hex()))
    return col.astype(str).map(lut)


def clustermap(*args, **kwargs):
    return sns.clustermap(*args, **kwargs)


def signal_clustermap(sdata, signal_threshold=None, fill_value=0, **kwargs):
    df = calculate_snr_ratio(sdata, signal_threshold=signal_threshold)
    df = df.groupby(["image", "channel"]).mean(numeric_only=True).reset_index().drop(columns="snr")
    df = df.set_index(["image", "channel"]).unstack()
    df.columns = df.columns.droplevel(0)
    df.fillna(fill_value, inplace=True)
    return clustermap(df, **kwargs)


def snr_clustermap(sdata, signal_threshold=None, fill_value=0, **kwargs):
    df = calculate_snr_ratio(sdata, signal_threshold=signal_threshold)
    df = df.groupby(["image", "channel"]).mean(numeric_only=True).reset_index().drop(columns="signal")
    df = df.set_index(["image", "channel"]).unstack()
    df.columns = df.columns.droplevel(0)
    df.fillna(fill_value, inplace=True)
    return clustermap(df, **kwargs)


def make_cols_colors(df, palettes=None):
    df = df.copy()
    if palettes is None:
        palettes = [f"Set{i+1}" for i in range(len(df.columns))]
    for c, p in zip(df.columns, palettes):
        df[c] = get_hexes(df[c], palette=p)
    return df
