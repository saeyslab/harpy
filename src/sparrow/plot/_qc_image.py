"""
Calculate various image quality metrics
"""

import matplotlib.pyplot as plt
import pandas as pd
import skimage as ski
import textalloc as ta


def calulate_snr(img):
    """
    Calculate the signal to noise ratio of an image. The threshold is calculated using the Otsu method.
    The signal is the mean intensity of the pixels above the threshold and the noise is the mean of the pixels below the threshold.
    """
    thres = ski.filters.threshold_otsu(img)
    mask = img > thres
    signal = img[mask].mean()
    noise = img[~mask].mean()
    snr = signal / noise
    return snr, signal


def calculate_snr_ratio(sdata, image="raw_image", block_size=10000, channel_names=None, cycles=None):
    data = []
    channels = sdata[image].c.data
    if channel_names is None:
        channel_names = channels
    if cycles is None:
        cycles = [None] * len(channel_names)
    # for ch in channels:
    #     if ch not in ["R9 CD133", "R25 CD133"]:
    #         cycle_number.append(int(ch.split(" ")[0][1:]))
    #         channels_names.append(ch.split(" ")[-1])

    for cycle, channel_name in zip(cycles, channel_names):
        float_block = sdata[image].sel(c=channel_name).data.rechunk(block_size)
        img = float_block.compute()
        snr, signal = calulate_snr(img)
        data += [(cycle, channel_name, snr, signal)]
        del img
    df_img = pd.DataFrame(data, columns=["cycle", "channel", "snr", "signal"])
    return df_img


def snr_ratio(sdata, ax=None, loglog=True, **kwargs):
    """Plot the signal to noise ratio. On the x-axis is the signal intensity and on the y-axis is the SNR-ratio"""
    if ax is None:
        fig, ax = plt.subplots()
    df_img = calculate_snr_ratio(sdata, **kwargs)
    if loglog:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    # create scatter plot
    for i, row in df_img.iterrows():
        # color by cycle number
        if row["cycle"] is not None:
            color = plt.cm.tab20(row["cycle"] % 20)
        else:
            color = "black"
        ax.scatter(row["signal"], row["snr"], color=color)
    # use textalloc to add channel names
    x = df_img["signal"]
    y = df_img["snr"]
    ta.allocate(ax, x=x, y=y, text_list=df_img["channel"], x_scatter=x, y_scatter=y)

    ax.set_xlabel("Signal intensity")
    ax.set_ylabel("Signal-to-noise ratio")
    return ax
