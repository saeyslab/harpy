import numpy as np
import scanpy as sc

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import joypy

except ImportError:
    log.warning(
        "'joypy' not installed, to use 'sp.pl.ridgeplot_channel' and 'sp.pl.ridgeplot_channel_sample', please install this library."
    )


def calculate_asinh(adata, q=0.2):
    df = adata.to_df()
    divider = 5 * np.quantile(df, q=q, axis=0)
    divider[divider == 0] = df.max(axis=0)[divider == 0]

    scaled = np.arcsinh(df / divider)
    transformed_df = (scaled - scaled.mean(0)) / scaled.std(0)
    adata.layers["arcsinh"] = transformed_df
    return adata


def heatmap(adata, **kwargs):
    sc.pl.heatmap(adata, **kwargs)


def plot_adata(adata, path_prefix):
    """_summary_

    :param adata: _description_
    :param path_prefix: _description_
    """
    for layer in adata.layers:
        f = ridgeplot_channel(adata, layer=layer)
        f.savefig(path_prefix + f"_ridgeplot_all_channels_{layer}.png")
        if "sample" in adata.obs:
            ridgeplot_channel_sample(adata, layer=layer, path_prefix=path_prefix)


def ridgeplot_channel(adata, layer=None):
    """_summary_

    :param adata: _description_
    :param layer: _description_, defaults to None
    :return: _description_
    """
    df = adata.to_df(layer=layer)
    df_melted = df.melt(var_name="channel")
    fig, ax = joypy.joyplot(df_melted, by="channel", column="value", legend=True, alpha=0.4, grid=True)
    return ax


# def ridgeplot_channel_sample(
#     adata, y="sample", channel_name="channel", channels=None, layer=None, path_prefix=None, **kwargs
# ):
#     """_summary_

#     :param adata: _description_
#     :param layer: _description_, defaults to None
#     :param path_prefix: _description_, defaults to None
#     :yield: _description_
#     """
#     df = adata.to_df(layer=layer)
#     df[y] = adata.obs[y]
#     df_melted = df.melt(id_vars=[y], var_name=channel_name)
#     for group_name, group in df_melted.groupby(channel_name):
#         if channels:
#             if group_name not in channels:
#                 continue
#         fig, ax = joypy.joyplot(
#             group, by=y, column="value", legend=True, alpha=0.4, grid=True, title=group_name, **kwargs
#         )
#         if path_prefix is not None:
#             fig.savefig(path_prefix + f"_ridgeplot_{group_name}_{layer}.png")
#         return ax


def ridgeplot_channel_sample(
    adata, y="sample", channel_name="channel", value_vars=None, layer=None, path_prefix=None, **kwargs
):
    """_summary_

    :param adata: _description_
    :param layer: _description_, defaults to None
    :param path_prefix: _description_, defaults to None
    :yield: _description_
    """
    df = adata.to_df(layer=layer)
    df[y] = adata.obs[y]
    df_melted = df.melt(id_vars=[y], var_name=channel_name, value_vars=value_vars)
    for group_name, group in df_melted.groupby(channel_name):
        fig, _ = joypy.joyplot(
            group, by=y, column="value", legend=True, alpha=0.4, grid=True, title=group_name, **kwargs
        )
        if path_prefix is not None:
            fig.savefig(path_prefix + f"_ridgeplot_{group_name}_{layer}.png")
