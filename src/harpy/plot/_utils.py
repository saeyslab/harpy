import matplotlib.colors as mcolors
import scanpy as sc
import seaborn as sns


def _get_distinct_colors(n: int):
    """
    Return n distinct hex colors.

    - Up to 20: use Scanpy's default_20
    - Up to 102: use Scanpy's default_102
    - Beyond that: extend with HUSL-generated colors
    """
    # Base palettes from scanpy
    PALETTE_20 = list(sc.plotting.palettes.default_20)
    PALETTE_102 = list(sc.plotting.palettes.default_102)
    if n <= len(PALETTE_20):
        return PALETTE_20[:n]
    elif n <= len(PALETTE_102):
        return PALETTE_102[:n]
    else:
        extra_needed = n - len(PALETTE_102)
        extra_colors = sns.color_palette("husl", extra_needed)
        extra_hex = [mcolors.to_hex(c) for c in extra_colors]
        return PALETTE_102 + extra_hex
