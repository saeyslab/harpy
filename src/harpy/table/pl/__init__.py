"""Plots of the columns of an AnnData table, in the style of ``scanpy.pl``."""

from harpy.table.pl._histogram import histogram, histograms
from harpy.table.pl._scatter import scatter

__all__ = ["histogram", "histograms", "scatter"]
