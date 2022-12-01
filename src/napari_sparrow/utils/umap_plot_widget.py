import math
import numpy as np
import pandas as pd
from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvas
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel


class UMAPPlotWidget(QWidget):
    """ A QWidget for displaying a UMAP plot. """

    # The UMAPPlotWidget offers a subset of the functionality of scanpy.pl.umap() 
    # but with more control over plot and legend aesthetics.

    def __init__(self):
        super().__init__()

        self.canvas = FigureCanvas()
        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.patch.set_facecolor("#262930")

        self.axes = self.canvas.figure.subplots()
        self.axes.set_facecolor("#262930")

        self.legend = QLabel()

        layout = QVBoxLayout()

        self.setLayout(layout)
        self.layout().addWidget(self.canvas)  # TODO: reduce spacing between plot and legend
        self.layout().addWidget(self.legend)

        self.data: Optional[np.ndarray] = None  # an array of shape (num_cells, 2) with the 2D UMAP coordinates of each cell
        self.cluster_colors: np.ndarray = np.array([], dtype='O')  # array of color strings; one string per cluster
        self.cluster_ids: pd.Series = pd.Series([], dtype='category')  # Id of the cluster that each cell belongs to

    def set_data(self, 
                 data: np.ndarray,   
                 cluster_ids: pd.Series,
                 cluster_colors: np.ndarray):
        """ Provide the coordinates of the points in the UMAP plot, 
        as well as color information for the different clusters in the plot. """

        assert cluster_ids.dtype == 'category'
        
        self.data = data
        self.cluster_ids = cluster_ids
        self.cluster_colors = cluster_colors

    def draw(self,
             legend_columns: int = 1,
             point_size: int = 1):

        """ Draw the UMAP plot with a legend underneath. """
        if self.data is None:
            return

        cluster_ids = self.cluster_ids.cat.codes
        colors = self.cluster_colors[cluster_ids]   # colors[i] is the color for cell i

        umap1 = self.data[:, 0]
        umap2 = self.data[:, 1]

        self.axes.scatter(umap1, umap2, c=colors, marker='.', s=point_size)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.canvas.draw()

        text = self._get_legend_html_text(legend_columns)
        self.legend.setText(text)

    def clear(self):
        """ Clear the UAMP plot and the associated legend. """
        self.axes.clear()
        self.legend.setText('')

    def _get_legend_html_text(self, columns: int, width: int = 120):
        # To display the UMAP legend with the names of the different clusters,
        # we will use an HTML table. The table has a given number of columns, 
        # and a given width in pixels.

        filled_dot = '&#x2B24;'
        colors = self.cluster_colors  # colors[i] is the color of the i-th cluster in the plot
        names = self.cluster_ids.cat.categories  # names[i] is the name of the i-th cluster in the plot (e.g. a cell type)

        rows = math.ceil(len(colors) / columns)

        # Note: QLabel text uses a HTML dialect. For example, absolute width attributes
        # *must not* use the "px" suffix, otherwise Qt simply ignores the width indication.

        text = f'<table align="center" width="{width}">>'
        for row in range(rows):
            text += '<tr>'
            for column in range(columns):
                text += f'<td width="{int(width/columns)}">'
                i = column * rows + row
                if i < len(colors):
                    color = colors[i]
                    text += f'<span style="color:{color};">{filled_dot}</span>&nbsp;{names[i]}'
                text += '</td>'
            text += '</tr>'
        text += '</table>'
        return text
