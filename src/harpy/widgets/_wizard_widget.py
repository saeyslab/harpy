"""Napari widget for managing the other widgets and giving a general overview of the workflow."""

from magicgui.widgets import ComboBox, Container, Label, TextEdit
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSizePolicy

from harpy.utils import get_pylogger
from harpy.widgets import (
    allocate_widget,
    annotate_widget,
    clean_widget,
    load_widget,
    segment_widget,
)

log = get_pylogger(__name__)


# Class for step widgets
class Step:
    """Class represents the steps of the plugin. It consists of a name, a label, the widget itself and a description about the arguments."""

    def __init__(self, name, label, widget, description):
        """Initalisation of steps."""
        self.name = name
        self.label = label
        self.widget = widget
        self.description = description

    def __str__(self):
        """Return the name of the step."""
        return self.name

    def get_widget(self):
        """Return the widget of the step."""
        widget = self.widget()
        widget.name = self.name
        return widget

    def get_description(self):
        """Return the descript of the step as a read only textEdit with markdown text."""
        description = TextEdit(
            value=self.description,
            name=self.name + "description",
            enabled=True,
        )
        description.native.setReadOnly(True)
        # description.min_height = 250
        description.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        description.native.setMarkdown(self.description)
        return description


# Step as choices
def get_choices():
    """Function represents the choices that will be displayed in the selection menu of the Wizard widget."""
    return [
        (
            "Step 0: Load",
            Step(
                "load",
                "#0 Load",
                load_widget,
                (
                    "## Step 0: Loading\n"
                    "### Load an image into napari.\n"
                    "- path zarr: choose the path to a zarr store that contains a SpatialData object. If you wish to create a new zarr store from an image, leave this field blank.\n"
                    "- path image: choose the path to an image, which can be either single-channel or multi-channel. This will be disregarded if the zarr path is specified.\n"
                    "- image layer: specify the image layer within the zarr store where you want to execute Harpy. If providing a path to an image, this will designate the image layer in the zarr store that will be created.\n"
                    "- output dir:  choose the directory where all figures and files produced by Harpy will be saved.\n"
                    "- x_min, x_min, x_max, y_min and y_max (optional): select cropping region.\n"
                ),
            ),
        ),
        (
            "Step 1: Clean",
            Step(
                "clean",
                "#1 Clean",
                clean_widget,
                (
                    "## Step 1: Cleaning\n"
                    "### Improves image quality.\n"
                    "- image: select the image loaded into napari in the previous step.\n"
                    "- subset (optional): select a subset of the image by adding a rectangular shapes layer.\n"
                    "- Choose whether the tiling correction step should be run.\n"
                    "- tile size: select the tile size of the image for the tiling correction step. Note that the tile size should be a multiple of the dimension of the image, both in x and y direction.\n"
                    "- Choose whether the min max filtering step (improves object separation) should be run.\n"
                    "- size min max filter: select the size of the minimum maximum filter. A good starting point is the estimated mean diameter of the nucleus or cell. For multi-channel images, one should specify a value for each channel.\n"
                    "- Select whether the contrast enhancing step (based on OpenCV CLAHE) should be run.\n"
                    "- contrast clip: select the contrast clip for the contrast enhancing step. For multi-channel images, one should specify a value for each channel."
                ),
            ),
        ),
        (
            "Step 2: Segment",
            Step(
                "segment",
                "#2 Segment",
                segment_widget,
                (
                    "## Step 2: Segment using cellpose\n"
                    "### Segments the nuclei or cells from the image by generating a masks layer.\n"
                    "- image: select an image. Can be either the cleaned image resulting from step 1, or the image loaded into napari in step 0 (thus without cleaning).\n"
                    "- subset (optional): select a subset of the image by adding a rectangular shapes layer. Segmentation will only be run on this subset.\n"
                    "- device: select a device, either cpu or cuda.\n"
                    "- min size: select the minimal amount of pixels required in the detected masks.\n"
                    "- flow threshold: select the flow threshold. All cells with errors below threshold are kept. Higher values means more non-round cells will be kept.\n"
                    "- diameter: select the diameter, i.e. the mean expected diameter of the cells.\n"
                    "- cellprob threshold: select the cellprob threshold. Decrease to find more and larger masks.\n"
                    "- model type: select the model type used for cell segmentation. Choose nuclei for nucleus segmentation (e.g. DAPI), and cyto for cell segmentation (e.g. PolyT).\n"
                    "- channels: select channels. For single channel images, the default value ([1,0]) should not be adapted. For multi channel images, the first element of the list is the channel to segment (count from 1), and the second element is the optional nuclear channel. E.g. for an image with PolyT in second channel, and DAPI in first channel use [2,1] if you want to segment PolyT + nuclei on DAPI; [2,0] if you only want to use PolyT and [1,0] if you only want to use DAPI.\n"
                    "- voronoi radius: select the radius that will be used to expand the masks, using a voronoi diagram.\n"
                    "- chunks: select the chunksize used by cellpose.\n"
                    "- depth: select the overlapping depth used by map_overlap. Setting depth to at least 2*diameter is advised to prevent potential chunking effects.\n"
                ),
            ),
        ),
        (
            "Step 3: Allocate and cluster",
            Step(
                "Allocate and cluster",
                "#3 Allocate and cluster",
                allocate_widget,
                (
                    "## Step 3: Allocation and clustering\n"
                    "### Allocates transcripts to cells and performs leiden clustering.\n"
                    "- transcripts file: choose the file that contains the transcripts, with each row listing the x and y coordinates, along with the gene name.\n"
                    "- delimiter: choose the delimiter, by default the tab character is used.\n"
                    "- header: select whether the transcripts file contains a header.\n"
                    "- column x: select the column index of the x coordinate in the transcripts file.\n"
                    "- column y: select the column index of the y coordinate in the transcripts file.\n"
                    "- column gene: select the column index of the gene name in the transcripts file.\n"
                    "- midcount: select whether a count column is present in the transcripts file.\n"
                    "- column midcount: select the column index for the count value in the transcripts file. Ignored when midcount option is not selected.\n"
                    "- transform matrix: select the transform matrix (tab separated file). This file should contain a 3x3 transformation matrix for the affine transformation. The matrix defines the linear transformation to be applied to the coordinates of the transcripts. If no transform matrix is specified, the identity matrix will be used.\n"
                    "- min_counts: select the minimum number of transcripts a cell should contain to be kept.\n"
                    "- min_cells: select the minimum number of cells a transcript should be in to be be kept.\n"
                    "- size normalization: select whether to normalize based on the size of the cells/nuclei.\n"
                    "- n comps: select the number of components to compute in principal component analysis.\n"
                    "- min size: select the minimum size of the cells in pixels. Smaller cells will be filtered out.\n"
                    "- max size: select maximum size of the cells in pixels. Larger cells will be filtered out.\n"
                    "- pcs: select the number of prinicipal components to be used for the calculation of the neighborhood graph.\n"
                    "- neighbors: select the size of the local neighborhood used for manifold approximation for the neighborhood graph.\n"
                    "- cluster resolution: select the cluster resolution used for leiden clustering."
                ),
            ),
        ),
        (
            "Step 4: Annotate",
            Step(
                "Annotate",
                "#4 Annotate",
                annotate_widget,
                (
                    "## Step 4: Annotation\n"
                    "### Annotates cells with their cell type using marker genes:\n"
                    "- markers file: a file containing marker genes for each cell type. This file should be one-hot encoded, with cell types listed in the first row, and marker genes in the first column.\n"
                    "- delimiter: select the delimiter used by the marker genes file, by default ',' is used.\n"
                    "- del celltypes: select the cell types that should not be included in the analysis."
                ),
            ),
        ),
    ]


def wizard_widget() -> None:
    """Napari widget for managing the other widgets and giving a general overview of the workflow."""
    # Set DaMBi Icon
    icon = Label(name="icon", value="Made by DaMBi")
    pixmap = QPixmap("./src/harpy/widgets/dambi-white.png")
    icon.native.setPixmap(pixmap)

    # Step selector
    step = ComboBox(label="Step:", choices=get_choices(), name="step")

    # Global container holds all widgets
    container = Container(
        name="global",
        widgets=[
            icon,
            step,
            get_choices()[0][1].get_description(),  # Set first step on startup
            get_choices()[0][1].get_widget(),
        ],
        labels=False,
        layout="vertical",
    )
    container.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def step_changed(event):
        """Callback that updates the current step when the step menu selection changes."""
        name = str(event)

        # Add widget if not yet exists
        if name not in [x.name for x in container._list]:
            container.append(event.get_description())
            container.append(event.get_widget())

        # Hide other widgets
        for widget in list(container):
            if widget.name not in ["icon", "step", name, name + "description"]:
                widget.visible = False
            else:
                widget.visible = True

    step.changed.connect(step_changed)

    return container
