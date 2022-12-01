"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""


from magicgui.widgets import ComboBox, Container, Label, TextEdit
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSizePolicy

from napari_sparrow.utils import get_pylogger
from napari_sparrow.widgets import (
    allocate_widget,
    annotate_widget,
    clean_widget,
    export_widget,
    segment_widget,
)

log = get_pylogger(__name__)


# Class for step widgets
class Step:
    """This class represents the steps of the plugin.
    It consists of a name, a label, the widget itself and a description about the arguments.
    """

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
    """This function represents the choices that will be displayed in the selection menu of the Wizard widget."""
    return [
        (
            "Step 1: Clean",
            Step(
                "clean",
                "#1 Clean",
                clean_widget,
                """## Step 1: Cleaning\n### Improves image quality:\n- Select an image\n- Select a subset of the image by adding a rectangular shapes layer\n - The tilesize of the image\n - The contrast adjusment based on OpenCV CLAHE\n - The tophat filter improves object separation""",
            ),
        ),
        (
            "Step 2: Segment",
            Step(
                "segment",
                "#2 Segment",
                segment_widget,
                """## Step 2: Segmentation\n### Segments the cells from the image:\n- Select an image\n - Select a subset of the image by adding a rectangular shapes layer\n - Select a device, either cpu or cuda:0\n - Minimum amount of pixels in mask\n - Shape of the mask filter, higher means less round\n - Mean expected diameter\n - Threshold amount mask kept, smaller means more\n - Model type\n - Channel selection for images with multiple channels""",
            ),
        ),
        (
            "Step 3: Allocate",
            Step(
                "Allocate",
                "#3 Allocate",
                allocate_widget,
                """## Step 3: Allocation\n### Allocates transcripts to cells:\n - Select the transcript file (.txt)\n - Enter the library name, will be used in AnnData\n - Minimum size of cells, smaller filtered out\n - Maximum size of cells, larger filtered out\n - Use this many PCs for neighborhood graph\n - The size of local neighborhood used for manifold approximation for neighborhood graph\n - Cluster resolution controls the coarseness of the leiden clustering\n - Number of components to compute in principal component analysis""",
            ),
        ),
        (
            "Step 4: Annotate",
            Step(
                "Annotate",
                "#4 Annotate",
                annotate_widget,
                """## Step 4: Annotation\n### Annotates cells with celltype:\n - Marker genes file with marker genes per celltype (.csv)\n - Normalize rows""",
            ),
        ),
        (
            "Step 5: Export",
            Step(
                "Export",
                "#5 Export",
                export_widget,
                """## Step 5 Export:\n### Checks cluster cleanliness and saves data\n - Select the output directory""",
            ),
        ),
    ]


def wizard_widget() -> None:
    """
    Napari widget for managing the other widgets and giving a general overview of the workflow.
    """

    # Set DaMBi Icon   -- CHECKME: this icon takes up a lot of screen real estate in the user interface, leave it out?
    # icon = Label(name="icon", value="Made by DaMBi")
    # pixmap = QPixmap("./src/napari_sparrow/widgets/dambi-white.png")
    # icon.native.setPixmap(pixmap)

    # Step selector
    step = ComboBox(label="Step:", choices=get_choices(), name="step")

    # Global container holds all widgets
    container = Container(
        name="global",
        widgets=[
            # icon,
            step,
            get_choices()[0][1].get_description(),  # Set first step on startup
            get_choices()[0][1].get_widget(),
        ],
        labels=False,
        layout="vertical",
    )
    container.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def step_changed(event):
        """This is a callback that updates the current step
        when the step menu selection changes
        """
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
