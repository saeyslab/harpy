"""Every element and coordinate system name the CosMx reader writes.

The one inventory of the naming rules of the reader's SpatialData elements and
coordinate systems: a module that writes such a name, or plans one to check
for collisions before writing, takes it from here. A sample's names start with
its sample ID (:func:`_sample_name`). Each mosaic of the sample then gets one
element per product (:func:`_element_name`) and its own pixel and micron
coordinate systems (:func:`_pixel_coordinate_system`,
:func:`_micron_coordinate_system`). For mosaic ``<n>`` of a sample::

    <sample>_<base>_mosaic_<n>                 element of one product
    <sample>_<coordinate_system>_<n>           pixel coordinate system
    <sample>_<coordinate_system>_<n>_micron    micron coordinate system

``<base>`` is the product's output name, such as ``morphology_image``, and
``<coordinate_system>`` the sample's base name for its coordinate systems,
``global`` by default.
"""


def _sample_name(sample_id: str, base: str) -> str:
    """Prefix a base name with its sample: ``{sample_id}_{base}``."""
    return f"{sample_id}_{base}"


def _element_name(base: str, mosaic: int) -> str:
    """Name the element of one product for one mosaic: ``{base}_mosaic_{mosaic}``."""
    return f"{base}_mosaic_{mosaic}"


def _pixel_coordinate_system(base: str, mosaic: int) -> str:
    """Name the pixel coordinate system of one mosaic: ``{base}_{mosaic}``."""
    return f"{base}_{mosaic}"


def _micron_coordinate_system(pixel_coordinate_system: str) -> str:
    """Name the physical counterpart of a pixel coordinate system: ``{pixel_coordinate_system}_micron``."""
    return f"{pixel_coordinate_system}_micron"
