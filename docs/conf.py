# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from fnmatch import fnmatch
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

info = metadata("harpy-analysis")
project = "Harpy"
author = "SaeysLab"
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

# Source repository for the theme's "Edit this page" link and repository card.
html_context = {
    "source_type": "github",
    "source_user": "saeyslab",
    "source_repo": "harpy",
    "source_version": "main",
    "source_docs_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_process_signature = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
# An Attributes section becomes a field list, so the class template's attribute
# entries are the only object descriptions of the attributes.
napoleon_use_ivar = True
myst_heading_anchors = 3  # create anchors for h1-h3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
# Render ```mermaid blocks as diagrams, as GitHub does.
myst_fence_as_directive = ["mermaid"]
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "rapids_singlecell": ("https://rapids-singlecell.scverse.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "datatree": ("https://datatree.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
    "spatialdata_io": ("https://spatialdata.scverse.org/projects/io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", "**.ipynb_checkpoints", "../notebooks/*", "../experiments/*", "../configs/*"]
# The tutorials live in the spatial multi-omics training repository. Of the
# tutorials submodule, only these pages are still built.
_BUILT_TUTORIALS = ("general/techno_specific.ipynb", "general/benchmark.ipynb", "hpc/*")
_TUTORIALS = HERE / "tutorials"
exclude_patterns += [
    f"tutorials/{page}"
    for page in sorted(
        path.relative_to(_TUTORIALS).as_posix() for path in _TUTORIALS.rglob("*") if path.suffix in {".md", ".ipynb"}
    )
    if not any(fnmatch(page, built) for built in _BUILT_TUTORIALS)
]
# Ignore warnings.
nitpicky = False  # TODO: solve upstream.
# nitpick_ignore = [
#     ("py:class", "spatial_image.SpatialImage"),
#     ("py:class", "multiscale_spatial_image.multiscale_spatial_image.MultiscaleSpatialImage"),
# ]


# -- Options for HTML output -------------------------------------------------

# Shibuya theme: https://shibuya.lepture.com/
html_theme = "shibuya"
html_static_path = ["_static"]
html_title = project
html_logo = "_static/img/logo.png"

# Leave `pygments_style` unset: Shibuya then switches code highlighting
# between its light and dark styles with the color mode.
html_theme_options = {
    "accent_color": "orange",
    "github_url": "https://github.com/saeyslab/harpy",
    "nav_links": [
        {"title": "Installation", "url": "installation"},
        {"title": "Quickstart", "url": "quickstart"},
        {"title": "API", "url": "api"},
        {"title": "Tutorials", "url": "https://github.com/vibspatial/spatial_multi_omics_training", "external": True},
    ],
}

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    ("py:class", "igraph.Graph"),
]
