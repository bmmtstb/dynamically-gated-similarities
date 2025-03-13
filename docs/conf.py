"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory, add these directories to sys.path here.
# If the directory is relative to the documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from importlib.metadata import PackageNotFoundError, version as ilib_version

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tracking via Dynamically Gated Similarities"
# noinspection PyShadowingBuiltins
copyright = "2023, Martin Steinborn"
author = "Martin Steinborn"

try:
    __version__ = ilib_version("dynamically_gated_similarities")
except PackageNotFoundError:
    __version__ = "0.0.0"

# The short X.Y version
version = __version__[: __version__.find(".", __version__.find(".") + 1)]
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.coverage",
    "sphinxcontrib.datatemplates",  # load default parameters from yaml
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.video",  # display videos in README
]

# force the primary domain to be python
primary_domain = "py"

# enable autosummary
autosummary_generate = True

# napoleon (google style docstrings with rst)
napoleon_attr_annotations = False
napoleon_custom_sections = [
    ("Params", "params_style"),
    ("Optional Params", "params_style"),
    ("Important Inherited Params", "params_style"),
]

# linkcheck
linkcheck_ignore = [
    ".*bmmtstb\.github\.io\/dynamically\-gated\-similarities\/.*",  # Server is down while running docs workflow
    ".*stackoverflow\.com\/questions\/8391411\/how\-to\-block\-calls\-to\-print\/.*",  # fixme: why doesn't this link work?
]
linkcheck_anchors_ignore_for_url = [  # some problem with GitHub text-anchors
    ".*github\.io.*",
    ".*github\.com.*",
    ".*stackoverflow\.com.*",
]

# settings for autosummary
nitpick_ignore = [
    ("py:class", "torch.device"),
    ("py:class", "torch.Tensor"),
    ("py:class", "torchvision.tv_tensors.BoundingBoxes"),
    ("py:class", "torchvision.tv_tensors.BoundingBoxesFormat"),
    ("py:class", "torchvision.tv_tensors.Image"),
    ("py:class", "torchvision.tv_tensors.Mask"),
]

# custom shortcuts
rst_prolog = """
.. |AP| replace:: ``AlphaPose``
.. _AP: https://github.com/MVIG-SJTU/AlphaPose/
.. |PT21| replace:: ``PoseTrack21``
.. _PT21: https://openaccess.thecvf.com//content/CVPR2022/html/Doring_PoseTrack21_A_Dataset_for_Person_Search_Multi-Object_Tracking_and_Multi-Person_CVPR_2022_paper.html
.. |torchreid| replace:: ``torchreid``
.. _torchreid: https://github.com/KaiyangZhou/deep-person-reid
.. |DT| replace:: ``DanceTrack``
.. _DT: https://dancetrack.github.io/
.. |MOT| replace:: ``MOT``
.. _MOT: https://motchallenge.net/
.. |MOTA| replace:: :ref:` ``MOTA`` <metrics_mota>`
.. |HOTA| replace:: :ref:` ``HOTA`` <metrics_hota>`
"""

# tell autodoc that we don't want these packages to be imported
autodoc_mock_imports = [
    "cv2",
    "posetrack21",
    "torch",
    "torcheval",
    "torchvision",
    "torchreid",
]

# File parsing
source_suffix = [".rst", ".md"]
source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/venv/**",
    "**/site-packages/**",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nature"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
