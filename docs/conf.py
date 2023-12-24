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

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tracking via Dynamically Gated Similarities"
# noinspection PyShadowingBuiltins
copyright = "2023, Martin Steinborn"
author = "Martin Steinborn"

version_file = "../dgs/__init__.py"
with open(version_file, "r") as f:
    exec(compile(f.read(), version_file, "exec"))
__version__ = locals()["__version__"]
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
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
linkcheck_anchors_ignore_for_url = [  # some problem with GitHub text-anchors
    ".*github\.io.*",
    ".*github\.com.*",
]
# tell autodoc that we don't want these packages to be imported
autodoc_mock_imports = [
    "alphapose",
    "detector",
    "halpecocotools",
    "opencv-python",
    # "matplotlib",
    "natsort",
    # "numpy",
    "pytorch",
    "torch",
    "torchvision",
    "torchreid",
    # "tqdm",
    "visdom",
]

# File parsing
source_suffix = [".rst", ".md"]
source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*/venv*", "venv*"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nature"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
