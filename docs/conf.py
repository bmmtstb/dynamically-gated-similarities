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

PROJECT = "Tracking via Dynamically Gated Similarities"
COPYRIGHT = "2023, Martin Steinborn"
AUTHOR = "Martin Steinborn"

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
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
]

# File parsing
source_suffix = [".rst", ".md"]
source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

language = None

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

HTML_THEME = "classic"
HTML_STATIC_PATH = ["_static"]