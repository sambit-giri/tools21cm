# Configuration file for the Sphinx documentation builder.
# For a full list of options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_readable_theme

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../src/'))

# -- Project information -----------------------------------------------------
project = 'tools21cm'
copyright = '2020, Sambit Giri'
author = 'Sambit Giri'
version = release = '2.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "nbsphinx",
]

templates_path = ['templates']

exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'templates',
]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'readable'
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
pygments_style = "trac"
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]
}
html_short_title = project

# -- Napoleon settings -------------------------------------------------------
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# -- MathJax path ------------------------------------------------------------
mathjax_path = (
    "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

# -- RST Epilog --------------------------------------------------------------
rst_epilog = """
.. |Cosmology| replace:: :class:`~astropy.cosmology.Cosmology`
.. |Table| replace:: :class:`~astropy.table.Table`
.. |Quantity| replace:: :class:`~astropy.units.Quantity`
"""
