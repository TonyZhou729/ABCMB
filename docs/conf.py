# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ABCMB'
copyright = '2025, Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
author = 'Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# # Make sure Sphinx can import your code
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # if your package is in repo root


templates_path = ['_templates']
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx']
intersphinx_mapping = {
    'hyrex': ('https://hyrex.readthedocs.io/en/latest/', None),
}
exclude_patterns = [
    'ABCMB/hyrex',      # exclude the subpackage
    '../ABCMB/hyrex',   # relative path safety
    'HyRex',            # in case of capitalization variants
    'ABCMB/linx',
    '../ABCMB/linx',
    'LINX',
    '_build',
    'Thumbs.db',
    '.DS_Store',
]




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
