# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Eyes On Me'
copyright = '2024, AI Team'
author = 'AI Team'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_static_path = ['_static']

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../')) # To find src modules and main.py

# -- Extension configuration -------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # Core library for autodoc
    'sphinx.ext.napoleon',   # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinx.ext.githubpages',# Facilitates GitHub Pages deployment
    'sphinx_rtd_theme',    # Read the Docs theme
]

# Napoleon settings - minimal
napoleon_google_docstring = True
# napoleon_numpy_docstring = False # Keep this commented or set to False if primarily Google style
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_param = True
# napoleon_use_rtype = True


# HTML theme
html_theme = 'sphinx_rtd_theme'
```
