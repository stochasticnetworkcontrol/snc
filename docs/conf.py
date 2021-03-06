# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings
# Temp. workaround for https://github.com/agronholm/sphinx-autodoc-typehints/issues/133
warnings.filterwarnings('ignore', message = 'sphinx.util.inspect.Signature\(\) is deprecated')

# Point to root source dir for API doc, relative to this file:
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'SNC'
author = 'SNC Opensource Group'

# The full version, including alpha/beta/rc tags
release = '1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.mathjax',  # Render math via Javascript
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'nbsphinx',  # Integrate Jupyter Notebooks with Sphinx
    'IPython.sphinxext.ipython_console_highlighting'
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc (.inv file)
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    # Unfort. doesn't work yet! See https://github.com/mr-ubik/tensorflow-intersphinx/issues/1
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/mr-ubik/tensorflow-intersphinx/master/tf2_py_objects.inv"
    ),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
#autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints

# Add any paths that contain Jinja2 templates here, relative to this directory.
templates_path = ['_templates']

# Exclusions
# To exclude a module, use autodoc_mock_imports. Note this may increase build time, a lot.
# (Also, when installing on readthedocs.org, we omit installing Tensorflow and
# Tensorflow Probability so mock them here instead.)
#autodoc_mock_imports = [
    # 'tensorflow',
    # 'tensorflow_probability',
#]
# To exclude a class, function, method or attribute, use autodoc-skip-member. (Note this can also
# be used in reverse, ie. to re-include a particular member that has been excluded.)
# 'Private' and 'special' members (_ and __) are excluded using the Jinja2 templates; from the main
# doc by the absence of specific autoclass directives (ie. :private-members:), and from summary
# tables by explicit 'if-not' statements. Re-inclusion is effective for the main doc though not for
# the summary tables.
# def autodoc_skip_member_callback(app, what, name, obj, skip, options):
#     # This would exclude the Matern12 class and to_default_float function:
#     exclusions = ('Matern12', 'to_default_float')
#     # This would re-include __call__ methods in main doc, previously excluded by templates:
#     inclusions = ('__call__')
#     if name in exclusions:
#         return True
#     elif name in inclusions:
#         return False
#     else:
#         return skip
# def setup(app):
#     # Entry point to autodoc-skip-member
#     app.connect("autodoc-skip-member", autodoc_skip_member_callback)

#https://sphinxguide.readthedocs.io/en/latest/sphinx_basics/settings.html
# -- Options for LaTeX -----------------------------------------------------
latex_elements = {
'preamble': r'''
\usepackage{amsmath,amsfonts,amssymb,amsthm}
''',
}


# -- Options for HTML output -------------------------------------------------

# on_rtd is whether on readthedocs.org, this line of code grabbed from docs.readthedocs.org...
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["color_theme.css"]
