# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
try:
    import peal

except Exception:
    sys.path.insert(0, os.path.abspath('..'))
    import peal

# The module you're documenting (assumes you've added the parent dir to sys.path)
project = 'peal'
copyright = '2023, Sidney Bender'
author = 'Sidney Bender'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib_autodocgen',
]

# templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_static_path = ['_static']

# Path to the root of your Python codebase


autodocgen_config = [{
    'modules': [peal],
    'generated_source_dir': '../docs/api',

    # produce a text file containing a list of everything documented. you can use this in a test to notice when you've
    # intentionally added/removed/changed a documented API
    'write_documented_items_output_file': 'autodocgen_documented_items.txt',

    # choose a different title for specific modules, e.g. the toplevel one
    'module_title_decider': lambda modulename: 'API Reference' if modulename == 'peal' else modulename,
}]

# List of modules to generate documentation for
# You can also use wildcards to include all modules in a package
# or subpackage, e.g. 'my_package.*'
# Note that the order of the modules in the list determines the
# order in which they will appear in the generated documentation
# To include submodules, use the 'submodules' option
# To include private members (those starting with an underscore),
# use the 'private-members' option
# To include special members (e.g. __init__), use the 'special-members' option
# For more options, see the Sphinx documentation
# (https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-auto-module)
# autosummary_generate = ['peal.data.dataset_interfaces']  # ['peal.*']

# Options for autodoc extension
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}
