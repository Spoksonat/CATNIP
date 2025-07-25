"""
Sphinx configuration file for ReadTheDocs documentation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'CATNIP'
author = 'Manuel Fernando Sanchez Alarcon'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
