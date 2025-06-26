import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'EnergyML'
author = 'Juan Felipe Rodriguez'
release = '1.0.0'

extensions = ['sphinx.ext.autodoc']
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
