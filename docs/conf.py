import os
import sys

sys.path.insert(
    0, os.path.abspath("../src")
)  # Adjust the path to point to your 'src' directory

project = "ACDS3 Wildfire Atlas"
copyright = "2024, Tim Du, Quan Gu, Jiangnan Meng, Shiyuan Meng, Daniel Seal, Anna Smith, Maddy Wiebe, Xuan Zhan"
author = "Tim Du, Quan Gu, Jiangnan Meng, Shiyuan Meng, Daniel Seal, Anna Smith, Maddy Wiebe, Xuan Zhan"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "English"

html_theme = "alabaster"
html_static_path = ["_static"]
