"""Sphinx configuration for the Quivira documentation."""

from datetime import datetime

import quivira


project = "Quivira"
author = "The Quivira developers"
copyright = f"{datetime.now().year}, {author}"
release = quivira.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

intersphinx_mapping = {
    "heyoka": ("https://bluescarni.github.io/heyoka.py/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Quivira"
html_logo = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/esa/quivira",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16" aria-hidden="true">
                    <path d="M8 0C3.58 0 0 3.64 0 8.13c0 3.59 2.29 6.64 5.47 7.71.4.08.55-.18.55-.39 0-.19-.01-.83-.01-1.51-2.01.38-2.53-.5-2.69-.96-.09-.23-.48-.96-.82-1.15-.28-.15-.68-.53-.01-.54.63-.01 1.08.59 1.23.83.72 1.23 1.87.88 2.33.67.07-.53.28-.88.51-1.08-1.78-.21-3.64-.91-3.64-4.02 0-.89.31-1.62.82-2.19-.08-.21-.36-1.04.08-2.16 0 0 .67-.22 2.2.84A7.5 7.5 0 0 1 8 3.91c.68 0 1.36.09 2 .27 1.53-1.06 2.2-.84 2.2-.84.44 1.12.16 1.95.08 2.16.51.57.82 1.3.82 2.19 0 3.12-1.87 3.81-3.65 4.02.29.25.54.74.54 1.51 0 1.09-.01 1.97-.01 2.24 0 .22.15.47.55.39A8.15 8.15 0 0 0 16 8.13C16 3.64 12.42 0 8 0Z"/>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/esa/quivira/",
    "source_branch": "main",
    "source_directory": "doc/",
    "light_css_variables": {
        "color-brand-primary": "#006b5e",
        "color-brand-content": "#006b5e",
        "color-api-name": "#9f2d20",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ee0c2",
        "color-brand-content": "#5ee0c2",
        "color-api-name": "#ff9d8d",
    },
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
nbsphinx_execute = "auto"
nbsphinx_allow_errors = False

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True