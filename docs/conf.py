# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
"""
Sphinx config file.
Uses several extensions to get API docs and sourcecode.
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from pathlib import Path
from typing import TypeVar

import tomlkit

# This assumes that we have the full project root above, containing pyproject.toml
_root = Path(__file__).parent.parent.absolute()
_toml = tomlkit.loads((_root / "pyproject.toml").read_text(encoding="utf-8"))

T = TypeVar("T")


def find(key: str, default: T | None = None, as_type: type[T] = str) -> T | None:
    """
    Gets a value from pyproject.toml, or a default.
    Args:
        key: A period-delimited TOML key; e.g. ``tools.poetry.name``
        default: Default value if any node in the key is not found
        as_type: Convert non-``None`` values to this type before returning
    Returns:
        The value converted to ``as_type``, or ``default`` if it was not found
    """
    at = _toml
    for k in key.split("."):
        at = at.get(k)
        if at is None:
            return default
    return as_type(at)


# Basic information used by Sphinx
language = "en-US"
project = find("tool.poetry.name")
version = find("tool.poetry.version")
release = version
author = ", ".join(find("tool.poetry.authors", as_type=list))

# Copyright string (for documentation)
# It's not clear whether we're supposed to, but we'll add the license
# noinspection PyShadowingBuiltins
copyright = "Contributors to typed-dfs, 2016-2023"
_license = "Apache License, Version 2.0"
_license_url = "https://www.apache.org/licenses/LICENSE-2.0"

# Load extensions
# These should be in docs/requirements.txt
# Napoleon is bundled in Sphinx, so we don't need to list it there
# NOTE: 'autoapi' here refers to sphinx-autoapi
# See https://sphinx-autoapi.readthedocs.io/
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "furo",
]
master_doc = "index"
napoleon_include_special_with_doc = True
autoapi_type = "python"
autoapi_dirs = [str(_root / project)]
autoapi_keep_files = True
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_options = ["private-members", "undoc-members", "special-members"]
html_theme = "furo"

# doc types to build
sphinx_enable_epub_build = False
sphinx_enable_pdf_build = False
exclude_patterns = ["_build", "Thumbs.db", ".*", "~*", "*~", "*#"]

if __name__ == "__main__":
    print(f"{project} v{version}\n© Copyright {copyright}")
