"""
Metadata for TypedDfs.
"""

from pathlib import Path

# importlib.metadata is compat with Python 3.8 only
from importlib_metadata import metadata as __load

from typeddfs.typed_dfs import BaseFrame, SimpleFrame, FinalFrame, ConvertibleFrame, OrganizingFrame

metadata = __load(Path(__file__).parent.name)
__status__ = "Development"
__copyright__ = "Copyright 2015–2020"
__date__ = "2020-05-12"
__uri__ = metadata["home-page"]
__title__ = metadata["name"]
__summary__ = metadata["summary"]
__license__ = metadata["license"]
__version__ = metadata["version"]
__author__ = metadata["author"]
__maintainer__ = metadata["maintainer"]
__contact__ = metadata["maintainer"]