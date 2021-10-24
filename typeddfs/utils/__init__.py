"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

from typing import Set

from tabulate import TableFormat

from typeddfs.utils._utils import (
    _AUTO_DROPPED_NAMES,
    _DEFAULT_HASH_ALG,
    _FORBIDDEN_NAMES,
)
from typeddfs.utils.dtype_utils import DtypeUtils
from typeddfs.utils.io_utils import IoUtils
from typeddfs.utils.json_utils import JsonUtils
from typeddfs.utils.misc_utils import MiscUtils
from typeddfs.utils.parse_utils import ParseUtils
from typeddfs.utils.sort_utils import SortUtils


class Utils(MiscUtils, SortUtils, JsonUtils, IoUtils, ParseUtils, DtypeUtils):

    json_encoder = JsonUtils.encoder
    json_decoder = JsonUtils.decoder

    @classmethod
    def default_hash_algorithm(cls) -> str:
        return _DEFAULT_HASH_ALG

    @classmethod
    def insecure_hash_functions(cls) -> Set[str]:
        return {"md5", "sha1"}

    @classmethod
    def banned_names(cls) -> Set[str]:
        """
        Lists strings that cannot be used for column names or index level names.
        """
        return {*_AUTO_DROPPED_NAMES, *_FORBIDDEN_NAMES}


__all__ = ["Utils", "TableFormat"]
