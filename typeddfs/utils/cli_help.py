r"""
Utils for getting nice CLI help on DataFrame inputs.

.. attention::
    The exact text used in this module are subject to change.

.. note ::
    Two consecutive newlines (``\n\n``) are used to separate sections.
    This is consistent with a number of formats, including Markdown,
    reStructuredText, and `Typer <https://github.com/tiangolo/typer>`_.
"""
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Sequence, Type

from typeddfs import MatrixDf, TypedDf
from typeddfs.abs_dfs import AbsDf
from typeddfs.df_typing import DfTyping
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.utils import Utils

# noinspection PyProtectedMember
from typeddfs.utils._utils import _FLEXWF_SEP, _HDF_KEY, _PICKLE_VR, _TOML_AOT


@dataclass(frozen=True, repr=True, order=True)
class DfFormatHelp:
    """
    Help text on a specific file format.
    """

    fmt: FileFormat
    desc: str

    @property
    def bare_suffixes(self) -> Sequence[str]:
        """
        Returns all suffixes, excluding compressed variants (etc. ``.gz``), naturally sorted.
        """
        suffixes = {CompressionFormat.strip_suffix(s).name for s in self.fmt.suffixes}
        return Utils.natsort(suffixes, str)

    @property
    def all_suffixes(self) -> Sequence[str]:
        """
        Returns all suffixes, naturally sorted.
        """
        return Utils.natsort(self.fmt.suffixes, str)

    def get_text(self) -> str:
        """
        Returns a 1-line string of the suffixes and format description.
        """
        suffixes = self.bare_suffixes
        if self.fmt.is_text:
            comps = ["." + c.name for c in CompressionFormat.list_non_empty()]
            return "/".join(suffixes) + "[" + "/".join(comps) + "]" + ": " + self.desc
        else:
            return "/".join(suffixes) + ": " + self.desc


class DfFormatsHelp(FrozenSet[DfFormatHelp]):
    """
    Help on file formats only.
    """

    def get_short_text(self, *, recommended_only: bool = False) -> str:
        """
        Returns a single-line text listing of allowed file formats.

        Args:
            recommended_only: Skip non-recommended file formats

        Returns:
            Something like::
                .csv, .tsv/.tab, or .flexwf [.gz,/.xz,/.zip/.bz2]; .feather, .pickle, or .snappy ...
        """
        fmts = [f for f in self if not recommended_only or f.fmt.is_recommended]
        text_fmts = Utils.natsort(
            ["/".join(f.bare_suffixes) for f in fmts if f.fmt.is_text], dtype=str
        )
        bin_fmts = Utils.natsort(
            ["/".join(f.bare_suffixes) for f in fmts if f.fmt.is_binary], dtype=str
        )
        txt = ""
        if len(text_fmts) > 0:
            txt += (
                Utils.join_to_str(*text_fmts, last="or")
                + " ["
                + "/".join([s.suffix for s in CompressionFormat.list_non_empty()])
                + "]"
            )
        if len(bin_fmts) > 0:
            txt += ("; " if len(text_fmts) > 0 else "") + Utils.join_to_str(*bin_fmts, last="or")
        return txt

    def get_long_text(
        self,
        *,
        recommended_only: bool = False,
        nl: str = "\n",
        bullet: str = "- ",
        indent: str = "  ",
    ) -> str:
        r"""
        Returns a multi-line text listing of allowed file formats.

        Args:
            recommended_only: Skip non-recommended file formats
            nl: Newline characters; use "\n", "\\n", or " "
            bullet: Prepended to each item
            indent: Spaces for nested indent

        Returns:
            Something like::
                [[ Supported formats ]]:

                .csv[.bz2/.gz/.xz/.zip]: comma-delimited

                .parquet/.snappy: Parquet

                .h5/.hdf/.hdf5: HDF5 (key 'df') [discouraged]

                .pickle/.pkl: Python Pickle [discouraged]
        """
        bullet = nl + indent + bullet
        fmts = [f for f in self if not recommended_only or f.fmt.is_recommended]
        formats = [f.get_text() + ("" if f.fmt.is_recommended else " [avoid]") for f in fmts]
        formats = Utils.natsort(formats, str)
        txt = bullet + bullet.join(formats)
        return f"[[ Supported formats ]]: {txt}"


@dataclass(frozen=True, repr=True, order=True)
class DfHelp:
    """
    Info on a TypedDf suitable for CLI help.
    """

    clazz: Type[AbsDf]
    formats: DfFormatsHelp

    @property
    def typing(self) -> DfTyping:
        return self.clazz.get_typing()

    def get_short_typing_text(self) -> str:
        """
        Returns 1-line text on only the required columns / structure.
        """
        raise NotImplementedError()

    def get_long_typing_text(self) -> str:
        """
        Returns multi-line text on only the required columns / structure.
        """
        raise NotImplementedError()

    def get_short_text(
        self, *, use_doc: bool = True, recommended_only: bool = False, nl: str = "\n"
    ) -> str:
        r"""
        Returns a multi-line description with compressed text.

        Args:
            use_doc: Include the docstring of the DataFrame type
            recommended_only: Only include recommended formats
            nl: Newline characters; use "\n", "\\n", or " "
        """
        t = self.get_short_typing_text()
        return (
            self.get_header_text(use_doc=use_doc)
            + ((nl + t) if len(t) > 0 else "")
            + nl
            + self.formats.get_short_text(recommended_only=recommended_only)
        ).replace(nl * 2, nl)

    def get_long_text(
        self,
        *,
        use_doc: bool = True,
        recommended_only: bool = False,
        nl: str = "\n",
        bullet: str = "- ",
        indent: str = "  ",
    ) -> str:
        r"""
        Returns a multi-line text description of the DataFrame.
        Includes its required and optional columns, and supported file formats.

        Args:
            use_doc: Include the docstring of the DataFrame type
            recommended_only: Only include recommended formats
            nl: Newline characters; use "\n", "\n\n", or " "
            bullet: Prepended to each item
            indent: Spaces for nested indent
        """
        t = self.get_long_typing_text()
        return (
            self.get_header_text(use_doc=use_doc)
            + ((nl + t) if len(t) > 0 else "")
            + nl
            + self.formats.get_long_text(
                recommended_only=recommended_only, nl=nl, bullet=bullet, indent=indent
            )
        ).replace(nl * 2, nl)

    def get_header_text(self, *, use_doc: bool = True, nl: str = "\n") -> str:
        r"""
        Returns a multi-line header of the DataFrame name and docstring.

        Args:
            use_doc: Include the docstring, as long as it is not ``None``
            nl: Newline characters; use "\n", "\n\n", or " "

        Returns:
            Something like::
                Path to a Big Table file.

                This is a big table for big things.
        """
        s = f"A {self.clazz.__name__} file."
        if use_doc and self.clazz.__doc__ is not None:
            s += nl + self.clazz.__doc__
        return s


class TypedDfHelp(DfHelp):
    def get_short_typing_text(self) -> str:
        """
        Returns a condensed text description of the required and optional columns.
        """
        t = self.typing
        req = self.get_required_cols(short=True)
        res = self.get_reserved_cols(short=True)
        s = ""
        if len(req) > 0:
            s += f"Requires columns {Utils.join_to_str(*req, last='and')}."
        if len(res) > 0:
            s += (
                (" " if len(s) > 0 else " ")
                + "Columns "
                + Utils.join_to_str(*res, last="and")
                + " are optional."
            )
        s += " "
        if t.is_strict:
            s += "More columns are ok."
        else:
            s += "No extra columns are allowed."
        return s

    def get_long_typing_text(
        self, *, nl: str = "\n", bullet: str = "- ", indent: str = "  "
    ) -> str:
        r"""
        Returns a long text description of the required and optional columns.

        Args:
            nl: Newline characters; use "\n", "\n\n", or " "
            bullet: Prepended to each item
            indent: Spaces for nested indent
        """
        bullet = nl + indent + bullet
        req = self.get_required_cols(short=False)
        res = self.get_reserved_cols(short=False)
        s = ""
        if len(req) > 0:
            s += f"[[ Required columns ]]: {bullet}{bullet.join(req)}"
        if len(res) > 0:
            if len(s) > 0:
                s += nl
            s += f"[[ Optional columns ]]: {bullet}{bullet.join(res)}"
        if len(s) == 0:
            return s
        if not self.typing.is_strict:
            s += f"{nl}Additional columns are allowed."
        else:
            s += f"{nl}No additional columns are allowed."
        return s

    def get_required_cols(self, *, short: bool = False) -> Sequence[str]:
        """
        Lists required columns and their data types.

        Args:
            short: Use shorter strings (e.g. "int" instead of "integer")
        """
        return self._cols(self.typing.required_names, short=short)

    def get_reserved_cols(self, *, short: bool = False) -> Sequence[str]:
        """
        Lists reserved (optional) columns and their data types.

        Args:
            short: Use shorter strings (e.g. "int" instead of "integer")
        """
        return self._cols(self.typing.reserved_names, short=short)

    def _cols(self, which: Sequence[str], *, short: bool) -> Sequence[str]:
        lst = []
        for c in which:
            t = self.typing.auto_dtypes.get(c)
            if t is not None:
                t = Utils.describe_dtype(t, short=short)
            if t is None:
                lst.append(c)
            else:
                lst.append(f"{c} ({t})")
        return lst


class MatrixDfHelp(DfHelp):
    def get_short_typing_text(self) -> str:
        """
        Returns a short text description of the required format for a matrix.
        """
        t = self.typing
        if t.value_dtype is None:
            s = "Matrix. "
        else:
            s = Utils.describe_dtype(t.value_dtype).capitalize()
            s += f" ({t.value_dtype.__name__}) matrix. "
        s += "List row names in the index or a special column 'row'."
        return s

    def get_long_typing_text(self, *, nl: str = " ") -> str:
        """
        Returns a long text description of the required format for a matrix.
        """
        t = self.typing
        s = "Numeric matrix with named (string-typed) rows and columns." + nl
        s += "List row names in the index or a special column 'row'."
        if t.value_dtype is not None:
            s += nl + f"Values are cast to {t.value_dtype.__name__}"
        return s


class UntypedDfHelp(DfHelp):
    def get_long_typing_text(self) -> str:
        """
        Returns ``""``.``.
        """
        return ""

    def get_short_typing_text(self) -> str:
        """
        Returns ``""``.``.
        """
        return ""


class DfCliHelp:
    @classmethod
    def help(cls, clazz: Type[AbsDf]) -> DfHelp:
        """
        Returns info suitable for CLI help.

        Display this info as the help description for an argument that's a path to a table file
        that will be read with :meth:`typeddfs.abs_dfs.AbsDf.read_file` for ``clazz``.

        Args:
            clazz: The :class:`typeddfs.typed_dfs.AbsDf` subclass
        """
        meta = clazz.get_typing()
        formats = cls.list_formats(
            flexwf_sep=meta.io.flexwf_sep,
            hdf_key=meta.io.hdf_key,
            toml_aot=meta.io.toml_aot,
        )
        if issubclass(clazz, TypedDf):
            return TypedDfHelp(clazz, DfFormatsHelp(formats))
        if issubclass(clazz, MatrixDf):
            return MatrixDfHelp(clazz, DfFormatsHelp(formats))
        return UntypedDfHelp(clazz, DfFormatsHelp(formats))

    @classmethod
    def list_formats(
        cls,
        *,
        flexwf_sep: str = _FLEXWF_SEP,
        hdf_key: str = _HDF_KEY,
        toml_aot: str = _TOML_AOT,
    ) -> DfFormatsHelp:
        """
        Lists all file formats with descriptions.

        For example, :attr:`typeddfs.file_formats.FileFormat.ods` is "OpenDocument Spreadsheet".
        """
        str_dict = dict(
            csv="Comma-delimited",
            tsv="Tab-delimited",
            json="JSON",
            xml="XML",
            feather="Feather",
            parquet="Parquet",
            xlsx="Excel Workbook",
            ods="OpenDocument Spreadsheet",
            fwf="Fixed-width",
            flexwf=f"Column-aligned (delimited by '{flexwf_sep}')",
            toml=f"TOML (Array of Tables '{toml_aot}')",
            hdf=f"HDF5 (key '{hdf_key}')",
            pickle=f"Python Pickle (v{_PICKLE_VR})",
            xls="Legacy Excel Workbook",
            xlsb="Legacy Excel Binary Workbook",
            ini="INI",
            properties=".properties",
            lines="Lines (with header)",
        )
        dct: Mapping[FileFormat, str] = {FileFormat.of(k): v for k, v in str_dict.items()}
        return DfFormatsHelp((DfFormatHelp(k, v) for k, v in dct.items()))


__all__ = ["DfFormatHelp", "DfFormatsHelp", "DfHelp", "DfCliHelp"]
