"""
Utils for getting nice CLI help on DataFrame inputs.

.. attention::
    The exact text used in this module are subject to change.

.. note ::
    Two consecutive newlines (``\\n\\n``) are used to separate sections.
    This is consistent with a number of formats, including Markdown,
    reStructuredText, and `Typer <https://github.com/tiangolo/typer>`_.
"""
from dataclasses import dataclass
from inspect import cleandoc
from typing import FrozenSet, Mapping, Sequence, Type

from typeddfs import TypedDf, MatrixDf

# noinspection PyProtectedMember
from typeddfs._utils import _FLEXWF_SEP, _HDF_KEY, _PICKLE_VR, _TOML_AOT
from typeddfs.abs_dfs import AbsDf
from typeddfs.df_typing import DfTyping
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.utils import Utils


@dataclass(frozen=True, repr=True, order=True)
class DfFormatHelp:
    """
    Help text on a specific file format.
    """

    fmt: FileFormat
    desc: str

    def get_text(self) -> str:
        """
        Returns a 1-line string of the suffixes and format description.
        """
        suffixes = Utils.natsort(self.fmt.suffixes, str)
        if self.fmt.is_text:
            suffixes = {CompressionFormat.strip_suffix(s).name for s in suffixes}
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

        Returns:
            Something like::
                .csv, .tsv/.tab, .flexwf [optionally .gz, .xz, .zip, or .bz2], .feather, .snappy, ...
        """
        fmts = [f for f in self if not recommended_only or f.fmt.is_recommended]
        text_fmts = Utils.natsort([f.get_text() for f in fmts if f.fmt.is_text], dtype=str)
        bin_fmts = Utils.natsort([f.get_text() for f in fmts if f.fmt.is_binary], dtype=str)
        txt = ""
        if len(text_fmts) > 0:
            txt += (
                ", ".join(text_fmts)
                + " [optionally "
                + ", ".join([s.name for s in CompressionFormat.list_non_empty()])
                + "]"
            )
        if len(bin_fmts) > 0:
            txt += (", " if len(text_fmts) > 0 else "") + ", ".join(bin_fmts)
        return txt

    def get_long_text(self, *, recommended_only: bool = False) -> str:
        """
        Returns a multi-line text listing of allowed file formats.

        Returns:
            Something like::
                [[ Supported formats ]]:

                .csv[.bz2/.gz/.xz/.zip]: comma-delimited

                .parquet/.snappy: Parquet

                .h5/.hdf/.hdf5: HDF5 (key 'df') [discouraged]

                .pickle/.pkl: Python Pickle [discouraged]
        """
        nl = "\n\n"
        bullet = nl + " " * 2 + "- "
        fmts = [f for f in self if not recommended_only or f.fmt.is_recommended]
        formats = [f.get_text() + ("" if f.fmt.is_recommended else " [discouraged]") for f in fmts]
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
        raise NotImplementedError()

    def get_long_typing_text(self) -> str:
        raise NotImplementedError()

    def get_short_text(self, *, use_doc: bool = True, recommended_only: bool = False) -> str:
        """
        Returns a multi-line description with compressed text.

        Args:
            use_doc: Include the docstring of the DataFrame type
            recommended_only: Only include recommended formats
        """
        nl = "\n\n"
        t = self.get_short_typing_text()
        return (
            self.get_header_text(use_doc=use_doc)
            + ((nl + t) if len(t) > 0 else "")
            + nl
            + self.formats.get_short_text()
        ).replace(nl * 2, nl)

    def get_long_text(self, *, use_doc: bool = True, recommended_only: bool = False) -> str:
        """
        Returns a multi-line text description of the DataFrame.
        Includes its required and optional columns, and supported file formats.

        Args:
            use_doc: Include the docstring of the DataFrame type
            recommended_only: Only include recommended formats
        """
        nl = "\n\n"
        t = self.get_long_typing_text()
        return (
            self.get_header_text(use_doc=use_doc)
            + ((nl + t) if len(t) > 0 else "")
            + nl
            + self.formats.get_long_text()
        ).replace(nl * 2, nl)

    def get_header_text(self, *, use_doc: bool = True) -> str:
        """
        Returns a multi-line header of the DataFrame name and docstring.

        Args:
            use_doc: Include the docstring, as long as it is not ``None``

        Returns:
            Something like::
                Path to a Big Table file.

                This is a big table for big things.
        """
        nl = "\n\n"
        s = f"A {self.clazz.__name__} file."
        if use_doc and self.clazz.__doc__ is not None:
            s += nl + self.clazz.__doc__
        return s


class TypedDfHelp(DfHelp):
    def get_short_typing_text(self) -> str:
        """
        Returns a condensed text description of the required and optional columns.
        """
        s = ""
        if len(self.required_cols) > 0:
            s += f"Requires columns: {', '.join(self.required_cols)}."
        if len(self.reserved_cols) > 0 and self.typing.is_strict:
            s += (" " if len(s) > 0 else " ") + f"Permits columns: {', '.join(self.reserved_cols)}"
        elif len(self.reserved_cols) > 0:
            s += (
                (" " if len(s) > 0 else " ")
                + f"Recognizes columns: {', '.join(self.reserved_cols)}. Additional columns are ignored."
            )
        return s

    def get_long_typing_text(self) -> str:
        """
        Returns a long text description of the required and optional columns.
        """
        nl = "\n\n"
        bullet = nl + " " * 2 + "- "
        s = ""
        if len(self.required_cols) > 0:
            s += f"[[ Required columns ]]: {bullet}{bullet.join(self.required_cols)}"
        if len(self.reserved_cols) > 0:
            if len(s) > 0:
                s += nl
            s += f"[[ Optional columns ]]: {bullet}{bullet.join(self.reserved_cols)}"
        if len(s) == 0:
            return s
        if not self.typing.is_strict:
            s += f"{nl}Additional columns are allowed."
        else:
            s += f"{nl}No additional columns are allowed."
        return s

    @property
    def required_cols(self) -> Sequence[str]:
        """
        Lists required columns and their data types.
        """
        return self._cols(self.typing.required_names)

    @property
    def reserved_cols(self) -> Sequence[str]:
        """
        Lists reserved (optional) columns and their data types.
        """
        return self._cols(self.typing.reserved_names)

    def _cols(self, which: Sequence[str]) -> Sequence[str]:
        lst = []
        for c in which:
            t = self.typing.auto_dtypes.get(c)
            t = Utils.describe_dtype(t)
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
            s = "Numeric matrix. "
        else:
            s = f"{t.value_dtype.__name__} matrix. "
        s += f"List row names in the index or a special column 'row'."
        return s

    def get_long_typing_text(self) -> str:
        """
        Returns a long text description of the required format for a matrix.
        """
        t = self.typing
        s = "Numeric matrix with named rows and columns. "
        s += f"List row names in the index or a special column 'row'. "
        if t.value_dtype is not None:
            s += f"Values are cast to {t.value_dtype.__name__}"
        return s


class UntypedDfHelp(DfHelp):
    def get_long_typing_text(self) -> str:
        """
        Returns ``""``.``
        """
        return ""

    def get_short_typing_text(self) -> str:
        """
        Returns ``""``.``
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
