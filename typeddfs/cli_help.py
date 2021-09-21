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

    def get_text(self) -> str:
        """
        Returns a text listing of allowed file formats.

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
        formats = [f.get_text() + ("" if f.fmt.is_recommended else " [discouraged]") for f in self]
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

    def get_typing_text(self) -> str:
        raise NotImplementedError()

    def get_full_text(self, *, use_doc: bool = True) -> str:
        """
        Returns a multi-line text description of the DataFrame.
        Includes its required and optional columns, and supported file formats.

        Args:
            use_doc: Include the docstring of the DataFrame type
        """
        nl = "\n\n"
        t = self.get_typing_text()
        return (
            self.get_header_text(use_doc=use_doc)
            + ((nl + t) if len(t) > 0 else "")
            + nl
            + self.get_formats_text()
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
        s = f"Path to a {self.clazz.__name__} file."
        if use_doc and self.clazz.__doc__ is not None:
            s += nl + self.clazz.__doc__
        return s

    def get_formats_text(self) -> str:
        """
        Returns a text listing of allowed file formats.

        Returns:
            Something like::
                [[ Supported formats ]]:

                .csv[.bz2/.gz/.xz/.zip]: comma-delimited

                .parquet/.snappy: Parquet

                .h5/.hdf/.hdf5: HDF5 (key 'df') [discouraged]

                .pickle/.pkl: Python Pickle [discouraged]
        """
        return self.formats.get_text()


class TypedDfHelp(DfHelp):
    def get_typing_text(self) -> str:
        """
        Returns a text description of the required and optional columns.
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
        if self.typing.more_columns_allowed or self.typing.more_indices_allowed:
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
    def get_typing_text(self) -> str:
        """
        Returns a text description of the required format for a matrix.
        """
        t = self.typing
        return cleandoc(
            f"""
            Must contain string column names ("{t.column_series_name}") and row names ("{t.index_series_name}").

            Values will be cast to {self.typing.value_dtype.__name__}
            """
        )


class UntypedDfHelp(DfHelp):
    def get_typing_text(self) -> str:
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
        dct = dict(
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
        dct: Mapping[FileFormat, str] = {FileFormat.of(k): v for k, v in dct.items()}
        return DfFormatsHelp((DfFormatHelp(k, v) for k, v in dct.items()))


__all__ = ["DfFormatHelp", "DfFormatsHelp", "DfHelp", "DfCliHelp"]
