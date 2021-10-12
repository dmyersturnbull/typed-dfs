🗨️ Q & A
====================================

What are the different types of typed DataFrames?
#####################################################################
You should generally use two: :class:`typeddfs.typed_dfs.TypedDf`
and :class:`typeddfs.matrix_dfs.MatrixDf`.
There is also a specialized matrix type, :class:`typeddfs.matrix_dfs.AffinityMatrixDf`.
You can construct these easily with :meth:`typeddfs._entries.TypedDfs.typed`,
:meth:`typeddfs._entries.TypedDfs.matrix`, and :meth:`typeddfs._entries.TypedDfs.affinity_matrix`.
There is a final type, defined to have no typing rules, that can be constructed with
:meth:`typeddfs._entries.TypedDfs.untyped`. You can convert a vanilla Pandas DataFrame to an "untyped"
variant via :meth:`typeddfs._entries.TypedDfs.wrap` to give it the additional methods.

.. code-block:: python
    from typeddfs import TypedDfs

    MyDf = TypedDfs.typed("MyDf").build()


What is the hierarchy of DataFrames?
#####################################################################

It's confusing. In general, you won't need to know the difference.

:class:`typeddfs.typed_dfs.TypedDf`
and :class:`typeddfs.matrix_dfs.MatrixDf` inherit
from :class:`typeddfs.base_dfs.BaseDf`, which inherits from :class:`typeddfs.abs_dfs.AbsDf`,
which inherits from :class:`typeddfs._core_dfs.CoreDf`.
(Technically, ``CoreDf`` inherits from :class:`typeddfs._pretty_dfs.PrettyDf`.)
The difference is:

* :class:`typeddfs.base_dfs.BaseDf` has methods ``convert`` and ``of`` (generally overridden).
* :class:`typeddfs.abs_dfs.AbsDf` contains :meth:`typeddfs.abs_dfs.AbsDf.get_typing`,
  overrides IO methods from DataFrame,
  and adds :meth:`typeddfs.abs_dfs.AbsDf.read_file` and :meth:`typeddfs.abs_dfs.AbsDf.write_file`.
* :class:`typeddfs._core_dfs.CoreDf` wraps DataFrame methods to retain the same type for returned
  DataFrames and adds a few extra methods.


What is the difference between ``__init__``, ``convert``, and ``of``?
#####################################################################

These three methods in :class:`typeddfs.typed_dfs.TypedDf` (and its superclasses) are a bit different.
:meth:`typeddfs.typed_dfs.TypedDf.__init__` does NOT attempt to reorganize or validate your DataFrame,
while :meth:`typeddfs.typed_dfs.TypedDf.convert` and :meth:`typeddfs.typed_dfs.TypedDf.of` do.``of``
is simply more flexible than ``convert``: ``convert`` only accepts a DataFrame,
while ``of`` will take anything that ``DataFrame.__init__`` will.


When do typed DFs "detype" during chained invocations?
#####################################################################

Most DataFrame-level functions that ordinarily return DataFrames themselves
try to keep the same type.
This includes :meth:`typeddfs.abs_dfs.AbsDf.reindex`,
:meth:`typeddfs.abs_dfs.AbsDf.drop_duplicates`,
:meth:`typeddfs.abs_dfs.AbsDf.sort_values`,
and :meth:`typeddfs.abs_dfs.AbsDf.set_index`.
This is to allow for easy chained invocation, but it’s important to note
that the returned DataFrame might not conform to your requirements.
Call :meth:`typeddfs.abs_dfs.AbsDf.retype` at the end to reorganize and verify.

.. code-block:: python
    from typeddfs import TypedDfs

    MyDf = TypedDfs.typed("MyDf").require("valid").build()
    my_df = MyDf.read_csv("x.csv")
    my_df_2 = my_df.drop_duplicates().rename_cols(valid="ok")
    print(type(my_df_2))  # type(MyDf)
    # but this fails!
    my_df_3 = my_df.drop_duplicates().rename_cols(valid="ok").retype()
    # MissingColumnError "valid"

You can call :meth:`typeddfs.abs_dfs.AbsDf.dtype` to remove any typing rules
and :meth:`typeddfs.abs_dfs.AbsDf.vanilla` if you need a plain DataFrame,
though this should rarely be needed.


How does one get the typing info?
#####################################################################

Call :meth:`typeddfs.base_dfs.BaseDf.get_typing`

.. code-block:: python

    from typeddfs import TypedDfs

    MyDf = TypedDfs.typed("MyDf").require("valid").build()
    MyDf.get_typing().required_columns  # ["valid"]


How are toml documents read and written?
#####################################################################

These are limited to a single array of tables (AOT).
The AOT is named ``row`` by default (set with ``aot=``).
On read, you can pass ``aot=None`` to have it use the unique outermost key.
`

How are INI files read and written?
#####################################################################

These require exactly 2 columns after ``reset_index()``.
Parsing is purposefully minimal because these formats are flexible.
Trailing whitespace and whitespace surrounding ``=`` is ignored.
Values are not escaped, and keys may not contain ``=``.
Line continuation with ``\`` is not allowed.
Quotation marks surrounding values are not dropped,
unless ``drop_quotes=True`` is passed.
Comments begin with ``;``, along with ``#`` if ``hash_sign=True`` is passed.

On read, section names are prepended to the keys.
For example, the key name will be ``section.key`` in this example:

.. code-block:: ini

    [section]
    key = value

On write, the inverse happens.


What about .properties?
#####################################################################

These are similar to INI files.
Only hash signs are allowed for comments, and reserved chars
*are* escaped in keys.
This includes ``\\``,``\ ``, ``\=``, and ``\:`` These are not escaped in values.

What is "flex-width format"?
#####################################################################

This is a format that shows up a lot in the wild, but doesn’t seem to have a name.
It’s just a text format like TSV or CSV, but where columns are preferred to line up
in a fixed-width font. Whitespace is ignored on read, but on write the columns are made
to line up neatly. These files are easy to view.
By default, the delimiter is three vertical bars (``|||``).

When are read and write guaranteed to be inverses?
#####################################################################

In principle, this invariant holds when you call ``.strict()`` to disallow
additional columns and specify ``dtype=`` in all calls to ``.require`` and ``.reserve``.
In practice, this might break down for certain combinations of
DataFrame structure, dtypes, and serialization format.
It seems pretty solid for Feather, Parquet, and CSV/TSV-like variants,
especially if the dtypes are limited to bools, real values, int values, and strings.
There may be corner cases for XML, TOML, INI, Excel, OpenDocument, and HDF5,
as well as for categorical and miscellaneous ``object`` dtypes.

How do I include another filename suffix?
#####################################################################

Use ``.suffix()`` to register a suffix or remap it to another format.

.. code-block:: python

    from typeddfs import TypedDfs, FileFormat

    MyDf = TypedDfs.typed("MyDf").suffix(tabbed="tsv").build()
    # or:
    MyDf = TypedDfs.typed("MyDf").suffix(**{".tabbed": FileFormat.tsv}).build()


How do the checksums work?
#####################################################################

There are simple convenience flags to write sha1sum-like files while
writing files, and to verify them when reading.


.. code-block:: python

    from pathlib import Path
    from typeddfs import TypedDfs

    MyDf = TypedDfs.typed("MyDf").build()
    df = MyDf()
    df.write_file("here.csv", file_hash=True)
    # a hex-encoded hash and filename
    Path("here.csv.sha256").read_text(encoding="utf8")
    MyDf.read_file("here.csv", file_hash=True)  # verifies that it matches


You can change the hash algorithm with ``.hash()``.
The second variant is ``dir_hash``.

.. code-block:: python

    from pathlib import Path
    from typeddfs import TypedDfs, Checksums

    MyDf = TypedDfs.typed("MyDf").build()
    df = MyDf()
    path = Path("dir", "here.csv")
    df.write_file(path, dir_hash=True, mkdirs=True)
    # potentially many hex-encoded hashes and filenames; always appended to
    MyDf.read_file(path, dir_hash=True)  # verifies that it matches
    # read it
    sums = Checksums.parse_hash_file_resolved(Path("my_dir", "my_dir.sha256"))
