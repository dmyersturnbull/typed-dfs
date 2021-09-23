# Typed DataFrames

[![Version status](https://img.shields.io/pypi/status/typeddfs?label=status)](https://pypi.org/project/typeddfs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python version compatibility](https://img.shields.io/pypi/pyversions/typeddfs?label=Python)](https://pypi.org/project/typeddfs)
[![Version on Github](https://img.shields.io/github/v/release/dmyersturnbull/typed-dfs?include_prereleases&label=GitHub)](https://github.com/dmyersturnbull/typed-dfs/releases)
[![Version on PyPi](https://img.shields.io/pypi/v/typeddfs?label=PyPi)](https://pypi.org/project/typeddfs)  
[![Build (Actions)](https://img.shields.io/github/workflow/status/dmyersturnbull/typed-dfs/Build%20&%20test?label=Tests)](https://github.com/dmyersturnbull/typed-dfs/actions)
[![Coverage (coveralls)](https://coveralls.io/repos/github/dmyersturnbull/typed-dfs/badge.svg?branch=main&service=github)](https://coveralls.io/github/dmyersturnbull/typed-dfs?branch=main)
[![Documentation status](https://readthedocs.org/projects/typed-dfs/badge)](https://typed-dfs.readthedocs.io/en/stable/)
[![Maintainability](https://api.codeclimate.com/v1/badges/6b804351b6ba5e7694af/maintainability)](https://codeclimate.com/github/dmyersturnbull/typed-dfs/maintainability)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/dmyersturnbull/typed-dfs/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/dmyersturnbull/typed-dfs/?branch=main)
[![Created with Tyrannosaurus](https://img.shields.io/badge/Created_with-Tyrannosaurus-0000ff.svg)](https://github.com/dmyersturnbull/tyrannosaurus)

Pandas DataFrame subclasses that self-organize and read/write correctly.

```python
Film = TypedDfs.typed("Film").require("name", "studio", "year").build()
df = Film.read_csv("file.csv")
assert df.columns.tolist() == ["name", "studio", "year"]
```

Your types will remember how they’re supposed to be read,
including dtypes, columns for set_index, and custom requirements.
Then you can stop passing index_cols=, header=, set_index, and astype each time you read.
Instead, `read_csv`, `read_parquet`, ..., will just work.

You can also document your functions clearly,
and read and write _any_ format in a single file.

```python
def hello(df: Film):
    print("read!")

df = Film.read_file(
    input("input file? [.csv/.tsv/.tab/.feather/.snappy/.json.gz/.h5/...]")
)
hello(df)
```

You can read/write TOML, INI, .properties, fixed-width format, and any compressed variants.
Need dataclasses?

```python
instances = df.to_dataclass_instances()
Film.from_dataclass_instances(instances)
```

Want to save metadata?

```python
df = df.set_attrs(timestamp=datetime.now().isoformat())
df.write_file("df.csv", attrs=True)  # saved to a corresponding metadata file
df = Film.read_file("df.csv", attrs=True)
print(df.attrs)  # e.g. {"timestamp": "2021-04-15T09:32:11Z")
```

### 🐛 Pandas serialization bugs fixed

Pandas has several issues with serialization.
Depending on the format and columns, these issues occur:

- columns being silently added or dropped,
- errors on either read or write of empty DataFrames,
- the inability to use DataFrames with indices in Feather,
- writing to Parquet failing with half-precision,
- lingering partially written files on error,
- the buggy xlrd being preferred by read_excel,
- the buggy odfpy also being preferred,
- writing a file and reading it back results in a different DataFrame,
- you can’t write fixed-width format,
- and the platform text encoding being used rather than utf-8.

### 🎁️ New methods, etc.

Docs coming soon...


### 🎨 Simple example

```python
from typeddfs import TypedDfs

MyDfType = (
    TypedDfs.typed("MyDfType")
    .require("name", index=True)  # always keep in index
    .require("value", dtype=float)  # require a column and type
    .drop("_temp")  # auto-drop a column
    .verify(lambda ddf: len(ddf) == 12)  # require exactly 12 rows
).build()

df = MyDfType.read_file(input("filename? [.feather/.csv.gz/.tsv.xz/etc.]"))
df = df.sort_natural()
df.write_file("myfile.feather", mkdirs=True)
# want to write to a sha1sum-like (.sha256) file?
df.write_file("myfile.feather", file_hash=True)
# verify it?
MyDfType.read_file("myfile.feather", check_hash="file")
```

### 📉 A matrix-style DataFrame

```python
import numpy as np
from typeddfs import TypedDfs

Symmetric64 = (
    TypedDfs.matrix("Symmetric64", doc="A symmetric float64 matrix")
    .dtype(np.float64)
    .verify(lambda df: df.values.sum().sum() == 1.0)
    .add_methods(product=lambda df: df.flatten().product())
).build()

mx = Symmetric64.read_file("input.tab")
print(mx.product())  # defined above
if mx.is_symmetric():
    mx = mx.triangle()  # it's symmetric, so we only need half
    long = mx.drop_na().long_form()  # columns: "row", 'column", and "value"
    long.write_file("long-form.xml")
```

### 🔍 More complex example

For a CSV like this:

| key | value | note |
| --- | ----- | ---- |
| abc | 123   | ?    |

```python
from typeddfs import TypedDfs

# Build me a Key-Value-Note class!
KeyValue = (
    TypedDfs.typed("KeyValue")  # With enforced reqs / typing
    .require("key", dtype=str, index=True)  # automagically add to index
    .require("value")  # required
    .reserve("note")  # permitted but not required
    .strict()  # disallow other columns
).build()

# This will self-organize and use "key" as the index:
df = KeyValue.read_csv("example.csv")

# For fun, let"s write it and read it back:
df.to_csv("remake.csv")
df = KeyValue.read_csv("remake.csv")
print(df.index_names(), df.column_names())  # ["key"], ["value", "note"]

# And now, we can type a function to require a KeyValue,
# and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
def my_special_function(df: KeyValue) -> float:
    return KeyValue(df)["value"].sum()
```

All of the normal DataFrame methods are available.
Use `.untyped()` or `.vanilla()` to make a detyped copy that doesn’t enforce requirements.
Use `.of(df)` to convert a DataFrame to your type.

### 🗨️ Q & A

**What is the difference between `__init__`, `convert`, and `of`?**

These three methods in `TypedDf` (and its superclasses) are a bit different.
`__init__` does NOT attempt to reorganize or validate your DataFrame,
while `convert` and `of` do.
`of` is simply more flexible than `convert`: `convert` only accepts a DataFrame,
while `of` will take anything that `DataFrame.__init__` will.

**When do typed DFs "detype" during chained invocations?**

Most DataFrame-level functions that ordinarily return DataFrames themselves
try to keep the same type.
This includes `reindex`, `drop_duplicates`, `sort_values`, and `set_index`.
This is to allow for easy chained invocation, but it’s important to note
that the returned DataFrame might not conform to your requirements.
Call `retype` at the end to reorganize and verify.

```python
from typeddfs import TypedDfs

MyDf = TypedDfs.typed("MyDf").require("valid").build()
my_df = MyDf.read_csv("x.csv")
my_df_2 = my_df.drop_duplicates().rename_cols(valid="ok")
print(type(my_df_2))  # type(MyDf)
# but this fails!
my_df_3 = my_df.drop_duplicates().rename_cols(valid="ok").retype()
# MissingColumnError "valid"
```

You can call `.detype()` to remove any typing rules
and `.vanilla()` if you need a plain DataFrame.

**How does one get the typing info?**

Call `.get_typing()`

```python
from typeddfs import TypedDfs

MyDf = TypedDfs.typed("MyDf").require("valid").build()
MyDf.get_typing().required_columns  # ["valid"]
```

**How are toml documents read and written?**

These are limited to a single array of tables (AOT).
The AOT is named `row` by default (set with `aot=`).
On read, you can pass `aot=None` to have it use the unique outermost key.

**How are INI files read and written?**

These require exactly 2 columns after `reset_index()`.
Parsing is purposefully minimal because these formats are flexible.
Trailing whitespace and whitespace surrounding `=` is ignored.
Values are not escaped, and keys may not contain `=`.
Line continuation with `\` is not allowed.
Quotation marks surrounding values are not dropped,
unless `drop_quotes=True` is passed.
Comments begin with `;`, along with `#` if `hash_sign=True` is passed.

On read, section names are prepended to the keys.
For example, the key name will be `section.key` in this example:

```ini
[section]
key = value
```

On write, the inverse happens.

**What about .properties?**

These are similar to INI files.
Only hash signs are allowed for comments, and reserved chars
*are* escaped in keys.
This includes `\\`,`\ `, `\=`, and `\:` These are not escaped in values.

**What is "flex-width format"?**

This is a format that shows up a lot in the wild, but doesn’t seem to have a name.
It’s just a text format like TSV or CSV, but where columns are preferred to line up
in a fixed-width font. Whitespace is ignored on read, but on write the columns are made
to line up neatly. These files are easy to view.
By default, the delimiter is three vertical bars (`|||`).

**When are read and write guaranteed to be inverses?**

In principle, this invariant holds when you call `.strict()` to disallow
additional columns and specify `dtype=` in all calls to `.require` and `.reserve`.
In practice, this might break down for certain combinations of
DataFrame structure, dtypes, and serialization format.
It seems pretty solid for Feather, Parquet, and CSV/TSV-like variants,
especially if the dtypes are limited to bools, real values, int values, and strings.
There may be corner cases for XML, TOML, INI, Excel, OpenDocument, and HDF5,
as well as for categorical and miscellaneous `object` dtypes.

**How do I include another filename suffix?**

Use `.suffix()` to register a suffix or remap it to another format.

```python
from typeddfs import TypedDfs, FileFormat

MyDf = TypedDfs.typed("MyDf").suffix(tabbed="tsv").build()
# or:
MyDf = TypedDfs.typed("MyDf").suffix(**{".tabbed": FileFormat.tsv}).build()
```

**How do the checksums work?**

There are simple convenience flags to write sha1sum-like files while
writing files, and to verify them when reading.


```python
from pathlib import Path
from typeddfs import TypedDfs

MyDf = TypedDfs.typed("MyDf").build()
df = MyDf()
df.write_file("here.csv", file_hash=True)
# a hex-encoded hash and filename
Path("here.csv.sha256").read_text(encoding="utf8")
MyDf.read_file("here.csv", file_hash=True)  # verifies that it matches
```

You can change the hash algorithm with `.hash()`.
The second variant is `dir_hash`.

```python
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
sums[path]  # return the hex hash
```


### 💔 Limitations

- Multi-level columns are not yet supported.
- Columns and index levels cannot share names.
- Duplicate column names are not supported. (These are strange anyway.)
- A typed DF cannot have columns "level_0", "index", or "Unnamed: 0".
- `inplace` is forbidden in some functions; avoid it or use `.vanilla()`.

### 🔌 Serialization support

Like Pandas, TypedDfs can read and write to various formats.
It provides the methods `read_file` and `write_file`, which guess the format from the
filename extension. For example, `df.write_file("myfile.snappy)` writes Parquet files,
and `df.write_file("myfile.tab.gz")` writes a gzipped, tab-delimited file.
The `read_file` method works the same way: `MyDf.read_file("myfile.feather")` will
read an Apache Arrow Feather file, and `MyDf.read_file("myfile.json.gzip")`reads
a gzipped JSON file. You can pass keyword arguments to those functions.

Serialization is provided through Pandas, and some formats require additional packages.
Pandas does not specify compatible versions, so typed-dfs specifies
[extras](https://python-poetry.org/docs/pyproject/#extras) are provided in typed-dfs
to ensure that those packages are installed with compatible versions.

- To install with [Feather](https://arrow.apache.org/docs/python/feather.html) support,
  use `pip install typeddfs[feather]`.
- To install with support for all formats,
  use `pip install typeddfs[all]`.

Feather offers massively better performance over CSV, gzipped CSV, and HDF5
in read speed, write speed, memory overhead, and compression ratios.
Parquet typically results in smaller file sizes than Feather at some cost in speed.
Feather is the preferred format for most cases.

### 📊 Serialization in-depth

**⚠ Note:** The `hdf5` extra is currently disabled.

| format   | packages                     | extra     | sanity | speed | file sizes |
| -------- | ---------------------------- | --------- | ------ | ----- | ---------- |
| Feather  | `pyarrow`                    | `feather` | +++    | ++++  | +++        |
| Parquet  | `pyarrow` or `fastparquet` † | `parquet` | ++     | +++   | ++++       |
| csv/tsv  | none                         | none      | ++     | −−    | −−         |
| flexwf ‡ | none                         | none      | ++     | −−    | −−         |
| .fwf     | none                         | none      | +      | −−    | −−         |
| json     | none                         | none      | −−     | −−−   | −−−        |
| xml      | `lxml`                       | `xml`     | −      | −−−   | −−−        |
| .properties | none                      | none      | −−     | −−    | −−         |
| toml     | `tomlkit`                    | `toml`    | −−     | −−    | −−         |
| INI      | none                         | none      | −−−   | −−    | −−         |
| .lines   | none                         | none      | ++     | −−    | −−         |
| .npy     | none                         | none      | −      | +     | +++        |
| .npz     | none                         | none      | −      | +     | +++        |
| .html    | `html5lib,beautifulsoup4`    | `html`    | −−     | −−−   | −−−        |
| pickle   | none                         | none      | −−     | −−−   | −−−        |
| XLSX     | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| ODS      | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| XLS      | `openpyxl,defusedxml`        | `excel`   | −−     | −−    | +          |
| XLSB     | `pyxlsb`                     | `xlsb`    | −−     | −−    | ++         |
| HDF5     | `tables`                     | `hdf5`    | −−     | −     | ++         |

**Notes:**

- † `fastparquet` can be used instead. It is slower but much smaller.
- Parquet only supports str, float64, float32, int64, int32, and bool.
  Other numeric types are automatically converted during write.
- ‡ `.flexwf` is fixed-width with optional delimiters.
- JSON has inconsistent handling of `None`. ([orjson](https://github.com/ijl/orjson) is more consistent).
- XML requires Pandas 1.3+.
- Not all JSON, XML, TOML, and HDF5 files can be read.
- .ini and .properties can only be written with exactly 2 columns + index levels:
  a key and a value. INI keys are in the form `section.name`.
- .lines can only be written with exactly 1 column or index level.
- .npy and .npz only serialize numpy objects.
  They are not supported in `read_file` and `write_file`.
- .html is not supported in `read_file` and `write_file`.
- Pickle is insecure and not recommended.
- Pandas supports odfpy for ODS and xlrd for XLS. In fact, it prefers those.
  However, they are very buggy; openpyxl is much better.
- XLSM, XLTX, XLTM, XLS, and XLSB files can contain macros, which Microsoft Excel will ingest.
- XLS is a deprecated format.
- XLSB is not fully supported in Pandas.
- HDF may not work on all platforms yet due to a
  [tables issue](https://github.com/PyTables/PyTables/issues/854).

### 🔒 Security

Refer to the [security policy](https://github.com/dmyersturnbull/typed-dfs/blob/main/SECURITY.md).

### 📝 Extra notes

Dependencies in the extras are only restricted to minimum version numbers;
libraries that use them can set their own version ranges.
For example, typed-dfs only requires tables >= 0.4, but Pandas can further restrict it.
[natsort](https://github.com/SethMMorton/natsort) is also only assigned a minimum version number;
this is because it receives frequent major version bumps.
This means that the result of typed-df’s `sort_natural` could change.
To fix this, pin natsort to a specific major version;
e.g. `natsort = "^7"` with [Poetry](https://python-poetry.org/) or `natsort>=7,<8` with pip.

### 🍁 Contributing

Typed-Dfs is licensed under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
[New issues](https://github.com/dmyersturnbull/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/dmyersturnbull/typed-dfs/blob/main/CONTRIBUTING.md).
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus).
