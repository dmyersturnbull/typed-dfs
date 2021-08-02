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

Pandas DataFrame subclasses that enforce structure, self-organize, and read/write correctly.
`pip install typeddfs`

Stop passing `index_cols=` and `header=` on read/write.
Your types will remember how they’re supposed to be read.
That means columns are used for the index, columns are given the correct types,
and constraints are verified.

As a bonus, adds clear documentation and early failure to your code.
As in, `def my_func(df: MyDataFrameType)`.
Because your functions can’t exactly accept _any_ DataFrame.

### 🎁️ Features

Need to read a tab-delimited file? `read_file("myfile.tab")`.
Feather? Parquet? HDF5? .json.zip? XML?
Use `read_file`. Write a file? Use `write_file`.
As in: `df.write_file(input("Output path?"))`.
`read_file`/`write_file`, `read_csv`/`to_csv`, `read_json`/`to_json`, `read_xml`/`to_xml`,
etc., are now inverses.

Specific issues with Pandas functions fixed, too:

- No more indices silently dropped when writing some DataFrames and formats.
- No more columns silently renamed when reading some DataFrames and formats.
- No more blank extra columns added when reading some DataFrames and formats.
- You can read empty DataFrames, just like you can write them.
- You can write an empty DataFrame to any format, not just some.
- No more empty `.feather`/`.snappy`/`.h5` files written on error.
- Have type-level defaults, instead of passing `encoding=`, `skip_blank_lines=`, etc., everywhere.

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
df.sort_natural().write_file("myfile.feather")
```

### 📉 A matrix-style DataFrame

```python
import numpy as np
from typeddfs import TypedDfs

Symmetric64 = (
    TypedDfs.matrix("Symmetric64", doc="A symmetric float64 matri")
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
Use `.convert(df)` to convert a DataFrame to your type.

### 💔 Limitations

- Multi-level columns are not yet supported.
- Duplicate column names are not supported.
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
  use `pip install typeddfs[feather,fwf,xml,xlsx,xls,ods,xlsb` and `pip install tables`.

Feather offers massively better performance over CSV, gzipped CSV, and HDF5
in read speed, write speed, memory overhead, and compression ratios.
Parquet typically results in smaller file sizes than Feather at some cost in speed.
Feather is the preferred format for most cases.

**⚠ Note:** The `hdf5` extra is currently disabled.

| format   | packages                     | extra     | sanity | speed | file sizes |
| -------- | ---------------------------- | --------- | ------ | ----- | ---------- |
| Feather  | `pyarrow`                    | `feather` | ++     | ++++  | +++        |
| Parquet  | `pyarrow` or `fastparquet` † | `parquet` | ++     | +++   | ++++       |
| csv/tsv  | none                         | none      | ++     | −−    | −−         |
| flexwf ‡ | none                         | none      | ++     | −−    | −−         |
| .fwf     | none                         | none      | +      | −−    | −−         |
| json     | none                         | none      | −−     | −−−   | −−−        |
| xml      | `lxml`                       | `xml`     | +      | −−−   | −−−        |
| .npy     | none                         | none      | ++     | +     | +++        |
| .npz     | none                         | none      | ++     | +     | +++        |
| .html    | `html5lib,beautifulsoup4`    | `html`    | −−     | −−−   | −−−        |
| pickle   | none                         | none      | −− ️   | −     | −          |
| XLSX     | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| ODS      | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| XLS      | `openpyxl,defusedxml`        | `excel`   | −−     | −−    | +          |
| XLSB     | `pyxlsb`                     | `xlsb`    | −−     | −−    | ++         |
| HDF5     | `tables`                     | `hdf5`    | −−     | −     | ++         |

**Notes:**

- † `fastparquet` can be used instead. It is slower but much smaller.
- ‡ `.flexwf` is fixed-width with optional delimiters.
- JSON has inconsistent handling of `None`. ([orjson](https://github.com/ijl/orjson) is more consistent).
- XML requires Pandas 1.3+.
- .npy and .npz only serialize numpy objects.
- .html is not supported in `read_file` and `write_file`.
- Pickle is insecure and not recommended.
- Pandas supports odfpy for ODS and xlrd for XLS. In fact, it prefers those.
  However, they are very buggy; openpyxl is much better.
- XLSM, XLTX, XLTM, XLS, and XLSB files can contain macros, which Microsoft Excel will ingest.
- XLS is a deprecated format.
- XLSB is not fully supported in Pandas.
- HDF may not work on all platforms yet due to a
  [tables issue](https://github.com/PyTables/PyTables/issues/854).

### 📝 Extra notes

Dependencies in the extras are only restricted to minimum version numbers;
libraries that use them can set their own version ranges.
For example, typed-dfs only requires pyarrow >= 0.4, but Pandas can further restrict it.
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
