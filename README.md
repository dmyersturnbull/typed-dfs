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


Pandas DataFrame subclasses that enforce structure and self-organize.  
*Because your functions can’t exactly accept **any**  DataFrame**.  
`pip install typeddfs[feather,fwf]`

Stop passing `index_cols=` and `header=` to `to_csv` and `read_csv`.
Your “typed” dataframes will remember how they’re supposed to be written and read.
That means columns are used for the index, string columns are always read as strings,
and custom constraints are verified.

Need to read a tab-delimited file? `read_file("myfile.tab")`.
Feather? Parquet? HDF5? .json.zip? Gzipped fixed-width?
Use `read_file`. Write a file? Use `write_file`.

Some useful extra functions, plus various Pandas issues fixed:
- `read_csv`/`to_csv`,  `read_json`/`to_json`, etc., are inverses.
  `read_file`/`write_file`, too.
- In Pandas, you can write an empty DataFrame but not read it.
  Typed-dfs will always read in what you wrote out.
- No more empty `.feather`/`.snappy`/`.h5` files written on error.
- You can write fixed-width as well as read.

```python

from typeddfs._entries import TypedDfs

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

### 🎨 More complex example

For a CSV like this:

| key   | value  | note |
| ----- | ------ | ---- |
| abc   | 123    | ?    |

```python

from typeddfs._entries import TypedDfs

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
df.to_csv("remke.csv")
df = KeyValue.read_csv("remake.csv")
print(df.index_names(), df.column_names())  # ["key"], ["value", "note"]


# And now, we can type a function to require a KeyValue,
# and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
def my_special_function(df: KeyValue) -> float:
  return KeyValue(df)["value"].sum()
```

All of the normal DataFrame methods are available.
Use `.untyped()` or `.vanilla()` to make a detyped copy that doesn’t enforce requirements.

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
- To install with support for all serialization formats,
  use `pip install typeddfs[feather] fastparquet tables`.

However, hdf5 and parquet have limited compatibility,
restricted to some platforms and Python versions.
In particular, neither is supported in Python 3.9 on Windows as of 2021-03-02.
(See the [llvmlite issue](https://github.com/numba/llvmlite/issues/669)
and [tables issue](https://github.com/PyTables/PyTables/issues/854).)

Feather offers massively better performance over CSV, gzipped CSV, and HDF5
in read speed, write speed, memory overhead, and compression ratios.
Parquet typically results in smaller file sizes than Feather at some cost in speed.
Feather is the preferred format for most cases.

**⚠ Note:** The `hdf5` and `parquet` extras are currently disabled.

| format   | packages              | extra     | compatibility | performance  |
| -------- | --------------------  | --------- | ------------- | ------------ |
| pickle   | none                  | none      | ❗ ️           | −           |
| csv      | none                  | none      | ✅             | −−          |
| json     | none                  | none      | /️            | −−-         |
| .npy †   | none                  | none      | †️            | +           |
| .npz †   | none                  | none      | †️            | +           |
| flexwf   | none                  | `fwf`     | ✅             | −−-         |
| Feather  | `pyarrow`             | `feather` | ✅             | ++++        |
| Parquet  | `pyarrow,fastparquet` | `parquet` | ❌             | +++         |
| HDF5     | `tables`              | `hdf5`    | ❌             | −           |

❗ == Pickle is explicitly not supported due to vulnerabilities and other issues.  
/ == Mostly. JSON has inconsistent handling of `None`.  
† == .npy and .npz only serialize numpy objects and therefore skip indices.  
Note: `.flexwf` is fixed-width with optional delimiters; `.fwf` is not used
to avoid a potential future conflict with `pd.DataFrame.to_fwf` (which does not exist yet).

### 📝 Extra notes

A small note of caution: [natsort](https://github.com/SethMMorton/natsort) is not pinned
to a specific major version because it receives somewhat frequent major updates.
This means that the result of typed-df’s `sort_natural` could change.
You can pin natsort to a specific major version;
e.g. `natsort = "^7"` with [Poetry](https://python-poetry.org/) or `natsort>=7,<8` with pip.

Fixed-width format is provided through Pandas `read_fwf` but can be written
via [tabulate](https://pypi.org/project/tabulate/).

### 🍁 Contributing

Typed-Dfs is licensed under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
[New issues](https://github.com/dmyersturnbull/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/dmyersturnbull/typed-dfs/blob/main/CONTRIBUTING.md).
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus).
