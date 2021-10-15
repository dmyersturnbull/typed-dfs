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

Pandas DataFrame subclasses that self-organize and serialize robustly.

```python
Film = TypedDfs.typed("Film").require("name", "studio", "year").build()
df = Film.read_csv("file.csv")
assert df.columns.tolist() == ["name", "studio", "year"]
type(df)  # Film
```

Your types remember how to be read,
including columns, dtypes, indices, and custom requirements.
No index_cols=, header=, set_index, or astype needed.

**Read and write any format:**

```python
path = input("input file? [.csv/.tsv/.tab/.json/.xml.bz2/.feather/.snappy.h5/...]")
df = Film.read_file(path)
```

**Need dataclasses?**

```python
instances = df.to_dataclass_instances()
Film.from_dataclass_instances(instances)
```

**Save metadata?**

```python
df = df.set_attrs(dataset="piano")
df.write_file("df.csv", attrs=True)
df = Film.read_file("df.csv", attrs=True)
print(df.attrs)  # e.g. {"dataset": "piano")
```

**Make dirs? Don't overwrite?**

```python
df.write_file("df.csv", mkdirs=True, overwrite=False)
```

**Write and verify checksum?**

```python
df.write_file("df.csv", file_hash=True)
df = Film.read_file("df.csv", file_hash=True)  # fails if wrong
```

**[Read the docs 📚](https://typed-dfs.readthedocs.io/en/stable/)** for more info and examples.

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
- invalid JSON is written via the built-in json library

All standard DataFrame methods remain available.
Use `.untyped()` or `.vanilla()` if needed, and `.of(df)` for the inverse.

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

| format      | packages                     | extra     | sanity | speed | file sizes |
| ----------- | ---------------------------- | --------- | ------ | ----- | ---------- |
| Feather     | `pyarrow`                    | `feather` | +++    | ++++  | +++        |
| Parquet     | `pyarrow` or `fastparquet` † | `parquet` | ++     | +++   | ++++       |
| csv/tsv     | none                         | none      | ++     | −−    | −−         |
| flexwf ‡    | none                         | none      | ++     | −−    | −−         |
| .fwf        | none                         | none      | +      | −−    | −−         |
| json        | none                         | none      | −−     | −−−   | −−−        |
| xml         | `lxml`                       | `xml`     | −      | −−−   | −−−        |
| .properties | none                         | none      | −−     | −−    | −−         |
| toml        | `tomlkit`                    | `toml`    | −−     | −−    | −−         |
| INI         | none                         | none      | −−−    | −−    | −−         |
| .lines      | none                         | none      | ++     | −−    | −−         |
| .npy        | none                         | none      | −      | +     | +++        |
| .npz        | none                         | none      | −      | +     | +++        |
| .html       | `html5lib,beautifulsoup4`    | `html`    | −−     | −−−   | −−−        |
| pickle      | none                         | none      | −−     | −−−   | −−−        |
| XLSX        | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| ODS         | `openpyxl,defusedxml`        | `excel`   | +      | −−    | +          |
| XLS         | `openpyxl,defusedxml`        | `excel`   | −−     | −−    | +          |
| XLSB        | `pyxlsb`                     | `xlsb`    | −−     | −−    | ++         |
| HDF5        | `tables`                     | `hdf5`    | −−     | −     | ++         |

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
