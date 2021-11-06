# Typed DataFrames

[![Version status](https://img.shields.io/pypi/status/typeddfs?label=status)](https://pypi.org/project/typeddfs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python version compatibility](https://img.shields.io/pypi/pyversions/typeddfs?label=Python)](https://pypi.org/project/typeddfs)
[![Version on GitHub](https://img.shields.io/github/v/release/dmyersturnbull/typed-dfs?include_prereleases&label=GitHub)](https://github.com/dmyersturnbull/typed-dfs/releases)
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
df.write_file("output.snappy")
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

**Make dirs? Don’t overwrite?**

```python
df.write_file("df.csv", mkdirs=True, overwrite=False)
```

**Write / verify checksums?**

```python
df.write_file("df.csv", file_hash=True)
df = Film.read_file("df.csv", file_hash=True)  # fails if wrong
```

**Get example datasets?**

```python
print(ExampleDfs.penguins().df)
#    species     island  bill_length_mm  ...  flipper_length_mm  body_mass_g     sex
# 0    Adelie  Torgersen            39.1  ...              181.0       3750.0    MALE
```

**Pretty-print the obvious way?**

```python
df.pretty_print(to="all_data.md.zip")
wiki_txt = df.pretty_print(fmt="mediawiki")
```

All standard DataFrame methods remain available.
Use `.of(df)` to convert to your type, or `.vanilla()` for a plain DataFrame.

**[Read the docs 📚](https://typed-dfs.readthedocs.io/en/stable/)** for more info and examples.

### 🐛 Pandas serialization bugs fixed

Pandas has several issues with serialization.

<details>
<summary><em>See: Fixed issues</em></summary>
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

</details>

### 🎁 Other features

See more in the [guided walkthrough ✏️](https://typed-dfs.readthedocs.io/en/latest/guide.html)

<details>
<summary><em>See: Short feature list</em></summary>

- Dtype-aware natural sorting
- UTF-8 by default
- Near-atomicity of read/write
- Matrix-like typed dataframes and methods (e.g. `matrix.is_symmetric()`)
- DataFrame-compatible frozen, hashable, ordered collections (dict, list, and set)
- Serialize JSON robustly, preserving NaN, inf, −inf, enums, timezones, complex numbers, etc.
- Serialize more formats like TOML and INI
- Interpreting paths and formats (e.g. `FileFormat.split("dir/myfile.csv.gz").compression # gz`)
- Generate good CLI help text for input DataFrames
- Parse/verify/add/update/delete files in a .shasum-like file

</details>

### 💔 Limitations

<details>
<summary><em>See: List of limitations</em></summary>

- Multi-level columns are not yet supported.
- Columns and index levels cannot share names.
- Duplicate column names are not supported. (These are strange anyway.)
- A typed DF cannot have columns "level_0", "index", or "Unnamed: 0".
- `inplace` is forbidden in some functions; avoid it or use `.vanilla()`.

</details>

### 🔌 Serialization support

TypedDfs provides the methods `read_file` and `write_file`, which guess the format from the
filename extension. For example, this will convert a gzipped, tab-delimited file to Feather:

```python
TastyDf = typeddfs.typed("TastyDf").build()
TastyDf.read_file("myfile.tab.gz").write_file("myfile.feather")
```

Pandas does most of the serialization, but some formats require extra packages.
Typed-dfs specifies [extras](https://python-poetry.org/docs/pyproject/#extras)
to help you get required packages and with compatible versions.

Here are the extras:

- `feather`: [Feather](https://arrow.apache.org/docs/python/feather.html) (uses: pyarrow)
- `parquet`: [Parquet (e.g. .snappy)](https://github.com/apache/parquet-format) (uses: pyarrow)
- `xml` (uses: lxml)
- `excel`: Excel and LibreOffice .xlsx/.ods/.xls, etc. (uses: openpyxl, defusedxml)
- `toml`: [TOML](https://toml.io/en/) (uses: tomlkit)
- `html` (uses: html5lib, beautifulsoup4)
- `xlsb`: rare binary Excel file (uses: pyxlsb)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) _{no extra provided}_ (_use:_ `tables`)

For example, for Feather and TOML support use: `typeddfs[feather,toml]`  
As a shorthand for all formats, use `typeddfs[all]`.

### 📊 Serialization in-depth

<details>
<summary><em>See: Full table</em></summary>

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

**⚠ Note:** The `hdf5` extra is currently disabled.

</details>

<details>
<summary><em>See: serialization notes</em></summary>

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

Feather offers massively better performance over CSV, gzipped CSV, and HDF5
in read speed, write speed, memory overhead, and compression ratios.
Parquet typically results in smaller file sizes than Feather at some cost in speed.
Feather is the preferred format for most cases.

</details>

### 🔒 Security

Refer to the [security policy](https://github.com/dmyersturnbull/typed-dfs/blob/main/SECURITY.md).

### 📝 Extra notes

<details>
<summary><em>See: Pinned versions</em></summary>

Dependencies in the extras only have version minimums, not maximums.
For example, typed-dfs requires pyarrow >= 4.
[natsort](https://github.com/SethMMorton/natsort) is also only assigned a minimum version number.
This means that the result of typed-df’s `sort_natural` could change.
To fix this, pin natsort to a specific major version;
e.g. `natsort = "^8"` with [Poetry](https://python-poetry.org/) or `natsort>=8,<9` with pip.

</details>

### 🍁 Contributing

Typed-Dfs is licensed under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
[New issues](https://github.com/dmyersturnbull/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/dmyersturnbull/typed-dfs/blob/main/CONTRIBUTING.md).
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus).
