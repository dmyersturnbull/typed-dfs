# Typed DataFrames

[![Version status](https://img.shields.io/pypi/status/typeddfs?label=status)](https://pypi.org/project/typeddfs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python version compatibility](https://img.shields.io/pypi/pyversions/typeddfs?label=Python)](https://pypi.org/project/typeddfs)
[![Version on Github](https://img.shields.io/github/v/release/dmyersturnbull/typed-dfs?include_prereleases&label=GitHub)](https://github.com/dmyersturnbull/typed-dfs/releases)
[![Version on PyPi](https://img.shields.io/pypi/v/typeddfs?label=PyPi)](https://pypi.org/project/typeddfs)
[![Build (Actions)](https://img.shields.io/github/workflow/status/dmyersturnbull/typed-dfs/Build%20&%20test?label=Tests)](https://github.com/dmyersturnbull/typed-dfs/actions)
[![Documentation status](https://readthedocs.org/projects/typed-dfs/badge)](https://typed-dfs.readthedocs.io/en/stable/)
[![Coverage (coveralls)](https://coveralls.io/repos/github/dmyersturnbull/typed-dfs/badge.svg?branch=main&service=github)](https://coveralls.io/github/dmyersturnbull/typed-dfs?branch=main)
[![Maintainability](https://api.codeclimate.com/v1/badges/6b804351b6ba5e7694af/maintainability)](https://codeclimate.com/github/dmyersturnbull/typed-dfs/maintainability)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/dmyersturnbull/typed-dfs/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/dmyersturnbull/typed-dfs/?branch=main)  
[![Created with Tyrannosaurus](https://img.shields.io/badge/Created_with-Tyrannosaurus-0000ff.svg)](https://github.com/dmyersturnbull/tyrannosaurus)


Pandas DataFrame subclasses that enforce structure and can self-organize.
Because your functions can’t exactly accept _any_  DataFrame.

The subclassed DataFrames can have required and/or optional columns and indices,
and support custom requirements.
Columns are automatically turned into indices,
which means **`read_csv` and `to_csv` are always inverses**.
`MyDf.read_csv(mydf.to_csv())` is just `mydf`.

The DataFrames will display nicely in Jupyter notebooks,
and a few convenience methods are added, such as `sort_natural` and `drop_cols`.
**[See the docs](https://typed-dfs.readthedocs.io/en/stable/)** for more information.

`pip install typeddfs[hdf5]` to install.

Please note that HDF5 via pytables is 
[unsupported in Python 3.9 on Windows](https://github.com/PyTables/PyTables/issues/854)
as of 2021-02-03.

Simple example for a CSV like this:

| key   | value  | note |
| ----- | ------ | ---- |
| abc   | 123    | ?    |

```python
from typeddfs import TypedDfs

# Build me a Key-Value-Note class!
KeyValue = (
    TypedDfs.typed("KeyValue")        # typed means enforced requirements
    .require("key", dtype=str, index=True)  # automagically make this an index
    .require("value")                 # required
    .reserve("note")                  # permitted but not required
    .strict()                         # don’t allow other columns
).build()

# This will self-organize and use "key" as the index:
df = KeyValue.read_csv("example.csv")

# For fun, let"s write it and read it back:
df.to_csv("remke.csv")
df = KeyValue("remake.csv")
print(df.index_names(), df.column_names())  # ["key"], ["value", "note"]

# And now, we can type a function to require a KeyValue,
# and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
def my_special_function(df: KeyValue) -> float:
    return KeyValue(df)["value"].sum()
```

All of the normal DataFrame methods are available.
Use `.untyped()` or `.vanilla()` to make a detyped copy that doesn’t enforce requirements.

A small note of caution: [natsort](https://github.com/SethMMorton/natsort) is no longer pinned
to a specific major version as of version 0.5 because it receives somewhat frequent major updates.
This means that the result of typed-df’s `sort_natural` could change.
You can pin natsort to a specific major version; e.g. `natsort = "^7"` with [Poetry](https://python-poetry.org/).

Typed-Dfs is licensed under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
[New issues](https://github.com/dmyersturnbull/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/dmyersturnbull/typed-dfs/blob/main/CONTRIBUTING.md).  
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus).
