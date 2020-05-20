# Typed DataFrames

[![Latest version on PyPi](https://badge.fury.io/py/typeddfs.svg)](https://pypi.org/project/typeddfs/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/typeddfs.svg)](https://pypi.org/project/typeddfs/)
[![Documentation status](https://readthedocs.org/projects/typed-dfs/badge/?version=latest&style=flat-square)](https://readthedocs.org/projects/typed-dfs)
[![Build & test](https://github.com/dmyersturnbull/typed-dfs/workflows/Build%20&%20test/badge.svg)](https://github.com/dmyersturnbull/typed-dfs/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![Build status](https://img.shields.io/pypi/status/typeddfs)](https://pypi.org/project/typeddfs/)
[![Maintainability](https://api.codeclimate.com/v1/badges/6b804351b6ba5e7694af/maintainability)](https://codeclimate.com/github/dmyersturnbull/typed-dfs/maintainability)
[![Coverage Status](https://coveralls.io/repos/github/dmyersturnbull/typed-dfs/badge.svg?branch=master)](https://coveralls.io/github/dmyersturnbull/typed-dfs?branch=master)

Pandas DataFrame subclasses that enforce structure and can self-organize.
Because your functions can’t exactly accept _any_  DataFrame.

The subclassed DataFrames can have required and/or optional columns and indices,
and support custom requirements.
Columns are automatically turned into indices,
which means **`read_csv` and `to_csv` are always inverses**.
`MyDf.read_csv(mydf.to_csv())` is just `mydf`.

The DataFrames will display nicely in Jupyter notebooks,
and few convenience methods are added, such as `sort_natural` and `drop_cols`.
**[See the docs](https://typed-dfs.readthedocs.io/en/stable/)** for more information.

Simple example for a CSV like this:

| key   | value  | note |
| ----- | ------ | ---- |
| abc   | 123    | ?    |

```python
from typeddfs import TypedDfs

# Build me a Key-Value-Note class!
KeyValue = (
    TypedDfs.typed('KeyValue')   # typed means enforced requirements
    .require('key', index=True)  # automagically make this an index
    .require('value')            # required
    .reserve('note')             # permitted but not required
    .strict()                    # don't allow other columns
).build()

# This will self-organize and use 'key' as the index:
df = KeyValue.read_csv('example.csv')

# For fun, let's write it and read it back:
df.to_csv('remke.csv')
df = KeyValue('remake.csv')
print(df.index_names(), df.column_names())  # ['key'], ['value', 'note']

# And now, we can type a function to require a KeyValue,
# and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
def my_special_function(df: KeyValue) -> float:
    return KeyValue(df)['value'].sum()
```

All of the normal DataFrame methods are available.
Use `.untyped()` or `.vanilla()` to make a detyped copy that doesn't enforce requirements.


[New issues](https://github.com/dmyersturnbull/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/dmyersturnbull/typed-dfs/blob/master/CONTRIBUTING.md).
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus): `tyrannosaurus new typed-dfs`.
