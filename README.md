# Typed DataFrames

[![Build status](https://img.shields.io/pypi/status/typed-dfs)](https://pypi.org/project/typed-dfs/)
[![Latest version on PyPi](https://badge.fury.io/py/typed-dfs.svg)](https://pypi.org/project/typed-dfs/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/typed-dfs.svg)](https://pypi.org/project/typed-dfs/)
[![Documentation status](https://readthedocs.org/projects/typed-dfs/badge/?version=latest&style=flat-square)](https://readthedocs.org/projects/typed-dfs)
[![Build & test](https://github.com/kokellab/typed-dfs/workflows/Build%20&%20test/badge.svg)](https://github.com/kokellab/typed-dfs/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Pandas DataFrame subclasses that enforce structure and can self-organize.
Because your functions can’t exactly accept _any_  DataFrame.
[See the docs](https://typed-dfs.readthedocs.io/en/stable/) for more information.

Simple example for a CSV like this:

| key   | value  | note |
| ----- | ------ | ---- |
| abc   | 123    | ?    |


```python
from typing import Sequence
from typeddfs import SimpleFrame, OrganizingFrame

class KeyValue(OrganizingFrame):

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ['key']

    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ['value']

    @classmethod
    def reserved_columns(cls) -> Sequence[str]:
        return ['note']

# will self-organizing and use 'key' as the index
df = KeyValue.read_csv('example.csv')
print(df.index.names, list(df.columns))  # ['key'], ['value', 'note']
```

[New issues](https://github.com/kokellab/typed-dfs/issues) and pull requests are welcome.
Please refer to the [contributing guide](https://github.com/kokellab/typed-dfs/blob/master/CONTRIBUTING.md).
Generated with [Tyrannosaurus](https://github.com/dmyersturnbull/tyrannosaurus): `tyrannosaurus new typed-dfs`.


