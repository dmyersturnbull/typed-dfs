# Changelog for Typed-Dfs

Adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.7.0] - 2021-06-07

### Added
- `can_read` and `can_write`
- Write (and read) to fixed-width and "flex" fixed-width
- `pretty_print`, which delegates to [tabulate](https://pypi.org/project/tabulate)

### Removed
- `comment` from `to_lines`

### Changed
- `assign` now overridden to handle indices
- Bumped pyarrow to 4.0
- All `write_file` serialization now requires column names to be str for consistency
- Empty DataFrames are read via `BaseDf.read_csv`, etc. without issue (`pd.read_csv` normally fails)

### Fixed
- `to_lines` and `read_lines` are fully inverses
- Read/write are inverses for *untyped* DFs for all formats
- Deleted .dockerignore and codemeta.json
- Check workflow error on push

## [0.6.1] - 2021-03-31

### Added
- `vanilla_reset`

### Removed
- Unused Sphinx/readthedocs files

### Fixed
- Not passing kwargs to `UntypedDf.to_csv`
- Simplified some read/write code

## [0.6.0] - 2021-03-30

## Added
- Read/write wrappers for Feather, Parquet, and JSON
- Added general functions `read_file` and `write_file`
- `TypeDfs.wrap` and `FinalDf`

### Fixed
- `to_csv` was not passing along `args` and `kwargs`
- Slightly better build config

## [0.5.0] - 2021-01-19

### Changed
- Made `tables` an optional dependency; use `typeddfs[hdf5]`
- `natsort` is no longer pinned to version 7; it's now `>=7`.
   Added a note in the readme that this just requires some caution.

### Fixed
- Slight improvement to build and metadata

## [0.4.0] - 2020-08-29

### Removed
- support for Python 3.7

#### Changed
- Bumped Pandas to 1.2
- Updated build


## [0.3.0] - 2020-08-29

### Removed:
- `require_full` argument
- support for Pandas <1.1

## Changed:
- `convert` now keeps non-reserved indices in the index as long as `more_indices_allowed` is false
- Moved builder to a separate module
- Changed or added type annotations using `__qualname__`
- Moved some basic functions from `AbsFrame` to its superclass `PrettyFrame`

## Added:
- A method on `BaseFrame` called `such_that` to do type-retaining slicing

### Fixed:
- A bug in `only`
- A bug in checking symmetry
- Dropped unnecessary imports
- Clarified that `detype` is needed for functions like `applymap` if requirements will fail the returned value
- Improved test coverage
- Added docstrings


## [0.2.0] - 2020-05-19

### Added:
- Builder and static factory for new classes
- Symmetry and custom conditions

### Changed:
- Renamed most classes
- Renamed `to_vanilla` to `vanilla`, dropping the latter
- Split code into several files


## [0.1.0] - 2020-05-12

### Added:
- Main code.
