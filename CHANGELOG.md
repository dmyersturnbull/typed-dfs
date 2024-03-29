# Changelog for Typed-Dfs

Adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.17.0] - 2023-10-22

### Added

- Full support for Pandas 2
- [zstd](https://facebook.github.io/zstd/) compression
- YAML format

### Removed

- ZIP compression
- Support for Python < 3.9
- Support for Pandas < 1.5.2
- Support for Poetry < 1.4
- `MiscUtils.table_format`
- Positional args to `to_` and `read_` methods
- `cli_help`
- `example`

### Changed

- Use `import typeddfs` instead of `from typeddfs import TypedDfs`
- Excel write methods no longer use `encoding`
- Custom exceptions no longer have multiple superclasses
- `FrozeSet` now subclasses `frozenset` instead of `AbstractSet`

### Fixed

- Use poetry-core and new Poetry options
- Enable virtualenv>20.0.33
- Dropped `pd.DataFrame.append` for Pandas>=2 (it was removed)
- Added `pd.DataFrame.map` for Pandas>=2.1
- Replaced deprecated Pandas functions

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.5...v0.17.0

## [0.16.5] - 2022-02-28

No changes.

## [0.16.4] - 2021-11-06

### Added

- `MiscUtils.delete_file`
- `Checksums.delete_any`
- `Checksums.generate_dirsum`

### Fixed

- One less `.resolve` in `Checksums`
- Return a more specific exception for non-relative path
- `JsonUtils.preserve_inf` would error on some lists

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.3...v0.16.4

## [0.16.3] - 2021-11-06

### Added

- Can concatenate with `.of`
- `to=` arg to `pretty_print`
- `Utils.choose_table_format`

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.2...v0.16.3

## [0.16.2] - 2021-11-05

### Added

- `EMPTY` class attributes on froze collections
- `is_empty` and `length` on froze collections
- `SortUtils.core_natsort_flags`

### Changed

- `read_file` and `write_file` now call `.resolve` on the paths
- Renamed ExampleDataframes to ExampleDfs and LazyDataframe to LazyDf
- Returned tuples are now mainly namedtuples

### Fixed

- Compat with natsort 8 (now required)

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.1...v0.16.2

## [0.16.1] - 2021-11-01

### Added

- `datasets.py`
- `FileFormat.split` and related methods
- `AbsDf.read_url`

### Fixed

- `read_file` uses IOTyping when `attrs=None`

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.0...v0.16.1

## [0.16.0] - 2021-10-23

### Added

- Support for custom formats and per-suffix kwargs
- Support for json encoding args for .attrs

### Changed

- `Checksums`
- Simplified `.suffixes` (breaking change to an uncommon function)

### Fixed

- Duplicate IO methods
- Checksum path resolution

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.15.0...v0.16.0

## [0.15.0] - 2021-10-16

### Added

- New functions in `Checksums`
- JSON encoding functions in `JsonUtils`

### Changed

- Checksums now holds the encoding (breaking)
- `use_filename=False` by default (was `None`)
- attrs now written with `orjson` and custom default
- orjson is now required
- internal refactoring

### Fixed

- `attrs=True` less brittle
- Pandas >= 1.3 is explicitly required; this was already effectively true

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.14.2...v0.15.0

## [0.14.4] - 2021-10-12

### Added

- Top-level imports

### Changed

- Moved some docs to readthedocs

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.14.3...v0.14.4

## [0.14.3] - 2021-10-05

### Added

- Some new methods to `Checksums`
- `Checksums` functions can now accept strings

**Full Changelog**: https://github.com/dmyersturnbull/typed-dfs/compare/v0.16.2...v0.14.3

## [0.14.2] - 2021-10-05

### Fixed

- Two minor bugs

## [0.14.1] - 2021-09-23

### Changed

- Greatly improved `cli_help`
- `CoreDf.set_attrs`

## [0.14.0] - 2021-09-23

### Added

- Reading and writing dataset metadata (attrs)

### Changed

- Improved `cli_help`
- Arguments in `write_file` and `read_file` are now keyword-only
- Small parts of `Checksums`

### Fixed

- `write_file` checks for existing hashes and write permissions
  before attempting to write
- extras "all" and "main"
- docstring rst issues, esp. broken links
- metadata always passed to custom exceptions

## [0.13.3] - 2021-09-18

### Added

- Some uncommon Excel suffixes
- `cli_help`
- `FileFormat.is_recommended`
- `FileFormat.matches`
- `recommended_only()` to builders

## [0.13.1] - 2021-09-13

### Fixed

- `CompressionFormat.strip`
- Deprecation warning in test
- matrix row and column names now always typed as str

## [0.13.0] - 2021-09-11

### Added

- Preview support for .properties, INI, and TOML
- Top-level imports `typed`, `untyped`, etc.
- Q & A in the readme

### Changed

- `remap_suffixes()` renamed to `suffix()`
- `.subclass()` now supports multiple inheritance

### Removed

### Fixed

- `to_parquet()` doesn't change short to int

## [0.12.0] - 2021-09-08

### Added

- `CoreDf.strip_control_chars`
- `Utils.exact_natsort_alg`, `Utils.guess_natsort_alg`, and `Utils.all_natsort_flags`
- Functions from `pandas.api.types` to `Utils`
- `DfSupport.reload()`

### Changed

- `DfTyping` is now generic
- `FrozeSet` and `FrozeDict` are now ordered
- Moved checksum utils to `checksums.py`
- Moved `DfSupport` to `_format_support.py`
- `sort_natural` now infers the best algorithm from the data type, by default
- `drop_cols` can now accept \*args
- Split `parse_hash_file` into `parse_hash_file_resolved` and `parse_hash_file_generic`
- `regex` is now a dependency
- Hashing options in `write_file`
- Some `Utils` and `FileFormat` params are keyword-only

### Removed

- `Utils.verify_any_hash`
- Positional args from `ffill` and `bfill`

### Fixed

- Bugs in `exact_natsort_alg`
- Small bugs in `FrozeDict` and `FrozeSet`

## [0.11.0] - 2021-08-24

### Added

- Dataclass conversion
- Utils for freezing types

### Changed

- `from_records` now calls `convert`

## [0.10.0] - 2021-08-21

### Added

- Hash utils in `Utils`
- `file_hash`, `dir_hash`, and `mkdirs` to `write_file`
- `to_rst`

### Changed

- Moved DF classmethods to `DfTyping`
- DF operators now attempt to keep typing
- All MatrixDfs are now strict
- MatrixDF row and column names now must always be "row" and "column"

### Removed

- `.newline` in builder

## [0.9.0] - 2021-08-04

### Added

- `BaseDf.of` as an alias to `BaseDf.convert`
- `empty_df` methods
- `index_series_name` and `column_series_name`
- `FileFormat.strip_compression`, `FileFormat.compression_from_path`, and related

### Changed

- Index series and column series names are set to None by default in `TypedDf`
- String types are now required for column/index names in `MatrixDf`
- `MatrixDf.strict` is True by default

### Fixed

- `Utils.table_formats`
- Added tests for `symmetrize` and a few others

## [0.8.0] - 2021-08-03

### Added

- Matrix DFs
- Pickle support
- `Utils`
- `AbsDf.text_encoding`
- Extras `excel` and `xlsb`
- `AbsDf.read_html`
- To `TypedDfBuilder`: `remap_suffixes`, `encoding`, `newlines`, `subclass`, and `add_methods`

### Changed

- `TypedDf.is_valid` no longer tries to convert; it just uses the DataFrame as-is
- Text encoding is UTF-8 by default, dictated by `AbsDf.text_encoding`
- `extra_requirements` renamed to `verifications`
- `fastparquet` no longer used in `parquet` extra
- `CoreDf.transpose` now overridden and re-types.
- `read_excel` uses openpyxl by default for XLSX-like, XLS, and ODS-like (in contrast to Pandas)
- `post_processing`, `verifications`, and related functions were moved up to `BaseDf`
- Some `AbsDf` delegates to `DataFrame` now just take `*args` and `**kwargs` for simplicity.
- `tabulate` and `wcwidth` are now required dependencies.
- Optional dependency that are not used directly now have >= version ranges

### Fixed

- You can now write empty DataFrames to Feather.
- `to_excel` is much less likely to error for ODF, ODS, ODT, and XLS.
- Keyword arguments added via `write_kwargs` and `read_kwargs` no longer clash between CSV and TSV.
- Possible bugs reading and writing to fwf and flexwf (use `disable_numparse`)

### Removed

- `nl` and `bom` options. See `.newline` and `.encoding` in `TypedDfBuilder` for alternatives.
- Some deprecated options.

## [0.7.1] - 2021-07-19

### Added

- Support for `to_xml` and `read_xml`

## [0.7.0] - 2021-06-08

### Added

- `can_read` and `can_write` on `BaseDf` to get supported file formats
- Write (and read) to "flex" fixed-width;
  currently, this is only used for ".flexwf" as a preview
- `pretty_print`, which delegates to [tabulate](https://pypi.org/project/tabulate)
- Optional post-processing method (`TypedDf.post_process`)
- `known_column_names`, `known_index_names`, and `known_names`
- Methods to set default read_file/to_file args

### Removed

- All args from `read_file` and `to_file`
- `comment` from `to_lines`; it was too confusing because no other write functions had one

### Changed

- `dtype` values in `TypedDfBuilder` are now used;
  specifically, `TypedDf.convert` calls `pd.Series.astype` with them.
- Overrode `assign` to handle indices
- Split some functionality of `AbsDf` into a superclass `_CoreDf`
- Bumped pyarrow to 4.0
- Various functions return more specific error types
- Deprecated `TypedDfBuilder.condition` (renamed to `verify`)
- Passing `inplace=True` where not supported now raises an error instead of warning
- All `write_file` serialization now requires column names to be str for consistency
- Empty DataFrames are read via `BaseDf.read_csv`, etc. without issue (`pd.read_csv` normally fails)

### Fixed

- `to_lines` and `read_lines` are fully inverses
- Read/write are inverses for _untyped_ DFs for all formats
- Deleted .dockerignore and codemeta.json
- `check` workflow no longer errors on push
- Better read/write tests; enabled Parquet-format tests

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
