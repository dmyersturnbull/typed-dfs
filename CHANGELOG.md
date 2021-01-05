# Changelog for Typed-Dfs

Adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


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
