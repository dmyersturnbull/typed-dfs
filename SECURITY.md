# Security policy

This describes how security issues will be addressed and how they can be reported.

### 🔪 Attack surfaces

Typed-dfs will mostly share security issues with Pandas.
Serialization and IO are the major attack surface.
In particular, avoid using Pickle along with
XLSM, XLTX, XLTM, XLS, and XLSB files (which support macros).

### 💪 Hardening

You can disable reading and writing these formats with `.secure()` in the builders.
For example: `TypedDfs.typed("MyDf").secure().build()`.
This will also disable sha-1 and md5 as choices for sha1sum-style file hashes.
These files can be added with `.write_file(df, file_hash=True)`
and verified with `.read_file(path, check_hash="file")`.

### 📫 How to report

Please report security problems with the
[security issue template](https://github.com/dmyersturnbull/typed-dfs/issues/new?labels=kind%3A+security+%F0%9F%94%92&template=security.md).
If there is a remarkably significant vulnerability, please exclude details and request contact information in the issue.
