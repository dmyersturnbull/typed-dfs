"""
Exceptions used by typeddfs.
"""


class UnsupportedOperationError(Exception):
    """
    Something could not be performed, in general.
    """


class FilenameSuffixError(UnsupportedOperationError):
    """
    A filename extension was not recognized.
    """


class FormatInsecureError(UnsupportedOperationError):
    """
    A requested format is less secure than required or requested.
    """


class NonStrColumnError(UnsupportedOperationError):
    """
    A column name is not a string.
    """


class NotSingleColumnError(UnsupportedOperationError):
    """
    A DataFrame needs to contain exactly 1 column.
    """


class ClashError(ValueError):
    """
    Duplicate columns or other keys were added.
    """


class InvalidDfError(ValueError):
    """
    A general typing failure of typeddfs.
    """


class VerificationFailedError(InvalidDfError):
    """
    A custom typing verification failed.
    """


class MissingColumnError(InvalidDfError):
    """
    A required column is missing.
    """


class RowColumnMismatchError(InvalidDfError):
    """
    The row and column names differ.
    """


class UnexpectedColumnError(InvalidDfError):
    """
    An extra/unrecognized column is present.
    """


class UnexpectedIndexNameError(InvalidDfError):
    """
    An extra/unrecognized index level is present.
    """


class ValueNotUniqueError(ValueError):
    """
    There is more than 1 unique value.
    """


class NoValueError(ValueError):
    """
    No value because the collection is empty.
    """


class LengthMismatchError(ValueError):
    """
    The lengths of two lists do not match.
    """


class HashError(OSError):
    """
    Something went wrong with hash file writing or reading.
    """


class HashWriteError(HashError):
    """
    Something went wrong when writing a hash file.
    """


class HashContradictsExistingError(HashWriteError, ValueError):
    """
    A hash for the filename already exists in the directory hash list, but they differ.
    """


class HashAlgorithmMissingError(HashWriteError, LookupError):
    """
    The hash algorithm was not found in :py.mod:`hashlib`.
    """


class HashVerificationError(HashError):
    """
    Something went wrong when validating a hash.
    """


class HashDidNotValidateError(HashVerificationError):
    """
    The hashes did not validate (expected != actual).
    """


class HashFileInvalidError(HashVerificationError, ValueError):
    """
    The hash file could not be parsed.
    """


class HashFileMissingError(HashVerificationError, FileNotFoundError):
    """
    The hash file does not exist.
    """


class HashFilenameMissingError(HashVerificationError, LookupError):
    """
    The filename was not found listed in the hash file.
    """


class MultipleHashFilenamesError(HashVerificationError, ValueError):
    """
    There are multiple filenames listed in the hash file where only 1 was expected.
    """


class HashFileExistsError(HashVerificationError, FileExistsError):
    """
    The hash file already exists and cannot be overwritten.
    """
