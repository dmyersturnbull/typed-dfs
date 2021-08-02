"""
Exceptions used by typeddfs.
"""


class UnsupportedOperationError(Exception):
    """ """


class FilenameSuffixError(UnsupportedOperationError):
    """ """


class NonStrColumnError(UnsupportedOperationError):
    """ """


class NotSingleColumnError(UnsupportedOperationError):
    """ """


class ClashError(ValueError):
    """ """


class InvalidDfError(ValueError):
    """ """


class VerificationFailedError(InvalidDfError):
    """ """


class MissingColumnError(InvalidDfError):
    """ """


class RowColumnMismatchError(InvalidDfError):
    """ """


class UnexpectedColumnError(InvalidDfError):
    """ """


class UnexpectedIndexNameError(InvalidDfError):
    """ """


class ValueNotUniqueError(ValueError):
    """ """


class NoValueError(ValueError):
    """ """


class LengthMismatchError(ValueError):
    """ """
