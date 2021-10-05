from pathlib import Path

import pytest

from typeddfs.checksums import ChecksumMappingOpt, Checksums
from typeddfs.df_errors import (
    HashContradictsExistingError,
    HashExistsError,
    HashFilenameMissingError,
)


class TestBuilders:
    def test_get_algorithm(self):
        assert Checksums.get_algorithm("sha-256") == "sha256"

    def test_update(self):
        assert Checksums.get_updated_hashes({}, {}) == ChecksumMappingOpt({})
        original = {Path("cat"): "0x5", "ok": "0x63"}
        update = {"cat": None, "other": "wads"}
        expected = {
            Path("cat").resolve(): None,
            Path("ok").resolve(): "0x63",
            Path("other").resolve(): "wads",
        }
        assert Checksums.get_updated_hashes(original, update) == ChecksumMappingOpt(expected)
        with pytest.raises(HashExistsError):
            Checksums.get_updated_hashes({"x": "5"}, {"x": "5"}, overwrite=False)
        assert Checksums.get_updated_hashes({"x": "5"}, {"x": "5"}, overwrite=None) == {
            Path("x").resolve(): "5"
        }
        with pytest.raises(HashContradictsExistingError):
            Checksums.get_updated_hashes({"x": "5"}, {"x": "4"}, overwrite=None)
        with pytest.raises(HashFilenameMissingError):
            Checksums.get_updated_hashes({}, {"x": "4"}, missing_ok=False)


if __name__ == "__main__":
    pytest.main()
