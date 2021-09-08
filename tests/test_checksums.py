import pytest

from typeddfs.checksums import Checksums


class TestBuilders:
    def test(self):
        assert Checksums.get_algorithm("sha-256") == "sha256"


if __name__ == "__main__":
    pytest.main()
