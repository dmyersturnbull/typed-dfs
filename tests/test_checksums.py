from pathlib import Path

import pytest

from typeddfs.df_errors import HashExistsError, HashFilenameMissingError
from typeddfs.utils.checksums import ChecksumMapping, Checksums


class TestChecksums:
    def test_mapping(self):
        path = Path("my_file.txt")
        x = ChecksumMapping(Path(".") / f".{Path('').name}", {path: "aabb"})
        assert x == x
        assert x[path] == "aabb"
        assert x.line(path) == "aabb *my_file.txt"
        assert x.lines() == ["aabb *my_file.txt"]

    def test_get_algorithm(self):
        assert Checksums.resolve_algorithm("sha-256") == "sha256"

    def test_update(self):
        home = Path(__file__).parent / "resources"
        original = ChecksumMapping(
            home / ".resources", {home / Path("cat"): "0x5", home / Path("ok"): "0x63"}
        )
        update = {home / "cat": None, home / "other": "wads"}
        assert original.update(update).entries == {
            home / "ok": "0x63",
            home / "other": "wads",
        }
        assert original.remove(home / "ok").entries == {home / "cat": "0x5"}
        assert original.remove([home / "ok", home / "cat"]).entries == {}
        with pytest.raises(HashFilenameMissingError):
            original.remove("does not exist")
        assert original.remove("does not exist", missing_ok=True).entries == {
            home / "cat": "0x5",
            home / "ok": "0x63",
        }
        assert original.append({home / "yay": "hi"}).entries == {
            home / Path("cat"): "0x5",
            home / Path("ok"): "0x63",
            home / Path("yay"): "hi",
        }
        assert original.update({home / "ok": "5"}, overwrite=True).entries == {
            home / Path("cat"): "0x5",
            home / Path("ok"): "5",
        }
        assert original.update({home / "ok": "0x63"}, overwrite=None).entries == {
            home / Path("cat"): "0x5",
            home / Path("ok"): "0x63",
        }
        with pytest.raises(HashExistsError):
            original.update({home / "ok": "0x63"}, overwrite=False)
        with pytest.raises(HashExistsError):
            original.update({home / "ok": "5"}, overwrite=None)
        with pytest.raises(HashFilenameMissingError):
            original.update({home / "x": "4"}, missing_ok=False)
        assert original.update(original.get) == original

    def test_append(self):
        home = Path(__file__).parent / "resources"
        original = ChecksumMapping.new(home / ".resources", {home / "x": "0x1", home / "y": "0x2"})
        new = original.append({home / "z": "0x3"}).entries
        assert new == {home / "x": "0x1", home / "y": "0x2", home / "z": "0x3"}
        assert original.entries == {home / "x": "0x1", home / "y": "0x2"}
        with pytest.raises(HashExistsError):
            original.append({home / "x": "0x1"})

    def test_add(self):
        home = Path(__file__).parent / "resources"
        original = ChecksumMapping.new(home / ".resources", {home / "x": "0x1", home / "y": "0x2"})
        assert (original + {home / "z": "0x3"}).entries == {
            home / Path("x"): "0x1",
            home / Path("y"): "0x2",
            home / Path("z"): "0x3",
        }
        new = ChecksumMapping.new(home, {home / "z": "0x3"})
        assert (original + new).entries == {home / "x": "0x1", home / "y": "0x2", home / "z": "0x3"}
        with pytest.raises(ValueError):
            original + dict(x="aaa")

    def test_sub(self):
        home = Path(__file__).parent / "resources"
        original = ChecksumMapping.new(
            home / ".resources", {home / "x": "0x1", home / "y": "0x2", home / "z": "0x3"}
        )
        assert (original - {home / "z": "0x3"}).entries == {home / "x": "0x1", home / "y": "0x2"}
        assert (original - {home / "z"}).entries == {home / "x": "0x1", home / "y": "0x2"}
        assert (original - home / "z").entries == {home / "x": "0x1", home / "y": "0x2"}
        assert (original - {home / "z", home / "m"}).entries == {
            home / "x": "0x1",
            home / "y": "0x2",
        }

    def test_resolve(self):
        home = Path(__file__).parent / "resources"
        original = ChecksumMapping.new(home / ".resources", {home / "x": "0x1"})
        resolved = original.resolve()
        assert resolved.hash_path == (home / ".resources").resolve()
        assert resolved.entries == {home.resolve() / "x": "0x1"}
        unresolved = resolved.unresolve()
        assert unresolved == original

    def test_guess_algorithm(self):
        assert Checksums.guess_algorithm("my_file.sha256") == "sha256"
        assert Checksums.guess_algorithm("my_file.sha1") == "sha1"


if __name__ == "__main__":
    pytest.main()
