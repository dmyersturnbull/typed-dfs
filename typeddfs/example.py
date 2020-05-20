"""
Near-replica of example from the readme.
"""

from pathlib import Path

from typeddfs import TypedDfs


def run():

    # Build me a Key-Value-Note class!
    KeyValue = TypedDfs.example()

    # This will self-organize and use 'key' as the index:
    df = KeyValue.read_csv(Path(__file__).parent / "example.csv")
    print(df.index_names(), df.column_names())  # ['key'], ['value', 'note']

    # And now, we can type a function to require a KeyValue,
    # and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
    def my_special_function(df: KeyValue) -> float:
        return KeyValue(df)["value"].sum()

    print(my_special_function(df))


__all__ = ["example", "run"]

if __name__ == "__main__":
    run()
