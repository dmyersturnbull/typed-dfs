🎨 Misc examples
====================================

🎨 Simple example
####################################

.. code-block:: python

    from typeddfs import TypedDfs

    MyDfType = (
        TypedDfs.typed("MyDfType")
        .require("name", index=True)  # always keep in index
        .require("value", dtype=float)  # require a column and type
        .drop("_temp")  # auto-drop a column
        .verify(lambda ddf: len(ddf) == 12)  # require exactly 12 rows
    ).build()

    df = MyDfType.read_file(input("filename? [.feather/.csv.gz/.tsv.xz/etc.]"))
    df = df.sort_natural()
    df.write_file("myfile.feather", mkdirs=True)
    # want to write to a sha1sum-like (.sha256) file?
    df.write_file("myfile.feather", file_hash=True)
    # verify it?
    MyDfType.read_file("myfile.feather", check_hash="file")


📉 A matrix-style DataFrame
####################################

.. code-block:: python

    import numpy as np
    import typeddfs

    Symmetric64 = (
        typeddfs.matrix("Symmetric64", doc="A symmetric float64 matrix")
        .dtype(np.float64)
        .verify(lambda df: df.values.sum().sum() == 1.0)
        .add_methods(product=lambda df: df.flatten().product())
    ).build()

    mx = Symmetric64.read_file("input.tab")
    print(mx.product())  # defined above
    if mx.is_symmetric():
        mx = mx.triangle()  # it's symmetric, so we only need half
        long = mx.drop_na().long_form()  # columns: "row", 'column", and "value"
        long.write_file("long-form.xml")


🔍 Example in terms of CSV
####################################

For a CSV like this:

.. code-block::

    key,value,note
    abc,123,?


.. code-block:: python

    import typeddfs

    # Build me a Key-Value-Note class!
    KeyValue = (
        typeddfs.typed("KeyValue")  # With enforced reqs / typing
        .require("key", dtype=str, index=True)  # automagically add to index
        .require("value")  # required
        .reserve("note")  # permitted but not required
        .strict()  # disallow other columns
    ).build()

    # This will self-organize and use "key" as the index:
    df = KeyValue.read_csv("example.csv")

    # For fun, let"s write it and read it back:
    df.to_csv("remake.csv")
    df = KeyValue.read_csv("remake.csv")
    print(df.index_names(), df.column_names())  # ["key"], ["value", "note"]


    # And now, we can type a function to require a KeyValue,
    # and let it raise an `InvalidDfError` (here, a `MissingColumnError`):
    def my_special_function(df: KeyValue) -> float:
        return KeyValue(df)["value"].sum()
