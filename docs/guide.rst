✏️Guide
====================================

We're going to wrangle and analyze data input from a bird-watching group.

Let's just read a CSV. It looks like this::

    species,person,date,notes
    Blue Jay,Kerri Johnson,2021-05-14,perched in a tree

We'd like to declare what this should look like.

.. code-block:: python

    import typeddfs as tdf

    Sightings = (
        tdf.typed("Sightings")
        .require("species", "person", "date")
        .reserve("notes")
        .strict()
        .build()
    )


Let's try reading a malformed CSV that is missing the "date" column.

.. code-block:: python

    Sightings.read_csv("missing_col.csv")


This will raise a :class:`typeddfs.errors.MissingColumnError`.

Much more to come...

Serialization
#######################################################################

Typing rules
#######################################################################

Construction and customization
#######################################################################

New functions
#######################################################################

Natural sorting.

Matrix types
#######################################################################

Imperative declaration
#######################################################################

Data types and freezing
#######################################################################

Checksums and caching
#######################################################################

Advanced serialization
#######################################################################

Generating CLI-style help
#######################################################################

Utilities
#######################################################################
