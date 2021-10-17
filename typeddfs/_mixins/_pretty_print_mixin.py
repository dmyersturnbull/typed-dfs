"""
Mixin that just overrides _repr_html.
"""


class _PrettyPrintMixin:
    """
    A DataFrame with an overridden ``_repr_html_`` and some simple additional methods.
    """

    def _repr_html_(self) -> str:
        """
        Renders HTML for display() in Jupyter notebooks.
        Jupyter automatically uses this function.

        Returns:
            Just a string containing HTML, which will be wrapped in an HTML object
        """
        # noinspection PyProtectedMember
        return (
            f"<strong>{self.__class__.__name__}: {self._dims()}</strong>\n{super()._repr_html_()}"
        )

    def _dims(self) -> str:
        """
        Returns a string describing the dimensionality.

        Returns:
            A text description of the dimensions of this DataFrame
        """
        # we could handle multi-level columns
        # but they're quite rare, and the number of rows is probably obvious when looking at it
        if len(self.index.names) > 1:
            return f"{len(self)} rows × {len(self.columns)} columns, {len(self.index.names)} index columns"
        else:
            return f"{len(self)} rows × {len(self.columns)} columns"


__all__ = ["_PrettyPrintMixin"]
