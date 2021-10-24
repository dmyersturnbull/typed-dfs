"""
Dataclass mixin.
"""
from __future__ import annotations

from dataclasses import Field
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Type


@dataclass(frozen=True)
class TypedDfDataclass:
    """
    Just a ``dataclass`` for TypedDfs.
    Contains :meth:`get_df_type` to point to the original DataFrame.
    """

    @classmethod
    def get_fields(cls) -> Sequence[Field]:
        """
        Returns the fields of this dataclass.
        """
        return list(dataclass_fields(cls))

    @classmethod
    def get_df_type(cls) -> Type["TypedDf"]:
        """
        Returns the original DataFrame type.
        """
        raise NotImplementedError()

    def get_as_dict(self) -> Mapping[str, Any]:
        """
        Returns a mapping from the dataclass field name to the value.
        """
        return dataclass_asdict(self)


class _DataclassMixin:
    @classmethod
    def from_dataclass_instances(cls, instances: Sequence[TypedDfDataclass]) -> __qualname__:
        """
        Creates a new instance of this DataFrame type from dataclass instances.
        This mostly delegates to ``pd.DataFrame.__init__``, calling ``cls.of(instances)``.
        It is provided for consistency with :meth:`to_dataclass_instances`.

        Args:
            instances: A sequence of dataclass instances.
                       Although typed as :class:`typeddfs.abs_dfs.TypedDfDataclass`,
                       any type created by Python's ``dataclass`` module should work.

        Returns:
            A new instance of this type
        """
        if len(instances) == 0:
            return cls.new_df()
        return cls.of(instances)

    def to_dataclass_instances(self) -> Sequence[TypedDfDataclass]:
        """
        Creates a dataclass from this DataFrame and returns instances.
        Also see :meth:`from_dataclass_instances`.

        .. note ::

            Dataclass elements are equal if fields and values match,
            even if they are of different types.
            This was done by overriding ``__eq__`` to enable comparing
            results from separate calls to this method.
            Specifically, :meth:`typeddfs.abs_dfs.TypedDfDataclass.get_as_dict`
            must return True.

        .. caution ::

            Fields cannot be included if columns are not present.
            If ``self.get_typing().is_strict is False``, then the dataclass
            created by two different DataFrames of type ``self.__class__``
            may have different fields.

        .. caution ::

            A new dataclass is created per call,
            so ``df.to_dataclass_instances()[0] is not df.to_dataclass_instances()[0]``.
        """
        df = self.convert(self)
        clazz = self.__class__._create_dataclass({c: df[c].dtype for c in df.column_names()})
        instances = []
        for row in df.itertuples():
            # ignore extra columns
            # if cols are missing, let it fail on clazz.__init__
            data = {field.name: getattr(row, field.name) for field in clazz.get_fields()}
            # noinspection PyArgumentList
            instances.append(clazz(**data))
        return instances

    @classmethod
    def _create_dataclass(cls, fields: Sequence[Tuple[str, Type[Any]]]) -> Type[TypedDfDataclass]:
        clazz = make_dataclass(
            f"{cls.__name__}Dataclass",
            fields,
            bases=(TypedDfDataclass,),
            frozen=True,
            repr=True,
            unsafe_hash=True,
            order=cls.get_typing().order_dataclass,
        )

        def _get_type(x):
            return cls.__class__

        _get_type.__name__ = "get_df_type"
        clazz.get_df_type = _get_type

        # If we don't do this, then, because we create a new type each call,
        # instances will never be equal under dataclass's built-in __eq__
        def eq(self_: TypedDfDataclass, other_: TypedDfDataclass) -> bool:
            return self_.get_as_dict() == other_.get_as_dict()

        clazz.__eq__ = eq
        return clazz

    @classmethod
    def create_dataclass(cls, reserved: bool = True) -> Type[TypedDfDataclass]:
        """
        Creates a best-effort immutable ``dataclass`` for this type.
        The fields will depend on the columns and index levels present
        in :meth:`get_typing`. The type of each field will correspond to
        the specified dtype (:meth:`typeddfs.df_typing.DfTyping.auto_dtypes`),
        falling back to ``Any`` if none is specified.

        .. note ::

            If this type can support additional columns
            (:meth:`typeddfs.df_typing.DfTyping.is_strict` is the default, ``False``),
            the dataclass will not be able to support extra fields.
            For most cases, :meth:`typeddfs.abs_dfs.AbsDf.to_dataclass_instances` is better.

        Args:
            reserved: Include reserved columns and index levels

        Returns:
            A subclass of :class:`typeddfs.abs_dfs.TypedDfDataclass`
        """
        fields = [
            (field, cls.get_typing().auto_dtypes.get(field, Any))
            for field in cls.get_typing().required_names
        ]
        if reserved:
            fields += [
                (field, Optional[cls.get_typing().auto_dtypes.get(field, Any)])
                for field in [
                    *cls.get_typing().reserved_columns,
                    *cls.get_typing().reserved_index_names,
                ]
            ]
        return cls._create_dataclass(fields)


__all__ = ["_DataclassMixin", "TypedDfDataclass"]
