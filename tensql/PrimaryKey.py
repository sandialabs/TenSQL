from typing import Optional

import pygraphblas.types

from . import Types
from .util import GBType


class PrimaryKey:
    def __init__(
        self,
        column_name: str,
        type_: Types = Types.BigInt(),
        max_idx: Optional[int] = None,
    ) -> None:
        self._column_name = column_name

        valid_type_default_sizes = {
            Types.TinyInt: 2**7 - 1,
            Types.SmallInt: 2**15 - 1,
            Types.Integer: 2**31 - 1,
            Types.BigInt: 2**60
        }

        self._type = type_

        if isinstance(type_, Types.TinyInt):
            type_max_idx = 2 ** 7 - 1
        elif isinstance(type_, Types.SmallInt):
            type_max_idx = 2 ** 15 - 1
        elif isinstance(type_, Types.Integer):
            type_max_idx = 2 ** 31 - 1
        elif isinstance(type_, Types.BigInt):
            type_max_idx = 2 ** 60
        else:
            raise ValueError("Argument type_ to PrimaryKey() must be an instance of Types.TinyInt, Types.SmallInt, Types.Integer, or Types.BigInt")

        if max_idx is not None:
            assert max_idx <= type_max_idx
        else:
            max_idx = type_max_idx
        self._max_idx = max_idx

    @property
    def column_name(self) -> str:
        return self._column_name

    @property
    def gbtype(self) -> GBType:
        return self._type.as_pygrb

    @property
    def type_(self) -> Types.Type:
        return self._type

    @property
    def max_idx(self) -> int:
        return self._max_idx
