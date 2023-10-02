from __future__ import annotations

from typing import Generator, Tuple, TYPE_CHECKING

from .ABCNode import ABCNode
from .ABCExpression import ABCExpression
from ..Column import Column

if TYPE_CHECKING:
    from .Table import QirTable


class QirColumn(ABCExpression):
    def __init__(self, table: QirTable, name: str, column: Column) -> None:
        self._table = table
        self._column = column

        super().__init__(name)

    @property
    def table(self) -> QirTable:
        return self._table

#    @property
#    def name(self) -> str:
#        return self._name

    @property
    def raw(self) -> Column:
        return self._column

    @property
    def qualname(self) -> Tuple[str, str]:
        return (self._table.alias, self._name)
        # return f"{self._table.alias!r}.{self._name!r}"

    @property
    def is_primary_key(self) -> bool:
        for it, (col_name, col) in self._table.raw.columns:
            if it < self._table.ndim:
                if col is self._column:
                    return True
            else:
                break
        return False

    @property
    def primary_key_size(self) -> int:
        for it, (col_name, col) in self._table.raw.columns:
            if it < self._table.ndim:
                if col is self._column:
                    return self._table.shape[it]
            else:
                break
        return None

    @property
    def description(self) -> str:
        if self._alias == self._name:
            return ".".join(self.qualname)
        else:
            return ".".join(self.qualname) + " AS " + self._alias

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("table", self._table)

    @property
    def columns(self) -> Generator[QirColumn, None, None]:
        yield self
