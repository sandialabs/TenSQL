from __future__ import annotations
from typing import Optional, Tuple, Generator

from .ABCNode import ABCNode
from .QirColumn import QirColumn
from .BooleanColumn import BooleanColumn
from .ArithmeticColumn import ArithmeticColumn
from ..Table import Table
from ..Database import Database


class GroupBy(ABCNode):
    def __init__(self, *columns: ArithmeticColumn) -> None:
        assert all(isinstance(col, ArithmeticColumn) for col in columns)
        self._columns = tuple(columns)

    @property
    def columns(self) -> Tuple[QirColumn]:
        return self._columns

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        for col in self._columns:
            yield ("column", col)
