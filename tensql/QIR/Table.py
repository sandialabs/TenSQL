from __future__ import annotations
import collections
from typing import Optional, List, Generator, Tuple

import numpy as np

from .ABCNode import ABCNode
from .QirColumn import QirColumn
from .BooleanColumn import BooleanColumn
from .ArithmeticColumn import ArithmeticColumn
from ..Table import Table
from ..Database import Database


class QirTable(ABCNode):
    def __init__(
        self, db: Database, name: str, table: Table, alias: Optional[str] = None
    ) -> None:
        self._db = db
        self._table = table
        self._name = name

        if alias is not None:
            self._alias = alias
        else:
            self._alias = name

        columns = list(self._table.columns)
        self._namedtuple = collections.namedtuple(
            self.name, 
            [c_name for c_name, c in columns],
            defaults = [None] * len(columns)
        )
        rtype = []
        for cname, c in columns:
            rtype.append((cname, c.type_.as_numpy))
        self._record_type = np.dtype(rtype)

    @property
    def db(self) -> Database:
        return self._db

    @property
    def raw(self) -> Table:
        return self._table

    @property
    def namedtuple(self):
      return self._namedtuple

    @property
    def record_type(self) -> np.dtype:
      return self._record_type

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def name(self) -> str:
        return self._name

    @property
    def primary_keys(self) -> List[QirColumn]:
        ret = []
        for it, (col_name, col) in enumerate(self._table.columns):
            if it < self._table.ndim:
                ret.append(ArithmeticColumn(self, col_name, col))
            else:
                break
        return ret

    def aliased(self, alias: str) -> QirTable:
        return QirTable(self._db, self._name, self._table, alias)

    def __getitem__(self, name: str) -> QirColumn:
        col = self._table._columns.get(name)
        if col is None or col.hidden:
            raise AttributeError()
        elif col.is_boolean:
            return BooleanColumn(self, name, col)
        else:
            return ArithmeticColumn(self, name, col)

    def __getattr__(self, name: str) -> QirColumn:
        col = self._table._columns.get(name)
        if col is None or col.hidden:
            raise AttributeError()
        elif col.is_boolean:
            return BooleanColumn(self, name, col)
        else:
            return ArithmeticColumn(self, name, col)

    def columns(self) -> Generator[QirColumn, None, None]:
        for col_name, col in self._table.columns:
            yield self[col_name]

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield from []

    @property
    def description(self) -> str:
        return self.alias
