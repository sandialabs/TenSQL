from __future__ import annotations

import copy
from typing import Optional, Sequence, Union, Generator, Tuple, Any, TYPE_CHECKING

from .ArithmeticColumn import ArithmeticColumn

from .Query import Query
from .Table import QirTable
from .From import From
from .Join import Join
from .GroupBy import GroupBy
from .QirColumn import QirColumn
from .ABCNode import ABCNode
from .ABCTableSequence import ABCTableSequence
from .ABCBooleanExpression import ABCBooleanExpression
from .ABCExpression import ABCExpression
from .ABCBooleanExpression import ABCBooleanExpression

from ..Table import Table

if TYPE_CHECKING:
    from .ABCBooleanExpression import ABCBooleanExpression


class SelectQuery(Query, ABCTableSequence):
    def __init__(
        self,
        qirdb: Any,
        table: Union[QirTable, ABCTableSequence],
        selects: Optional[Sequence[ABCExpression]] = None,
        where: Optional[ABCBooleanExpression] = None,
        group_by: Optional[GroupBy] = None,
    ):
        self.qirdb = qirdb
        db = qirdb.db

        if isinstance(table, QirTable):
            self._table_seq = From(table)
        elif isinstance(table, ABCTableSequence):
            self._table_seq = table
        else:
            raise ValueError(
                "Parameter table must be an instance of QirTable or ABCQirTableSequence"
            )

        if selects is None:
            self._selects = []
        else:
            self._selects = [col for col in selects]

        self._where = where
        self._group_by = group_by

        super().__init__(db)

    def __copy__(self) -> SelectQuery:
        return SelectQuery(
            self.qirdb, self._table_seq, selects=self._selects, where=self._where
        )

    def copy(self) -> SelectQuery:
        return copy.copy(self)

    def tables(self) -> Generator[QirTable]:
        yield from self._table_seq.tables

    def columns(self) -> Generator[QirColumn]:
        for col in self._selects:
            yield from col._columns

    def join(
        self,
        table: QirTable,
        on_clause: ABCBooleanExpression = None,
        *,
        kind: Optional[str] = None,
    ) -> SelectQuery:
        ret = self.copy()
        ret._table_seq = Join(
            left=self._table_seq, right=table, on_clause=on_clause, kind=kind
        )
        return ret

    def select(self, *selects: ABCExpression) -> SelectQuery:
        for col in selects:
            if not isinstance(col, ABCExpression):
                raise ValueError(f"Expression is not an instance of ABCQirExpression: {col}")
        if self._selects != []:
            raise ValueError("select must only be called once on a given query")
        ret = self.copy()
        ret._selects.extend(selects)
        return ret

    def where(self, clause: ABCBooleanExpression) -> SelectQuery:
        if not isinstance(clause, ABCBooleanExpression):
            raise ValueError("Expression is not an instance of ABCQirBooleanExpression")
        if self._where is not None:
            raise ValueError("where must only be called once on a given query")
        ret = self.copy()
        ret._where = clause
        return ret

    def group_by(self, *columns: ArithmeticColumn):
        if not all(isinstance(column, ArithmeticColumn) for column in columns):
            raise ValueError("Arguments to group_by must instances of ArithmeticColumn")
        if self._group_by is not None:
            raise ValueError("group_by must only be called once on a given query")
        ret = self.copy()
        ret._group_by = GroupBy(*columns)
        return ret

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        for select in self._selects:
            yield ("output", select)
        yield ("inputs", self._table_seq)
        if self._where is not None:
            yield ("where", self._where)
        if self._group_by is not None:
            yield ("group_by", self._group_by)

    def run(self) -> Table:
        return self.qirdb.run(self)

    def run_subquery(self, alias) -> QirTable:
        result = self.qirdb.run(self)
        return QirTable(self.db, None, result, alias=alias)
