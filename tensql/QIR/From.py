from typing import Generator, Tuple

from .ABCNode import ABCNode
from .ABCTableSequence import ABCTableSequence
from .Table import QirTable
from ..Database import Database


class From(ABCTableSequence):
    def __init__(self, table: QirTable):
        super().__init__(None)
        self._table = table

    @property
    def table(self) -> QirTable:
        return self._table

    @property
    def db(self) -> Database:
        return self._table.db

    @property
    def tables(self) -> Generator[QirTable, None, None]:
        yield self._table

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("table", self._table)
