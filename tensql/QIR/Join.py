from typing import Optional, Generator, Tuple

from .ABCNode import ABCNode
from .ABCTableSequence import ABCTableSequence
from .Table import QirTable
from .ABCBooleanExpression import ABCBooleanExpression
from ..Database import Database


class Join(ABCTableSequence):
    join_types = {"inner", "outer", "left", "right"}

    def __init__(
        self,
        left: ABCTableSequence,
        right: QirTable,
        *,
        on_clause: Optional[ABCBooleanExpression],
        kind: Optional[str] = None,
    ) -> None:
        if kind is None:
            kind = "inner"

        if kind not in self.join_types:
            raise ValueError(f"Parameter kind must be one of {self.join_types!r}")

        self._left = left
        self._right = right
        self._kind = kind
        self._on_clause = on_clause

    @property
    def left(self) -> ABCTableSequence:
        return self._left

    @property
    def right(self) -> QirTable:
        return self._right

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def db(self) -> Database:
        return self._right.db

    @property
    def tables(self) -> Generator[QirTable, None, None]:
        yield from self._left.tables
        yield self._right

    @property
    def on_clause(self) -> ABCBooleanExpression:
        return self._on_clause

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("left", self._left)
        yield ("table", self._right)
        yield ("on", self._on_clause)
