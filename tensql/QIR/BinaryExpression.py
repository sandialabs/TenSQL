from typing import Generator, Optional, Tuple

from .ABCNode import ABCNode
from .ABCExpression import ABCExpression
from .QirColumn import QirColumn


class BinaryExpression(ABCExpression):
    def __init__(
        self,
        op: str,
        first: ABCExpression,
        second: ABCExpression,
        name: Optional[str] = None,
    ) -> None:
        self._first = first
        self._second = second
        self._op = op
        if name is None:
            name = self.description
        super().__init__(name)

    @property
    def op(self) -> str:
        return self._op

    @property
    def description(self) -> str:
        return f"({self._first.description} {self._op} {self._second.description})"

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("first", self._first)
        yield ("second", self._second)

    @property
    def columns(self) -> Generator[QirColumn, None, None]:
        yield from self._first.columns
        yield from self._second.columns
