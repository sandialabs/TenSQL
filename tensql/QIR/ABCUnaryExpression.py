from typing import Generator, Optional, Tuple

from .ABCNode import ABCNode

from .ABCExpression import ABCExpression
from .ABCArithmeticExpression import ABCArithmeticExpression
from .QirColumn import QirColumn
from .ABCBooleanExpression import ABCBooleanExpression
from .. import Types

class ABCUnaryExpression(ABCExpression):
    def __init__(
        self, op: str, operand: ABCExpression, name: Optional[str] = None
    ) -> None:
        self._operand = operand
        self._op = op
        super().__init__(name)

    @property
    def operand(self) -> ABCExpression:
        return self._operand

    @property
    def op(self) -> str:
        return self._op

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("operand", self._operand)

    @property
    def columns(self) -> Generator[QirColumn, None, None]:
        yield from self._operand.columns


class LogicalNot(ABCUnaryExpression):
    def __init__(self, arg: ABCBooleanExpression, name: Optional[str] = None) -> None:
        super().__init__("NOT", arg, name)

    @property
    def operand(self) -> ABCBooleanExpression:
        return self._operand

    @property
    def description(self) -> str:
        return f"NOT {self.operand.description}"


