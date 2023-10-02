from typing import Generator, Optional, Tuple

from .ABCNode import ABCNode
from .ABCExpression import ABCExpression
from .ABCArithmeticExpression import ABCArithmeticExpression
from .QirColumn import QirColumn
from .ABCBooleanExpression import ABCBooleanExpression


class ABCReductionExpression(ABCExpression):
    def __init__(
        self, op: str, operand: ABCExpression, name: Optional[str] = None
    ) -> None:
        self._operand = operand
        self._op = op
        super().__init__(name)

    @property
    def op(self):
        return self._op

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield ("operand", self._operand)

    @property
    def columns(self) -> Generator[QirColumn, None, None]:
        yield from self._operand.columns

    @property
    def operand(self) -> ABCBooleanExpression:
        return self._operand


class Any(ABCReductionExpression, ABCArithmeticExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None):
        super().__init__("ANY", arg, name)

    @property
    def description(self) -> str:
        return f"ANY({self.operand.description})"


class Sum(ABCReductionExpression, ABCArithmeticExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None):
        super().__init__("+", arg, name)

    @property
    def description(self) -> str:
        return f"SUM({self.operand.description})"


class Count(ABCReductionExpression, ABCArithmeticExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None):
        super().__init__("+", arg, name)

    @property
    def description(self) -> str:
        return f"COUNT({self.operand.description})"


class Min(ABCReductionExpression, ABCArithmeticExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None):
        super().__init__("MIN", arg, name)

    @property
    def description(self) -> str:
        return f"MIN({self.operand.description})"


class Max(ABCReductionExpression, ABCArithmeticExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None):
        super().__init__("MAX", arg, name)

    @property
    def description(self) -> str:
        return f"MAX({self.operand.description})"
