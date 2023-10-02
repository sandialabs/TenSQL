from typing import Optional

from .BinaryExpression import BinaryExpression
from .ABCArithmeticExpression import ABCArithmeticExpression


class ArithmeticBinaryExpression(BinaryExpression, ABCArithmeticExpression):
    valid_ops = {"+", "-", "*", "/", "//", "%"}

    def __init__(
        self,
        op: str,
        left: ABCArithmeticExpression,
        right: ABCArithmeticExpression,
        name: Optional[str] = None,
    ) -> None:
        if op not in self.valid_ops:
            raise ValueError(f"Parameter op must be one of {self.valid_ops!r}")
        super().__init__(op=op, first=left, second=right, name=name)

    @property
    def left(self) -> ABCArithmeticExpression:
        return self._first

    @property
    def right(self) -> ABCArithmeticExpression:
        return self._second


class Add(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("+", *args, **kwargs)


class Sub(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("-", *args, **kwargs)


class Div(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("/", *args, **kwargs)


class Mul(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("*", *args, **kwargs)


class QirModulus(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("%", *args, **kwargs)


class QirFloorDiv(ArithmeticBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("//", *args, **kwargs)
