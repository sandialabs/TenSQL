from typing import Optional

from .BinaryExpression import BinaryExpression
from .ABCBooleanExpression import ABCBooleanExpression
from .ABCExpression import ABCExpression


class BooleanBinaryExpression(BinaryExpression, ABCBooleanExpression):
    valid_ops = {"==", "!=", ">", "<", "<=", ">=", "AND", "OR"}

    def __init__(
        self,
        op: str,
        left: ABCExpression,
        right: ABCExpression,
        name: Optional[str] = None,
    ) -> None:
        if op not in self.valid_ops:
            raise ValueError(f"Parameter op must be one of {self.valid_ops!r}")
        super().__init__(op=op, first=left, second=right, name=name)

    @property
    def left(self) -> ABCExpression:
        return self._first

    @property
    def right(self) -> ABCExpression:
        return self._second


class Equal(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("==", *args, **kwargs)


class NotEqual(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("!=", *args, **kwargs)


class GreaterThan(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(">", *args, **kwargs)


class LessThan(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("<", *args, **kwargs)


class GreaterThanOrEqual(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(">=", *args, **kwargs)


class LessThanOrEqual(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("<=", *args, **kwargs)


class LogicalOr(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("OR", *args, **kwargs)


class LogicalAnd(BooleanBinaryExpression):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("AND", *args, **kwargs)
