from __future__ import annotations

from typing import TYPE_CHECKING

from .ABCExpression import ABCExpression

if TYPE_CHECKING:
    from .ArithmeticBinaryExpression import Add, Sub, Mul, Div
    from .ArithmeticConstant import ArithmeticConstant
    from .BooleanBinaryExpression import (
        GreaterThan,
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        Equal,
        NotEqual,
    )
    from .BooleanConstant import BooleanConstant
    from .ABCUnaryExpression import Negate, AbsoluteValue


class ABCArithmeticExpression(ABCExpression):
    def __add__(self, other: ABCArithmeticExpression) -> Add:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Add

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Add(self, other)

    def __radd__(self, other: ABCArithmeticExpression) -> Add:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Add

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Add(other, self)

    def __sub__(self, other: ABCArithmeticExpression) -> Sub:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Sub

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Sub(self, other)

    def __rsub__(self, other: ABCArithmeticExpression) -> Sub:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Sub

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Sub(other, self)

    def __mul__(self, other: ABCArithmeticExpression) -> Mul:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Mul

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Mul(self, other)

    def __rmul__(self, other: ABCArithmeticExpression) -> Mul:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Mul

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Mul(other, self)

    def __truediv__(self, other: ABCArithmeticExpression) -> Div:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Div

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Div(self, other)

    def __rtruediv__(self, other: ABCArithmeticExpression) -> Div:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Div

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Div(other, self)

    def __floordiv__(self, other: ABCArithmeticExpression) -> Div:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Div

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Div(self, other)

    def __rfloordiv__(self, other: ABCArithmeticExpression) -> Div:
        from .ArithmeticConstant import ArithmeticConstant
        from .ArithmeticBinaryExpression import Div

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Div(other, self)

    def __gt__(self, other: ABCArithmeticExpression) -> GreaterThan:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import GreaterThan

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return GreaterThan(self, other)

    def __lt__(self, other: ABCArithmeticExpression) -> LessThan:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import LessThan

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return LessThan(self, other)

    def __ge__(self, other: ABCArithmeticExpression) -> GreaterThanOrEqual:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import GreaterThanOrEqual

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return GreaterThanOrEqual(self, other)

    def __le__(self, other: ABCArithmeticExpression) -> LessThanOrEqual:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import LessThanOrEqual

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return LessThanOrEqual(self, other)

    def __pos__(self) -> ABCArithmeticExpression:
        from .ArithmeticConstant import ArithmeticConstant

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return self

    def __neg__(self) -> Negate:
        from .ArithmeticConstant import ArithmeticConstant
        from .ABCUnaryExpression import Negate

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return Negate(self)

    def __abs__(self) -> AbsoluteValue:
        from .ArithmeticConstant import ArithmeticConstant
        from .ABCUnaryExpression import AbsoluteValue

        if isinstance(other, (int, float)):
            other = ArithmeticConstant(other)
        return AbsoluteValue(self)

    def __eq__(self, other: ABCArithmeticExpression) -> Equal:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import Equal

        if isinstance(other, (int, float, str)):
            other = ArithmeticConstant(other)
        elif isinstance(other, bool):
            other = BooleanConstant(other)
        return Equal(self, other)

    def __ne__(self, other: ABCArithmeticExpression) -> NotEqual:
        from .ArithmeticConstant import ArithmeticConstant
        from .BooleanBinaryExpression import NotEqual

        if isinstance(other, (int, float, str)):
            other = ArithmeticConstant(other)
        elif isinstance(other, bool):
            other = BooleanConstant(other)
        return NotEqual(self, other)
