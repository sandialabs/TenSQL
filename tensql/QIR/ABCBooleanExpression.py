from __future__ import annotations

from typing import TYPE_CHECKING

from .ABCExpression import ABCExpression

if TYPE_CHECKING:
    from .BooleanBinaryExpression import LogicalOr, LogicalAnd, Equal, NotEqual
    from .BooleanConstant import BooleanConstant
    from .ABCUnaryExpression import LogicalNot


class ABCBooleanExpression(ABCExpression):
    def __or__(self, other: ABCBooleanExpression) -> LogicalOr:
        from .BooleanConstant import BooleanConstant
        from .BooleanBinaryExpression import LogicalOr

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Right hand side must be a boolean expression")
        return LogicalOr(self, other)

    def __ror__(self, other: ABCBooleanExpression) -> LogicalOr:
        from .BooleanConstant import BooleanConstant
        from .BooleanBinaryExpression import LogicalOr

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Left hand side must be a boolean expression")
        return LogicalOr(other, self)

    def __and__(self, other: ABCBooleanExpression) -> LogicalAnd:
        from .BooleanConstant import BooleanConstant
        from .BooleanBinaryExpression import LogicalAnd

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Right hand side must be a boolean expression")
        return LogicalAnd(self, other)

    def __rand__(self, other: ABCBooleanExpression) -> LogicalAnd:
        from .BooleanConstant import BooleanConstant
        from .BooleanBinaryExpression import LogicalAnd

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Left hand side must be a boolean expression")
        return LogicalAnd(other, self)

    def __not__(self) -> LogicalNot:
        from .BooleanConstant import BooleanConstant
        from .ABCUnaryExpression import LogicalNot

        return LogicalNot(self)

    def __eq__(self, other: ABCBooleanExpression) -> Equal:
        from .BooleanBinaryExpression import Equal

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Left hand side must be a boolean expression")
        return Equal(self, other)

    def __ne__(self, other: ABCBooleanExpression) -> NotEqual:
        from .BooleanBinaryExpression import NotEqual

        if isinstance(other, bool):
            other = BooleanConstant(other)
        elif not isinstance(other, ABCBooleanExpression):
            raise NotImplementedError("Left hand side must be a boolean expression")
        return NotEqual(self, other)
