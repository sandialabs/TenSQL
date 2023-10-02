from typing import Optional

from .ABCUnaryExpression import ABCUnaryExpression
from .ABCArithmeticExpression import ABCArithmeticExpression
from .. import Types

class ArithmeticUnaryExpression(ABCUnaryExpression, ABCArithmeticExpression):
  pass
    
class Negate(ArithmeticUnaryExpression):
    def __init__(
        self, arg: ABCArithmeticExpression, name: Optional[str] = None
    ) -> None:
        super().__init__("-", arg, name)

    @property
    def operand(self) -> ABCArithmeticExpression:
        return self._operand

    @property
    def description(self) -> str:
        return f"-{self.operand.description}"


class AbsoluteValue(ArithmeticUnaryExpression):
    def __init__(
        self, arg: ABCArithmeticExpression, name: Optional[str] = None
    ) -> None:
        super().__init__("ABS", arg, name)

    @property
    def operand(self) -> ABCArithmeticExpression:
        return self._operand

    @property
    def description(self) -> str:
        return f"ABS({self.operand.description})"


class Ones(ArithmeticUnaryExpression):
    def __init__(self, arg: ABCArithmeticExpression, name: Optional[str] = None) -> None:
        super().__init__("ONE", arg, name)

    @property
    def operand(self) -> ABCArithmeticExpression:
        return self._operand

    @property
    def description(self) -> str:
        return f"Ones({self.operand.description})"


class Cast(ArithmeticUnaryExpression):
    def __init__(self, arg: ABCArithmeticExpression, type_: Types.Type, name: Optional[str] = None) -> None:
        self.type_ = type_
        super().__init__("CAST", arg, name)

    @property
    def operand(self) -> ABCArithmeticExpression:
        return self._operand

    @property
    def description(self) -> str:
        return f"CAST({self.operand.description} AS {type(self.type_).__name__})"
