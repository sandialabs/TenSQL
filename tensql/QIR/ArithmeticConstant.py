from typing import Optional, Union

from .ABCArithmeticExpression import ABCArithmeticExpression
from .Constant import Constant


class ArithmeticConstant(Constant, ABCArithmeticExpression):
    def __init__(self, value: Union[int, float], name: Optional[str] = None) -> None:
        self._value = value
        super().__init__(value=value, name=name)

    @property
    def value(self) -> Union[int, float]:
        return self._value
